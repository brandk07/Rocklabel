"""Flask app behind `rocklabel dash`.

Everything is JSON over a handful of endpoints; the page is one static bundle
that renders from them. The server holds exactly two pieces of mutable state —
the :class:`~rocklabel.dashboard.jobs.JobManager` and the memoized probes.
Outside its own ``.dashboard/`` folder it changes the project only by launching
commands, with two deliberate exceptions — ``/api/rename`` and ``/api/delete``,
the housekeeping a file manager would do, which has no CLI to defer to. Both go
through :mod:`inventory`, which only ever touches ``recordings/``, ``labels/``
and a direct child of ``datasets/``.

Binds to 127.0.0.1 by default: this thing can start processes, so it has no
business listening on the network unless you deliberately ask it to.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import webbrowser

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from . import inventory, netfix, spec, sysinfo
from .jobs import JobManager


def _resolve_binary(name: str) -> list[str]:
    """Find the console script, falling back to ``python -m``.

    Preferring the interpreter that is running the dashboard matters: launching
    the system ``rocklabel`` from inside a venv would import a different
    rocklabel than the one you are looking at.
    """
    bindir = os.path.dirname(sys.executable)
    local = os.path.join(bindir, name)
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return [local]
    found = shutil.which(name)
    if found:
        return [found]
    module = {"rocklabel": "rocklabel.cli", "rocklabel-train": "rocklabel.train.cli"}[name]
    return [sys.executable, "-m", module]


def _free_panel_port(taken: set[int]) -> int:
    """A port the live control panel can bind, skipping ones already spoken for.

    Prefers the CLI's own default so a single live job produces exactly the
    command line the drawer previewed. Two live jobs at once each need their
    own server, hence the walk. Binding to probe would race the child that is
    about to bind it for real, so this only checks what *we* handed out and
    lets a genuine collision surface in the job's log.
    """
    port = spec.PANEL_DEFAULT_PORT
    while port in taken and port < spec.PANEL_DEFAULT_PORT + 50:
        port += 1
    return port


def _safe_path(root: str, rel: str) -> str:
    """Resolve a client-supplied relative path inside the project, or 404."""
    if not rel:
        abort(400, "missing path")
    full = os.path.normpath(os.path.join(root, rel))
    if not full.startswith(os.path.normpath(root) + os.sep):
        abort(403, "path escapes the project root")
    if not os.path.exists(full):
        abort(404, rel)
    return full


def _job_points_at(job, root: str, full: str) -> bool:
    """Does a job's command line name this path, or something inside it?

    Renaming or deleting a file out from under a `generate` writing into it — or
    a cache build reading it — fails that job halfway through. Cheaper to refuse.
    """
    prefix = full + os.sep
    for arg in job.argv:
        resolved = os.path.normpath(arg if os.path.isabs(arg)
                                    else os.path.join(root, arg))
        if resolved == full or resolved.startswith(prefix):
            return True
    return False


def _write(fn, *args):
    """Run an inventory mutation, mapping its refusals onto HTTP codes.

    Everything :mod:`inventory` refuses, it refuses by raising: a path outside
    the folders it will touch is a ``ValueError`` (400), a vanished one is
    ``FileNotFoundError`` (404). Nothing here should ever surface as a 500.
    """
    try:
        return jsonify(fn(*args))
    except FileNotFoundError as e:
        abort(404, str(e))
    except ValueError as e:
        abort(400, str(e))
    except OSError as e:
        abort(400, f"{fn.__name__} failed: {e}")


def create_app(root: str) -> Flask:
    root = os.path.abspath(root)
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["JSON_SORT_KEYS"] = False
    jobs = JobManager(root)
    app.config["JOBS"] = jobs
    app.config["ROOT"] = root

    from rocklabel import __version__
    from rocklabel.live.config import AppConfig

    rig_defaults = AppConfig()

    # ------------------------------------------------------------------ page
    @app.get("/")
    def index():
        return render_template(
            "index.html",
            version=__version__,
            root=root,
            project=os.path.basename(root),
        )

    # --------------------------------------------------------------- catalog
    @app.get("/api/catalog")
    def api_catalog():
        return jsonify(spec.to_json() | {
            "version": __version__,
            "root": root,
            "sensor": {"ip": rig_defaults.source.sensor_ip,
                       "port": rig_defaults.source.udp_port},
        })

    # ----------------------------------------------------------------- state
    @app.get("/api/state")
    def api_state():
        return jsonify({
            "inventory": inventory.snapshot(root),
            "machine": sysinfo.machine(root),
            "jobs": jobs.list(),
        })

    @app.get("/api/inventory")
    def api_inventory():
        return jsonify(inventory.snapshot(root))

    @app.get("/api/machine")
    def api_machine():
        return jsonify(sysinfo.machine(root))

    @app.get("/api/torch")
    def api_torch():
        return jsonify(sysinfo.torch_status())

    @app.get("/api/sensor")
    def api_sensor():
        ip = request.args.get("ip") or rig_defaults.source.sensor_ip
        port = int(request.args.get("port") or rig_defaults.source.udp_port)
        # A running `live`/`record` job owns the UDP port; probing it would
        # steal datagrams out of the user's recording.
        busy = any(j.command_id in ("live", "record") for j in jobs.running())
        status = sysinfo.sensor_status(ip, port, rig_defaults.source.udp_bind_host,
                                       port_busy=busy)
        # Only diagnose the wired link when the sensor is not visibly working;
        # a running job's own throughput is the better signal while it holds
        # the port, and shelling out to nmcli on every 5 s poll is waste.
        streaming = status["listen"].get("streaming")
        fix = ({"supported": True, "suggested": False,
                "detail": "data is arriving — the wired link is fine"}
               if streaming or busy
               else netfix.diagnose(rig_defaults, streaming=streaming))
        return jsonify(status | {
            "interfaces": sysinfo.local_interfaces(),
            "same_subnet": sysinfo.same_subnet(ip),
            "network_fix": fix,
            "live_jobs": [j.summary() for j in jobs.running()
                          if j.command_id in ("live", "record")],
        })

    @app.get("/api/recording")
    def api_recording():
        rel = request.args.get("path", "")
        _safe_path(root, rel)
        try:
            return jsonify(inventory.recording_info(root, rel))
        except (ValueError, FileNotFoundError) as e:
            abort(404, str(e))

    @app.get("/api/figure")
    def api_figure():
        rel = request.args.get("path", "")
        full = _safe_path(root, rel)
        if not full.lower().endswith(".png"):
            abort(403, "only .png figures are served")
        return send_from_directory(os.path.dirname(full), os.path.basename(full))

    # ------------------------------------------------------- housekeeping
    # The two project writes not performed by a launched command. Both refuse
    # while a job holds the path, because a job that loses its input halfway
    # through leaves a half-written dataset behind.
    def _refuse_if_busy(paths: list[str]) -> None:
        for job in jobs.running():
            for full in paths:
                if _job_points_at(job, root, full):
                    abort(409, f"{os.path.relpath(full, root)} is in use by a "
                               f"running job ({job.title}, {job.id}) — stop it first")

    @app.post("/api/rename")
    def api_rename():
        """Rename a dataset folder, or a recording and its label file together."""
        body = request.get_json(silent=True) or {}
        rel = body.get("path", "")
        _safe_path(root, rel)
        try:
            group = inventory.rename_targets(root, rel)
        except FileNotFoundError:
            abort(404, rel)
        except ValueError as e:
            abort(400, str(e))
        _refuse_if_busy(group)
        return _write(inventory.rename, root, rel, body.get("name", ""))

    @app.post("/api/delete")
    def api_delete():
        """Delete one dataset folder, recording or label file. There is no undo.

        The page asks the user first and spells out what is about to go; this
        end of it only checks that the path is one we are willing to remove and
        that no running job is reading it.
        """
        body = request.get_json(silent=True) or {}
        rel = body.get("path", "")
        full = _safe_path(root, rel)
        _refuse_if_busy([full])
        return _write(inventory.delete, root, rel)

    # ------------------------------------------------------------------ jobs
    @app.post("/api/run")
    def api_run():
        body = request.get_json(silent=True) or {}
        cmd = spec.COMMANDS_BY_ID.get(body.get("command_id", ""))
        if cmd is None:
            abort(400, "unknown command")
        values = body.get("values") or {}
        # Panel commands get their own port so two live jobs never fight over
        # one. The child announces the URL it actually bound (see Job.panel_url)
        # — this only has to avoid handing the same number out twice.
        port = None
        if cmd.panel:
            port = _free_panel_port({j.panel_port for j in jobs.running()
                                     if j.panel_port})
        try:
            argv = spec.build_argv(cmd, values, panel_port=port)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        display = spec.quote_argv(argv)
        real = _resolve_binary(argv[0]) + argv[1:]
        job = jobs.launch(real, command_id=cmd.id, title=cmd.title, gui=cmd.gui,
                          display=display, panel_port=port)
        return jsonify({"job": job.summary()})

    @app.post("/api/preview")
    def api_preview():
        """The command line a form *would* run — the UI shows it live."""
        body = request.get_json(silent=True) or {}
        cmd = spec.COMMANDS_BY_ID.get(body.get("command_id", ""))
        if cmd is None:
            abort(400, "unknown command")
        try:
            argv = spec.build_argv(cmd, body.get("values") or {})
        except ValueError as e:
            return jsonify({"error": str(e), "command_line": ""}), 200
        return jsonify({"command_line": spec.quote_argv(argv), "error": ""})

    @app.get("/api/jobs")
    def api_jobs():
        return jsonify({"jobs": jobs.list()})

    @app.get("/api/jobs/<job_id>")
    def api_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            abort(404, job_id)
        since = int(request.args.get("since", 0))
        return jsonify(job.summary() | job.tail(since))

    @app.post("/api/jobs/<job_id>/rerun")
    def api_job_rerun(job_id: str):
        """Run a past job's command line again, as a new job.

        Refuses while the job itself is still running: every command that would
        be worth rerunning holds something exclusive — the UDP port, a GUI
        window, an output folder — and a second copy of it fights the first.
        Stop it, then rerun.
        """
        job = jobs.get(job_id)
        if job is None:
            abort(404, job_id)
        if job.status == "running":
            abort(409, f"{job_id} is still running — stop it first")
        return jsonify({"job": jobs.rerun(job).summary()})

    @app.post("/api/jobs/<job_id>/stop")
    def api_job_stop(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            abort(404, job_id)
        return jsonify({"stopped": job.stop(), "job": job.summary()})

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(409)
    def _json_error(e):
        return jsonify({"error": getattr(e, "description", str(e))}), e.code

    return app


def run_dashboard(root: str = ".", host: str = "127.0.0.1", port: int = 8765,
                  open_browser: bool = True, debug: bool = False) -> None:
    root = os.path.abspath(root)
    app = create_app(root)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"

    print(f"[rocklabel] dashboard for {root}")
    print(f"[rocklabel] serving on {url}  (Ctrl-C to stop)", flush=True)
    if host == "0.0.0.0":
        print("[rocklabel] WARNING: --host 0.0.0.0 exposes a process launcher to "
              "your network. Only do this on a trusted link.", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        app.run(host=host, port=port, debug=debug, threaded=True,
                use_reloader=False)
    finally:
        jm: JobManager = app.config["JOBS"]
        alive = jm.running()
        if alive:
            print(f"[rocklabel] stopping {len(alive)} running job(s)…", flush=True)
            jm.shutdown()


__all__ = ["create_app", "run_dashboard"]
