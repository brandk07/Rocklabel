"""Flask app behind `rocklabel live --web-ui`.

Four endpoints over a :class:`~rocklabel.live.webui.control.LiveController`:
the page, the schema it renders from, the state it polls, and the two writes.
There is no state here at all — the running pipeline is the state, and the
controller is the only thing allowed to touch it.

Runs on a daemon thread inside the live process, so a browser click reaches the
engine through a direct method call rather than any kind of IPC. That is also
why it binds ``127.0.0.1``: this thread can re-aim a LiDAR rig's scoring region
and start writing files, and has no business listening on the network.

The page borrows `rocklabel dash`'s stylesheet wholesale (see ``/theme/``)
rather than shipping a second copy of the same design tokens.
"""

from __future__ import annotations

import os
import threading
import webbrowser

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from .control import LiveController

#: `rocklabel dash` owns 8765; this is a different server in a different process.
DEFAULT_PORT = 8770


def create_app(controller: LiveController) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["JSON_SORT_KEYS"] = False
    app.config["CONTROLLER"] = controller

    from rocklabel import __version__

    dash_static = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "dashboard", "static",
    )

    @app.get("/")
    def index():
        schema = controller.schema()
        return render_template(
            "live.html",
            version=__version__,
            mode=schema["mode"],
            subtitle=schema["subtitle"],
        )

    #: Dashboard assets this page borrows rather than duplicating: the design
    #: tokens, and the chart implementation behind the trend plots. An explicit
    #: list, not an extension test — this must never become a general file
    #: server for the package directory.
    shared_assets = {"app.css", "charts.js"}

    @app.get("/theme/<path:filename>")
    def theme(filename: str):
        """Serve a dashboard asset shared with this page.

        One design system, one implementation: a second copy of app.css or
        charts.js here is how the two surfaces drift apart.
        """
        if filename not in shared_assets:
            abort(404, filename)
        return send_from_directory(dash_static, filename)

    @app.get("/api/schema")
    def api_schema():
        return jsonify(controller.schema())

    @app.get("/api/state")
    def api_state():
        return jsonify(controller.snapshot())

    @app.get("/api/scene")
    def api_scene():
        """The overhead map, its detections, and the charts' data.

        Tens of kilobytes against /api/state's few hundred, so the page polls
        this on its own slower clock.
        """
        return jsonify(controller.scene())

    @app.post("/api/set")
    def api_set():
        body = request.get_json(silent=True) or {}
        key = body.get("key", "")
        if "value" not in body:
            abort(400, "missing value")
        try:
            controller.set(key, body["value"])
        except KeyError:
            abort(404, f"unknown control: {key}")
        except ValueError as e:
            abort(400, str(e))
        # Echo the state back so the page can settle on the value that was
        # actually applied — clamped, rounded, or refused by the pipeline.
        return jsonify(controller.snapshot())

    @app.post("/api/action")
    def api_action():
        body = request.get_json(silent=True) or {}
        name = body.get("name", "")
        args = body.get("args") or {}
        if not isinstance(args, dict):
            abort(400, "args must be an object")
        try:
            controller.action(name, **args)
        except KeyError:
            abort(404, f"unknown action: {name}")
        except (ValueError, TypeError) as e:
            abort(400, str(e))
        return jsonify(controller.snapshot())

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    def _json_error(e):
        return jsonify({"error": getattr(e, "description", str(e))}), e.code

    return app


def start_server(controller: LiveController, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, open_browser: bool = True) -> str:
    """Serve the control panel on a daemon thread; returns its URL.

    Daemon because the Open3D window owns the process lifetime: closing it must
    not be blocked by a web server nobody is looking at any more.
    """
    app = create_app(controller)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{shown}:{port}/"

    def _serve() -> None:
        app.run(host=host, port=port, threaded=True, use_reloader=False,
                debug=False)

    threading.Thread(target=_serve, name="live-webui", daemon=True).start()
    if host == "0.0.0.0":
        print("[rocklabel] WARNING: --web-host 0.0.0.0 exposes the rig's "
              "controls to your network. Only do this on a trusted link.",
              flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    return url


__all__ = ["DEFAULT_PORT", "create_app", "start_server"]
