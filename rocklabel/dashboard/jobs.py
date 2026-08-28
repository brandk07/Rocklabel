"""Subprocess job manager: run a rocklabel command, stream its output, stop it.

One :class:`Job` per launched command. Output is captured line by line on a
reader thread into a bounded in-memory ring (what the browser polls) *and*
appended to a log file under ``.dashboard/logs/`` so a job's output outlives the
dashboard process.

The history outlives it too: every job is also written as a record in
``.dashboard/jobs.json``, and a fresh dashboard loads those back so closing the
browser — or the whole dashboard — does not erase what you ran. A restored job
reads its output back from its own log file the first time you open it.

Design notes:

* stdout and stderr are merged, because the CLIs interleave progress bars on
  stderr with results on stdout and splitting them scrambles the order.
* Jobs are line-buffered from the child's side via ``PYTHONUNBUFFERED``; without
  it a piped Python child buffers 8 KiB and the log looks frozen for a minute.
* GUI jobs (label, live, preview…) are perfectly normal jobs — their Open3D
  window opens on the user's desktop while their stats stream here.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

#: Lines kept in memory per job (the log file keeps everything).
MAX_LINES = 4000
#: Finished jobs retained before the oldest is dropped.
MAX_JOBS = 60
#: Name of the on-disk history, inside ``.dashboard/``.
HISTORY_FILE = "jobs.json"
#: Bumped if the record shape ever changes incompatibly; an older or unreadable
#: file is ignored rather than crashing a dashboard that is otherwise fine.
HISTORY_VERSION = 1

_JOB_ID = re.compile(r"^j(\d+)$")


def _job_number(job_id: str) -> int:
    m = _JOB_ID.match(job_id or "")
    return int(m.group(1)) if m else 0


def _still_running(pid: int | None, argv: list[str]) -> bool:
    """Is ``pid`` still the process we launched?

    Compares the live command line against the one we started, because process
    ids get reused: without the check, a job from last week could point at
    somebody's text editor. Reads /proc, so anywhere that has no /proc this
    simply answers "no" — the worst case is that a survivor is not noticed.
    """
    if not pid:
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            parts = [p.decode("utf-8", "replace") for p in fh.read().split(b"\0") if p]
    except OSError:
        return False
    return parts == list(argv)


@dataclass
class Job:
    id: str
    command_id: str
    title: str
    argv: list[str]
    cwd: str
    log_path: str
    #: What to *show* as the command line. argv[0] is resolved to an absolute
    #: interpreter/console-script path for exec; the display keeps the short
    #: `rocklabel …` spelling the user would type themselves.
    display: str = ""
    gui: bool = False
    #: Port this job's live control panel was told to bind (None: no panel).
    panel_port: int | None = None
    #: URL of that panel, latched from the child's own announcement once it is
    #: serving — see :meth:`_note_panel`. None until then, which is exactly the
    #: readiness signal the dashboard needs before it embeds the page.
    panel_url: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    returncode: int | None = None
    #: running | ok | failed | stopped | interrupted. "interrupted" only ever
    #: comes from the history file: a job that was still going when the
    #: dashboard it belonged to went away.
    status: str = "running"
    #: Process id of the child, kept so a job loaded back from the history can
    #: still be checked on (and stopped) if it outlived its dashboard.
    pid: int | None = None
    #: True for a job read back from ``.dashboard/jobs.json`` rather than
    #: launched by this dashboard.
    restored: bool = False
    lines: deque = field(default_factory=lambda: deque(maxlen=MAX_LINES))
    _seq: int = 0                  # total lines ever emitted (cursor for polling)
    _proc: subprocess.Popen | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    #: Set to True once a restored job's log file has been read back in. A live
    #: job has nothing to read back, so it starts loaded.
    _log_loaded: bool = True
    #: Called by the job whenever something worth persisting changes.
    _notify: Callable[[], None] | None = None

    # -------------------------------------------------------------- accessors
    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    @property
    def command_line(self) -> str:
        return self.display or " ".join(shlex.quote(a) for a in self.argv)

    @property
    def orphan(self) -> bool:
        """A job from an earlier dashboard whose process is *still going*.

        Closing the dashboard with Ctrl-C stops its jobs, but a hard kill (the
        terminal window closing, a reboot cut short) leaves them running. They
        show up in the history as interrupted, and this is how the rest of the
        dashboard knows one is still holding the sensor port or the GPU.
        """
        return (self.restored and self.status == "interrupted"
                and _still_running(self.pid, self.argv))

    @property
    def alive(self) -> bool:
        proc = self._proc
        if proc is not None:
            return proc.poll() is None
        return self.orphan

    def summary(self) -> dict:
        return {
            "id": self.id,
            "command_id": self.command_id,
            "title": self.title,
            "command_line": self.command_line,
            "status": self.status,
            "gui": self.gui,
            "returncode": self.returncode,
            "started": self.started,
            "finished": self.finished,
            "elapsed": round(self.elapsed, 1),
            "line_count": self._seq,
            "panel_url": self.panel_url,
            "restored": self.restored,
            "orphan": self.orphan,
        }

    def record(self) -> dict:
        """What gets written to the history file.

        Deliberately not the output: that is already on disk in the log file,
        and re-storing thousands of lines as JSON would make the history file
        enormous for no gain.
        """
        return {
            "id": self.id,
            "command_id": self.command_id,
            "title": self.title,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "log_path": self.log_path,
            "display": self.display,
            "gui": self.gui,
            "panel_port": self.panel_port,
            "started": self.started,
            "finished": self.finished,
            "returncode": self.returncode,
            "status": self.status,
            "line_count": self._seq,
            "pid": self.pid,
        }

    @classmethod
    def from_record(cls, rec: dict) -> "Job":
        """Rebuild a job from the history file.

        Anything that was still running belongs to a dashboard that no longer
        exists, so it is marked interrupted here — whether its process actually
        survived is a separate question, answered by :attr:`orphan`.
        """
        status = rec["status"]
        job = cls(
            id=str(rec["id"]),
            command_id=rec.get("command_id", ""),
            title=rec.get("title", rec.get("command_id", "job")),
            argv=list(rec.get("argv") or []),
            cwd=rec.get("cwd", ""),
            log_path=rec.get("log_path", ""),
            display=rec.get("display", ""),
            gui=bool(rec.get("gui")),
            panel_port=rec.get("panel_port"),
            started=float(rec.get("started") or 0.0),
            finished=rec.get("finished"),
            returncode=rec.get("returncode"),
            status="interrupted" if status == "running" else status,
            pid=rec.get("pid"),
            restored=True,
        )
        # An interrupted job has no honest end time; without one, "elapsed"
        # would tick upwards forever as if it were still going.
        if job.finished is None and not job.orphan:
            job.finished = job.started
        job._seq = int(rec.get("line_count") or 0)
        job._log_loaded = False
        return job

    def tail(self, since: int = 0) -> dict:
        """Lines emitted after cursor ``since``, plus the new cursor.

        The ring only holds the last ``MAX_LINES``; if the caller has fallen
        further behind than that we hand back what we have and say so, rather
        than silently skipping lines.
        """
        if not self._log_loaded:
            self._load_log()
        with self._lock:
            have = len(self.lines)
            first = self._seq - have          # cursor of lines[0]
            if since < first:
                out = list(self.lines)
                truncated = since > 0
                start = first
            else:
                out = list(self.lines)[since - first:]
                truncated = False
                start = since
            return {
                "lines": out,
                "cursor": self._seq,
                "dropped": start - since if truncated else 0,
                "status": self.status,
                "returncode": self.returncode,
                "elapsed": round(self.elapsed, 1),
            }

    # -------------------------------------------------------------- lifecycle
    def _load_log(self) -> None:
        """Read a restored job's output back from its log file, once.

        Done on demand rather than at startup: sixty jobs' worth of training
        logs is hundreds of megabytes, and the user only ever looks at one of
        them. The ring keeps the last ``MAX_LINES``, exactly as a live job does.
        """
        with self._lock:
            if self._log_loaded:
                return
            self._log_loaded = True
            total = 0
            try:
                with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        self.lines.append(line.rstrip("\n"))
                        total += 1
            except OSError as e:
                self.lines.append(f"[dashboard] the log file for this job is gone: {e}")
                total += 1
            self.lines.append(self._epitaph())
            self._seq = total + 1

    def _epitaph(self) -> str:
        """The closing line a restored job shows, since the log has no ending.

        The "exited with code" line is emitted after the log file is closed, so
        a job read back from disk stops mid-output without it.
        """
        if self.status != "interrupted":
            return (f"[dashboard] exited with code {self.returncode} "
                    f"after {self.elapsed:.1f}s")
        if self.orphan:
            return (f"[dashboard] the dashboard closed while this was running. "
                    f"The process (pid {self.pid}) is still going, but nothing is "
                    f"capturing its output any more — Stop still works.")
        return "[dashboard] the dashboard closed before this job finished."

    def _emit(self, text: str) -> None:
        with self._lock:
            self.lines.append(text)
            self._seq += 1
        if self.panel_port and self.panel_url is None:
            self._note_panel(text)

    #: What `rocklabel live --web-ui` prints once its server is accepting
    #: connections. Taking the URL from the child rather than assembling it
    #: from panel_port means the dashboard embeds what is actually serving —
    #: and not one instant before it is.
    _PANEL_MARKER = "control panel: http"

    def _note_panel(self, line: str) -> None:
        if self._PANEL_MARKER not in line:
            return
        # Take only the first token after the marker. stdout and stderr are
        # merged (see the module docstring), and Flask's own startup banner
        # lands on stderr the same instant this is printed — often flushed
        # into the middle of this very line. Everything after the URL is
        # somebody else's output.
        rest = line.split("control panel:", 1)[1].split()
        if rest and rest[0].startswith("http"):
            self.panel_url = rest[0]

    def stop(self) -> bool:
        """Ask the child to quit; escalate to SIGKILL if it ignores us."""
        proc = self._proc
        if proc is None:
            return self._stop_orphan()
        if proc.poll() is not None:
            return False
        self.status = "stopped"
        try:
            # Kill the whole process group: Open3D viewers and torch dataloaders
            # both spawn children that would otherwise survive.
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        threading.Thread(target=self._hard_kill, args=(proc,), daemon=True).start()
        self._changed()
        return True

    def _stop_orphan(self) -> bool:
        """Kill a survivor from an earlier dashboard, by process id.

        There is no pipe to it any more, so there is nothing to wait on: the
        signal goes to its whole group and the job is marked stopped.
        """
        if not self.orphan:
            return False
        try:
            os.killpg(os.getpgid(self.pid), signal.SIGTERM)
        except OSError:
            return False
        self.status = "stopped"
        self.finished = time.time()
        self._changed()
        return True

    def _changed(self) -> None:
        """Tell the manager to rewrite the history file."""
        if self._notify is not None:
            self._notify()

    @staticmethod
    def _hard_kill(proc: subprocess.Popen) -> None:
        try:
            proc.wait(timeout=6.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()


class JobManager:
    """Owns every job the dashboard has launched — this session and before.

    The history is reloaded from ``.dashboard/jobs.json`` at startup and
    rewritten whenever a job starts or ends, so the list of what you ran
    survives closing the dashboard.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        self.log_dir = os.path.join(root, ".dashboard", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.history_path = os.path.join(root, ".dashboard", HISTORY_FILE)
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._ids = itertools.count(self._restore())

    # ------------------------------------------------------------------ query
    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            jobs = [self._jobs[i] for i in reversed(self._order) if i in self._jobs]
        return [j.summary() for j in jobs]

    def running(self) -> list[Job]:
        """Every job still executing, including one that outlived a dashboard.

        Callers use this to decide what is busy — which port is taken, which
        folder must not be renamed — and a survivor holds those just as firmly
        as a job this dashboard started.
        """
        return [j for j in self._jobs.values() if j.status == "running" or j.orphan]

    # ---------------------------------------------------------------- history
    def _restore(self) -> int:
        """Load past jobs from disk; answer with the next free job number.

        A history file we cannot read is skipped rather than fatal — losing the
        list of past runs is annoying, refusing to start the dashboard is worse.
        """
        highest = self._highest_logged()
        try:
            with open(self.history_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return highest + 1
        if data.get("version") != HISTORY_VERSION:
            return highest + 1
        for rec in (data.get("jobs") or [])[-MAX_JOBS:]:
            try:
                job = Job.from_record(rec)
            except (KeyError, TypeError, ValueError):
                continue           # one bad record must not lose the rest
            # Point the job at *this* project's log folder rather than the
            # absolute path it was written with, so a project that has been
            # moved or renamed still finds its own logs.
            job.log_path = os.path.join(self.log_dir, f"{job.id}.log")
            job._notify = self._save
            self._jobs[job.id] = job
            self._order.append(job.id)
            highest = max(highest, _job_number(job.id))
        return highest + 1

    def _highest_logged(self) -> int:
        """Largest job number that already has a log file.

        Numbering past it as well as past the history means a job never writes
        over an older job's log, even if the history file was deleted.
        """
        try:
            names = os.listdir(self.log_dir)
        except OSError:
            return 0
        return max((_job_number(n[:-4]) for n in names if n.endswith(".log")),
                   default=0)

    def _save(self) -> None:
        """Rewrite the history file, oldest job first.

        Written to a temporary file and renamed into place so a dashboard that
        dies mid-write leaves the previous history intact rather than a
        half-file that reads as no history at all.
        """
        with self._lock:
            records = [self._jobs[i].record() for i in self._order if i in self._jobs]
        tmp = self.history_path + ".tmp"
        with self._write_lock:
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump({"version": HISTORY_VERSION, "jobs": records}, fh)
                os.replace(tmp, self.history_path)
            except OSError as e:
                # The history is a convenience; a read-only project directory
                # should not stop jobs from running.
                print(f"[dashboard] could not save job history: {e}", flush=True)

    # ----------------------------------------------------------------- launch
    def launch(self, argv: list[str], *, command_id: str, title: str,
               gui: bool = False, display: str = "",
               panel_port: int | None = None) -> Job:
        job_id = f"j{next(self._ids):04d}"
        log_path = os.path.join(self.log_dir, f"{job_id}.log")

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Open3D's Filament renderer needs X11 handles; the Wayland backend
        # segfaults during create_window. Same workaround the viewer applies to
        # itself, hoisted here so it also covers the offline viewers.
        if env.get("XDG_SESSION_TYPE") == "wayland":
            env["XDG_SESSION_TYPE"] = "x11"

        job = Job(id=job_id, command_id=command_id, title=title, argv=list(argv),
                  cwd=self.root, log_path=log_path, gui=gui, display=display,
                  panel_port=panel_port)

        try:
            proc = subprocess.Popen(
                argv, cwd=self.root, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, errors="replace",
                bufsize=1, start_new_session=True,
            )
        except OSError as e:
            job.status = "failed"
            job.returncode = -1
            job.finished = time.time()
            job._emit(f"[dashboard] could not start: {e}")
            self._remember(job)
            return job

        job._proc = proc
        job.pid = proc.pid
        job._emit(f"[dashboard] $ {job.command_line}")
        self._remember(job)
        threading.Thread(target=self._pump, args=(job, proc), name=f"job-{job_id}",
                         daemon=True).start()
        return job

    def rerun(self, job: Job) -> Job:
        """Launch a fresh job running the same command line as ``job``.

        Replays the argv the old job actually ran rather than rebuilding it from
        the form: the values behind a job are not kept, and the argv is the
        honest record of what happened. The new job gets its own id and log.
        """
        return self.launch(job.argv, command_id=job.command_id, title=job.title,
                           gui=job.gui, display=job.display,
                           panel_port=job.panel_port)

    def _remember(self, job: Job) -> None:
        job._notify = self._save
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > MAX_JOBS:
                old = self._order.popleft()
                stale = self._jobs.get(old)
                if stale is not None and stale.alive:
                    self._order.append(old)  # never evict a live job
                    break
                self._jobs.pop(old, None)
        self._save()

    def _pump(self, job: Job, proc: subprocess.Popen) -> None:
        try:
            with open(job.log_path, "w", encoding="utf-8") as log:
                log.write(f"$ {job.command_line}\n")
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    job._emit(line)
                    log.write(line + "\n")
                    log.flush()
        except Exception as e:  # a broken pipe must not leave the job "running"
            job._emit(f"[dashboard] log error: {e}")
        code = proc.wait()
        job.returncode = code
        job.finished = time.time()
        if job.status != "stopped":
            job.status = "ok" if code == 0 else "failed"
        job._emit(f"[dashboard] exited with code {code} after {job.elapsed:.1f}s")
        self._save()

    def shutdown(self) -> None:
        alive = list(self.running())
        for job in alive:
            job.stop()
        # Give the reader threads a moment to record how each job ended, so the
        # saved history says "stopped after four minutes" instead of freezing at
        # the moment it was launched.
        deadline = time.time() + 3.0
        while time.time() < deadline and any(j.finished is None for j in alive):
            time.sleep(0.1)
        self._save()
