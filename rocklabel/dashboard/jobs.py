"""Subprocess job manager: run a rocklabel command, stream its output, stop it.

One :class:`Job` per launched command. Output is captured line by line on a
reader thread into a bounded in-memory ring (what the browser polls) *and*
appended to a log file under ``.dashboard/logs/`` so a job's output outlives the
dashboard process.

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
import os
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field

#: Lines kept in memory per job (the log file keeps everything).
MAX_LINES = 4000
#: Finished jobs retained before the oldest is dropped.
MAX_JOBS = 60


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
    status: str = "running"        # running | ok | failed | stopped
    lines: deque = field(default_factory=lambda: deque(maxlen=MAX_LINES))
    _seq: int = 0                  # total lines ever emitted (cursor for polling)
    _proc: subprocess.Popen | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -------------------------------------------------------------- accessors
    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    @property
    def command_line(self) -> str:
        return self.display or " ".join(shlex.quote(a) for a in self.argv)

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
        }

    def tail(self, since: int = 0) -> dict:
        """Lines emitted after cursor ``since``, plus the new cursor.

        The ring only holds the last ``MAX_LINES``; if the caller has fallen
        further behind than that we hand back what we have and say so, rather
        than silently skipping lines.
        """
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
        if proc is None or proc.poll() is not None:
            return False
        self.status = "stopped"
        try:
            # Kill the whole process group: Open3D viewers and torch dataloaders
            # both spawn children that would otherwise survive.
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        threading.Thread(target=self._hard_kill, args=(proc,), daemon=True).start()
        return True

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
    """Owns every job the dashboard has launched this session."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.log_dir = os.path.join(root, ".dashboard", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ query
    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            jobs = [self._jobs[i] for i in reversed(self._order) if i in self._jobs]
        return [j.summary() for j in jobs]

    def running(self) -> list[Job]:
        return [j for j in self._jobs.values() if j.status == "running"]

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
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > MAX_JOBS:
                old = self._order.popleft()
                stale = self._jobs.get(old)
                if stale is not None and stale.status == "running":
                    self._order.append(old)  # never evict a live job
                    break
                self._jobs.pop(old, None)

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

    def shutdown(self) -> None:
        for job in list(self.running()):
            job.stop()
