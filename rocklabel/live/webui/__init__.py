"""Browser control panel for the live Open3D viewer (`rocklabel live --web-ui`).

The Open3D `gui` toolkit can host the 3D scene or a settings panel, but not a
good settings panel: no wrapping, no typography, and labels that reflow at
runtime and shove everything below them out of frame. So the controls move to a
page you put on a second monitor, and the Open3D window becomes just the scene.

Three pieces, in dependency order:

* :mod:`spec` — the declarative control table (no Flask, no engine),
* :mod:`control` — :class:`~rocklabel.live.webui.control.LiveController`, which
  reads and writes the running engine / scorer / viewer (no Flask),
* :mod:`server` — the Flask app over that controller.

Only :mod:`server` imports Flask, so the first two are importable — and
testable — with just the core install.
"""

from __future__ import annotations

__all__ = ["LiveController", "start_server"]


def __getattr__(name: str):
    # Lazy so that importing this package does not drag in Flask; `server` is
    # the only module that needs the [dash] extra.
    if name == "LiveController":
        from .control import LiveController

        return LiveController
    if name == "start_server":
        from .server import start_server

        return start_server
    raise AttributeError(name)
