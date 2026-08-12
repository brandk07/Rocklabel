"""`rocklabel dash` — a local web dashboard over the whole rocklabel pipeline.

The dashboard is a thin control surface, not a reimplementation: every button
shells out to the very same ``rocklabel`` / ``rocklabel-train`` command you
would type, shows the exact command line first, and streams its output back.
Nothing here can do something the CLI cannot.

Flask is an optional extra (``pip install -e '.[dash]'``) so the core CLI stays
free of a web dependency.
"""

from __future__ import annotations

__all__ = ["run_dashboard"]


def run_dashboard(*args, **kwargs):
    """Lazy re-export so importing this package never pulls in Flask."""
    from .server import run_dashboard as _run

    return _run(*args, **kwargs)
