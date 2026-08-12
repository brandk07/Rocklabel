"""Replay transport-bar guards in the live viewer.

The seek path is worth pinning down because its failure mode is expensive and
silent: a slider callback mistaken for a user scrub issues a *backward* seek,
which rewinds the file and re-fuses the whole surface from t=0. Mirroring the
position onto the slider then produces the next echo, and the replay ends up
stuck in a rewind loop instead of playing.
"""

import numpy as np
import pytest

gui = pytest.importorskip("open3d.visualization.gui")

from rocklabel.live.viz.app import (  # noqa: E402
    _SEEK_MIN_DELTA_SEC,
    _SLIDER_ECHO_EPS_SEC,
    VizApp,
)


class _FakeSource:
    """Just the transport surface `_on_slider_changed` reads."""

    is_replay = True

    def __init__(self, position: float = 0.0, finished: bool = False) -> None:
        self.position_sec = position
        self.finished = finished


def _transport(position: float = 0.0, finished: bool = False):
    """A VizApp with only its transport state — no window, no engine, no GPU."""
    app = object.__new__(VizApp)
    src = _FakeSource(position, finished)
    app._engine = type("E", (), {"source": src})()
    app._pending_seek = None
    app._pending_seek_from = 0.0
    app._pending_seek_time = 0.0
    app._last_user_seek = 0.0
    app._synced_values = __import__("collections").deque(maxlen=8)
    return app, src


@pytest.mark.parametrize("position", [0.5, 8.5, 17.3, 27.8, 31.9, 32.1, 45.0, 58.7])
def test_float32_slider_echo_is_not_a_scrub(position):
    """Open3D hands the mirrored value back rounded to float32.

    The tolerance must cover half a float32 ulp at *any* replay position. It
    doubles at every power of two, so a tolerance tuned below 32 s (where the
    error is <=9.5e-7) silently breaks above it — which is exactly what made
    long recordings rewind-loop at the 32 s mark.
    """
    app, _src = _transport(position)
    app._synced_values.append(position)

    echo = float(np.float32(position))  # what Open3D's SliderFloat gives back
    app._on_slider_changed(echo)

    assert app._pending_seek is None, f"float32 echo at {position}s read as a scrub"
    assert app._last_user_seek == 0.0, "an echo must not suppress position mirroring"


def test_echo_tolerance_covers_float32_at_max_replay_length():
    """Guards the constant itself against being tightened back to ~1e-6."""
    longest = 3600.0  # an hour of replay, far past anything we record
    ulp = np.spacing(np.float32(longest))
    assert _SLIDER_ECHO_EPS_SEC > ulp
    # ...and stays far below the threshold a real scrub has to clear, so it can
    # never swallow a seek that would actually have moved playback.
    assert _SLIDER_ECHO_EPS_SEC < _SEEK_MIN_DELTA_SEC / 10


def test_stale_echo_at_current_position_is_ignored():
    """An echo evicted from the 8-slot ring still must not seek.

    The value matches where playback is, so whatever its provenance it cannot
    be asking to move.
    """
    app, _src = _transport(position=40.0)
    for v in range(8):  # push the echo's value out of the deque
        app._synced_values.append(100.0 + v)

    app._on_slider_changed(40.0 + 2e-6)

    assert app._pending_seek is None


def test_real_scrub_registers():
    app, _src = _transport(position=40.0)
    app._on_slider_changed(12.0)
    assert app._pending_seek == pytest.approx(12.0)
    assert app._pending_seek_from == pytest.approx(40.0)
    assert app._last_user_seek > 0.0


def test_scrub_at_the_end_registers_even_when_tiny():
    """Parked at EOF, a nudge means "replay from here", not "do nothing"."""
    app, _src = _transport(position=58.7, finished=True)
    app._on_slider_changed(58.68)
    assert app._pending_seek == pytest.approx(58.68)


def test_pending_seek_reference_is_pinned_at_registration():
    """The settled seek is judged against where playback was when scrubbed.

    Playback keeps running through the debounce, so it can drift onto (or past)
    the scrubbed value before the seek fires. Judged against the position
    *then*, a genuine scrub forward would look stationary and be dropped.
    """
    app, src = _transport(position=40.0)
    target = 40.0 + 2 * _SEEK_MIN_DELTA_SEC
    app._on_slider_changed(target)

    src.position_sec = target  # playback caught up during _SEEK_DEBOUNCE_SEC

    assert app._pending_seek_from == pytest.approx(40.0), "reference must not drift"
    moved = abs(app._pending_seek - app._pending_seek_from)
    assert moved >= _SEEK_MIN_DELTA_SEC, "a real scrub must survive playback catching up"
