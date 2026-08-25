"""Two checkpoints on one live session, side by side.

The question this answers is "which of these two models is actually better on
my data", and the honest way to answer it is to show both looking at the *same
scan* with the *same settings* at the *same moment* — not two runs an hour
apart with a slider nudged in between.

So a session has two model **slots**:

* **a** — the model the run started with, drawn in the main window;
* **b** — the comparison, drawn in a second Open3D window that opens on demand.

Both scorers read the same :class:`~rocklabel.live.pipeline.IngestEngine`, and
— this is the part that makes the comparison mean anything — they **share one
:class:`~rocklabel.live.scoring.ScoreSettings` object**. Scoring region, scan
window, interval, persistence and the outline settings are therefore not
"kept in sync": they are the same numbers in the same place, and cannot drift.
The decision threshold is copied across on every write for the same reason.
The only difference left between the two windows is the checkpoint.

Loading a checkpoint takes seconds (torch reads the weights off disk), and the
Open3D GUI thread must never block for seconds, so every load runs on a worker
thread and only the swap itself is posted back to the GUI thread. Callers see
that as a state machine — ``closed`` → ``loading`` → ``open`` — which the panel
renders as a readout rather than as a frozen window.

Nothing here requires a window: with ``--headless --web-ui`` slot b still runs
and its numbers still appear in the panel, there is simply nothing to draw.
"""

from __future__ import annotations

import os
import threading

#: The two model slots. "a" always holds the model the run was started with.
SLOTS = ("a", "b")


class Comparison:
    """Owns the model slots of one live session, and the second window."""

    def __init__(self, config, engine, scorer=None, viz=None,
                 device: str | None = None) -> None:
        self._cfg = config
        self._engine = engine
        self._device = device
        self._scorers = {"a": scorer, "b": None}
        self._viz = {"a": viz, "b": None}
        self._paths = {
            "a": str(getattr(scorer, "checkpoint", "") or ""),
            "b": "",
        }
        #: What slot b is *set to* — which is not the same as what it is
        #: *running*: you pick a checkpoint, then press Open.
        self.selected_b = ""
        self.state = "closed"          # closed | loading | open | error
        self.message = ""
        #: Called as ``on_swap(slot, scorer)`` after a slot changes model, so
        #: whoever else holds a reference to the old scorer can rebind.
        self.on_swap = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # What is running
    # ------------------------------------------------------------------ #
    @property
    def scorer_a(self):
        return self._scorers["a"]

    @property
    def scorer_b(self):
        return self._scorers["b"]

    @property
    def scorers(self) -> list:
        """Every scorer currently running — what a settings write fans out to."""
        return [s for s in self._scorers.values() if s is not None]

    @property
    def open(self) -> bool:
        return self._scorers["b"] is not None

    def path(self, slot: str) -> str:
        return self._paths.get(slot, "")

    # ------------------------------------------------------------------ #
    # Writes that must reach every slot
    # ------------------------------------------------------------------ #
    def set_threshold(self, value: float) -> None:
        """One threshold for both models — see the module docstring."""
        for scorer in self.scorers:
            scorer.threshold = float(value)

    def clear_maps(self) -> None:
        for scorer in self.scorers:
            scorer.clear_map()

    # ------------------------------------------------------------------ #
    # Opening, closing, swapping
    # ------------------------------------------------------------------ #
    def select(self, slot: str, checkpoint: str) -> None:
        """Point a slot at a checkpoint.

        Slot a swaps immediately — there is always a model in window 1. Slot b
        only records the choice while the comparison is closed (opening it is
        the button's job), and swaps live while it is open, so picking a
        different model mid-session does the obvious thing.
        """
        if slot not in SLOTS:
            raise KeyError(slot)
        checkpoint = str(checkpoint or "").strip()
        if slot == "b":
            # Checked here rather than at Open: a picker that accepts a path
            # and only complains three clicks later is a picker that lies.
            if checkpoint and not os.path.exists(checkpoint):
                raise ValueError(f"no such checkpoint: {checkpoint}")
            self.selected_b = checkpoint
            if not self.open:
                return
            if not checkpoint:
                self.close()
                return
        elif not checkpoint:
            raise ValueError("window 1 always holds a model")
        self.load(slot, checkpoint)

    def open_comparison(self) -> None:
        """Open the second window on the selected checkpoint."""
        if not self.selected_b:
            raise ValueError("pick a model for window 2 first")
        if self._scorers["a"] is None:
            raise ValueError("comparison needs a run started with --model")
        self.load("b", self.selected_b)

    def load(self, slot: str, checkpoint: str) -> None:
        """Load ``checkpoint`` into ``slot``, on a worker thread.

        Returns as soon as the load is *started*: the caller is an HTTP request
        or a GUI callback, and neither may sit through a torch load.
        """
        if slot not in SLOTS:
            raise KeyError(slot)
        checkpoint = str(checkpoint or "").strip()
        if not checkpoint or not os.path.exists(checkpoint):
            raise ValueError(f"no such checkpoint: {checkpoint or '(none)'}")
        with self._lock:
            if self.state == "loading":
                raise ValueError("a checkpoint is already loading")
            running = self._scorers[slot]
            if running is not None and _same_file(checkpoint, self._paths[slot]):
                return                      # already the model in that slot
            self.state = "loading"
            self.message = f"loading {short_name(checkpoint)}…"
        threading.Thread(target=self._load_worker, args=(slot, checkpoint),
                         name=f"compare-load-{slot}", daemon=True).start()

    def close(self) -> None:
        """Shut the comparison down: window closed, scorer stopped."""
        if self._viz["b"] is not None:
            # Going through the window means one teardown path, whether the
            # close came from this panel or from the window's own title bar.
            self._post(self._viz["b"].close_window)
        else:
            self._release_b()

    def released(self, slot: str) -> None:
        """Called by a secondary window that has closed itself."""
        if slot == "b":
            self._release_b()

    # -- internals ------------------------------------------------------ #
    def _build(self, checkpoint: str):
        from rocklabel.live.scoring import LiveScorer, ScoreSettings

        anchor = self._scorers["a"]
        # The shared settings object is the whole point: two scorers reading
        # one ScoreSettings cannot disagree about the region, the interval or
        # anything else the panel touches.
        settings = anchor.settings if anchor is not None else ScoreSettings()
        return LiveScorer(checkpoint, self._engine, device=self._device,
                          settings=settings)

    def _load_worker(self, slot: str, checkpoint: str) -> None:
        try:
            scorer = self._build(checkpoint)
        except Exception as e:                       # bad file, no torch, OOM
            with self._lock:
                self.state = "error"
                self.message = f"{type(e).__name__}: {e}"
            print(f"[rocklabel] compare: could not load {checkpoint}: {e}",
                  flush=True)
            return
        self._post(lambda: self._install(slot, checkpoint, scorer))

    def _install(self, slot: str, checkpoint: str, scorer) -> None:
        """Swap a loaded scorer in. Runs on the GUI thread when there is one."""
        # The session's threshold wins over the new checkpoint's own tuned one:
        # one slider drives every model here, and a model that quietly arrived
        # at a different cut would make the two pictures incomparable. Its
        # tuned value is kept on the scorer and reported in the caveat line.
        running = self.scorers
        if running:
            scorer.threshold = float(running[0].threshold)
        old = self._scorers[slot]
        if old is not None:
            old.stop()
        self._scorers[slot] = scorer
        self._paths[slot] = checkpoint
        if slot == "b":
            self.selected_b = checkpoint
        scorer.start()

        viz = self._viz[slot]
        if viz is not None:
            viz.set_scorer(scorer)
        elif slot == "b":
            self._open_window(scorer)
        if callable(self.on_swap):
            self.on_swap(slot, scorer)
        with self._lock:
            self.state = "open" if self.open else "closed"
            self.message = ""
        print(f"[rocklabel] compare: window {'2' if slot == 'b' else '1'} -> "
              f"{scorer.model_name} ({short_name(checkpoint)})", flush=True)

    def _open_window(self, scorer) -> None:
        primary = self._viz["a"]
        if primary is None:
            return                 # headless: the readouts are the comparison
        viz = primary.open_comparison_window(
            scorer, title=f"rocklabel · compare · {short_name(scorer.checkpoint)}",
            on_closed=lambda: self.released("b"),
        )
        self._viz["b"] = viz

    def _release_b(self) -> None:
        scorer = self._scorers["b"]
        if scorer is not None:
            scorer.stop()
        self._scorers["b"] = None
        self._viz["b"] = None
        self._paths["b"] = ""
        with self._lock:
            if self.state != "loading":
                self.state = "closed"
                self.message = ""

    def _post(self, fn) -> None:
        """Run ``fn`` on the GUI thread if there is one, else right here."""
        viz = self._viz["a"]
        if viz is None or not viz.post(fn):
            fn()

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        """State, the two models, and anything that makes them incomparable."""
        a, b = self._scorers["a"], self._scorers["b"]
        note = ""
        if a is not None and b is not None:
            wa = float(getattr(a, "frame_window_s", 0.0))
            wb = float(getattr(b, "frame_window_s", 0.0))
            if abs(wa - wb) > 1e-6:
                # Both are fed whatever the shared 'Scan window' says, so one of
                # them is being shown a density it never trained on. That is a
                # real caveat on the comparison, not a detail.
                note = (f"trained on different scan windows "
                        f"({wa:g} s vs {wb:g} s) — they are not fed the same "
                        f"density by one setting")
            elif abs(float(a.tuned_threshold) - float(b.tuned_threshold)) > 1e-6:
                note = (f"tuned thresholds differ: {a.tuned_threshold:.3f} vs "
                        f"{b.tuned_threshold:.3f} — one slider drives both")
        return {
            "state": self.state,
            "message": self.message,
            "open": self.open,
            "selected_b": self.selected_b,
            "model_a": getattr(a, "model_name", "") if a is not None else "",
            "model_b": getattr(b, "model_name", "") if b is not None else "",
            "path_a": self._paths["a"],
            "path_b": self._paths["b"],
            "windowed": self._viz["b"] is not None,
            "note": note,
        }


def short_name(path: str) -> str:
    """``…/compare/pointnet_f3/best.pt`` -> ``pointnet_f3/best.pt``.

    A full checkpoint path is four directories of experiment structure and does
    not fit a readout; the run directory plus the file is what tells two of
    them apart.
    """
    path = str(path or "")
    if not path:
        return ""
    head, tail = os.path.split(path)
    parent = os.path.basename(head)
    return f"{parent}/{tail}" if parent else tail


def _same_file(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return os.path.abspath(a) == os.path.abspath(b)


__all__ = ["Comparison", "SLOTS", "short_name"]
