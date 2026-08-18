"""Launcher for the full-sweep volleyball experiments.

Caps this process's share of the GPU before torch allocates anything, so a
concurrently running sweep can never be pushed into an out-of-memory crash by
this one: if the cap is hit, THIS process raises, not the other one.
"""
from __future__ import annotations

import os
import sys

import torch

#: Fraction of total VRAM this process may ever hold. The machine's other
#: sweep sits at ~1.4 GB of 8 GB, so 45% leaves it more headroom than it uses.
GPU_FRACTION = float(os.environ.get("VB_GPU_FRACTION", "0.45"))


def cap_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION, 0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        print(f"GPU cap: {GPU_FRACTION:.0%} of {total:.0f} MiB "
              f"= {GPU_FRACTION * total:.0f} MiB", flush=True)


if __name__ == "__main__":
    cap_gpu()
    from rocklabel.train.cli import main
    sys.exit(main(sys.argv[1:]))
