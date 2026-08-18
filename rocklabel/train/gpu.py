"""Cap this process's share of the GPU.

With two sweeps on one machine, whichever one asks for memory second is the one
that crashes — and that is usually not the one doing anything wrong. Capping
the *asking* process instead means a run that overreaches fails on its own
account, and a long sweep already in flight is never knocked over by a job
started beside it.

Lifted out of the throwaway ``vb_run.py`` launcher that sat in the repo root,
because a genuinely useful guard rail should be a flag, not a script only its
author knows about.
"""

from __future__ import annotations


def cap_gpu(fraction: float | None) -> None:
    """Limit this process to ``fraction`` of total VRAM. No-op without CUDA.

    Must be called before the first allocation, which in practice means before
    a model is built — hence the call sitting at the top of the CLI's main()
    rather than inside the training loop.
    """
    if not fraction or fraction >= 1.0:
        return
    import torch

    if not torch.cuda.is_available():
        print(f"--gpu-fraction {fraction} ignored: no CUDA device", flush=True)
        return
    torch.cuda.set_per_process_memory_fraction(float(fraction), 0)
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"GPU cap: {fraction:.0%} of {total_mib:.0f} MiB "
          f"= {fraction * total_mib:.0f} MiB for this process", flush=True)
