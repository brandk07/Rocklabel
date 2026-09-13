"""How synthetic phantom negatives are drawn, importable without torch.

Same reason as :mod:`rocklabel.train.models_meta`: the argparse layer and the
dashboard both have to name these modes, and neither may import torch to do it.
:mod:`rocklabel.train.engine` re-exports the tuple beside the code that uses it.
"""

from __future__ import annotations

#: Draw policies for :func:`rocklabel.train.engine._phantom_clumps`.
PHANTOM_MODES = ("legacy", "clear-only", "matched")
