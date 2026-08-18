"""An alternative, offline SLAM for rocklabel recordings.

Self-contained: nothing under ``rocklabel/`` imports this package, and this
package only ever *reads* from it. Point it at a recording and it writes a new
recording with better poses, leaving the original untouched.

    python -m altslam recordings/VolleyBallTest1.mcap

See ``altslam/README.md`` for what it does differently and why.
"""

from altslam.config import AltSlamConfig
from altslam.solver import OfflineSolver

__all__ = ["AltSlamConfig", "OfflineSolver"]
