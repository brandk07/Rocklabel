"""An alternative, offline SLAM for rocklabel recordings.

Self-contained: nothing under ``rocklabel/`` imports this package, and this
package only ever *reads* from it. Point it at a recording and it writes a new
recording with better poses, leaving the original untouched.

    python -m rocklabel.slam recordings/VolleyBallTest1.mcap

See ``rocklabel/slam/README.md`` for what it does differently and why.
"""

from rocklabel.slam.config import AltSlamConfig
from rocklabel.slam.solver import OfflineSolver

__all__ = ["AltSlamConfig", "OfflineSolver"]
