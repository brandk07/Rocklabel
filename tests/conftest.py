import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import make_synthetic_mcap as synth  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_recording(tmp_path_factory):
    """(mcap_path, labels_path) for the synthetic 100-scan, 3-rock recording."""
    root = tmp_path_factory.mktemp("synthetic")
    mcap_path = root / "synthetic.mcap"
    labels_path = root / "synthetic.labels.json"
    synth.write_synthetic_mcap(str(mcap_path))
    synth.write_matching_labels(str(labels_path), mcap_name="synthetic.mcap")
    return str(mcap_path), str(labels_path)
