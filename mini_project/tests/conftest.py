"""Shared pytest configuration: import path and dataset fixtures."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def kaggle_root() -> Path:
    """Root of the Kaggle denoising data; skip when raw/train is missing."""
    root = PROJECT_ROOT / "data" / "raw"
    if not (root / "train").is_dir():
        pytest.skip("Kaggle data not found at data/raw/train")
    return root


@pytest.fixture
def dibco_root() -> Path:
    """Root of the DIBCO datasets; skip when missing."""
    root = PROJECT_ROOT / "data" / "external" / "dibco"
    if not root.is_dir():
        pytest.skip("DIBCO data not found at data/external/dibco")
    return root
