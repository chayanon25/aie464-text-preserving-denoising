"""Generated Colab notebook: committed copy is current and the smoke run executes."""
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

MINI_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MINI_PROJECT / 'notebooks'))
from build_final_colab import NB_PATH, build_notebook  # noqa: E402


def test_notebook_is_generated_and_current():
    nb = nbformat.read(NB_PATH, as_version=4)
    assert nb == build_notebook()


def test_pack_bundle_entries_cover_inputs():
    sys.path.insert(0, str(MINI_PROJECT / 'tools'))
    from pack_drive_bundle import bundle_entries
    if not (MINI_PROJECT / 'data' / 'raw' / 'train').is_dir():
        pytest.skip('local data not available')
    arcs = [arc for _, arc in bundle_entries()]
    assert any(a.startswith('external/dibco/dibco2017/') for a in arcs)
    assert not any(a.endswith('.dat') for a in arcs)
    assert sum(a.startswith('checkpoints/') for a in arcs) == 4


@pytest.mark.slow
def test_notebook_smoke_executes(tmp_path):
    nb = build_notebook(smoke=True, local_repo=MINI_PROJECT)
    NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(tmp_path)}}).execute()
    assert (tmp_path / 'runs' / 'pilot_unet_edge_aug' / 'best.pt').exists()
    assert (tmp_path / 'runs' / 'summary.json').exists()
