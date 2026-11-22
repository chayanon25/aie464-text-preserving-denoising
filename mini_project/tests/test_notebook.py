"""Generated Colab notebook: committed copy is current and the smoke run executes."""
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

MINI_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MINI_PROJECT / 'notebooks'))
from build_final_colab import NB_PATH, build_notebook  # noqa: E402


def test_submission_notebook_keeps_runner_code_and_outputs():
    # final_colab.ipynb is the executed Colab run plus explanations (tools/annotate_colab_notebook.py):
    # its runner code cells must match the builder except the pinned GIT_SHA, and keep their outputs.
    nb = nbformat.read(NB_PATH, as_version=4)
    runner = [c for c in nb.cells if c.cell_type == 'code'][:10]
    built = [c for c in build_notebook().cells if c.cell_type == 'code']
    strip = lambda s: '\n'.join(l for l in s.splitlines() if not l.startswith('GIT_SHA'))  # noqa: E731
    assert [strip(c.source) for c in runner] == [strip(c.source) for c in built]
    assert "GIT_SHA = '75664d9" in runner[0].source
    assert sum(bool(c.outputs) for c in runner) >= 7


def test_pack_bundle_entries_cover_inputs():
    sys.path.insert(0, str(MINI_PROJECT / 'tools'))
    from pack_drive_bundle import bundle_entries
    # The packer runs on the dev machine only; Colab has the unzipped bundle, not the sources.
    if not (MINI_PROJECT / 'data' / 'raw' / 'train').is_dir() or not (MINI_PROJECT / 'backups').is_dir():
        pytest.skip('local bundle sources not available')
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
