"""instaGRAAL testing

Tests for the instagraal-pre preprocessing module, downstream compatibility
with the instaGRAAL scaffolder's pyramid-building step, and a CPU-only check of
``run_instagraal``'s calls into ``instagraal_class``.

Validation of main-command (GPU) output artefacts lives in test_instagraal_gpu.py.
"""

import pathlib
import shutil
import sys
import types
from unittest.mock import MagicMock, create_autospec, patch

import h5py
import pandas as pd
import pytest

from instagraal import pyramid_sparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
TEST_DATA = REPO_ROOT / "tests" / "data"

REF_FASTA = TEST_DATA / "yeast.contigs.fa.gz"
REF_PAIRS = TEST_DATA / "yeast.pairs.gz"

# TODO: calibrate EXPECTED_N_FRAGS and EXPECTED_N_PIXELS after first run with yeast data.
# Exact counts are dataset-dependent and must be updated once the pipeline has been run.
EXPECTED_N_FRAGS = None  # placeholder - test below uses a sanity check instead
EXPECTED_N_PIXELS = None  # placeholder - test below uses a sanity check instead

# pre_output_dir is a session-scoped fixture defined in conftest.py and shared
# with the GPU test suite.


# ---------------------------------------------------------------------------
# Tests: output file existence
# ---------------------------------------------------------------------------


def test_output_files_exist(pre_output_dir):
    """All expected output files are created."""
    for name in (
        "fragments_list.txt",
        "info_contigs.txt",
        "abs_fragments_contacts_weighted.txt",
        "yeast.cool",
    ):
        assert (pre_output_dir / name).exists(), f"Missing output: {name}"


def test_pre_logs_assembly_stats(tmp_path_factory):
    """instagraal-pre calls print_assembly_stats on the input FASTA."""
    from click.testing import CliRunner

    from instagraal.cli.pre import main as pre_main

    out = tmp_path_factory.mktemp("pre_stats_check")
    runner = CliRunner()
    with patch("instagraal.cli.pre.print_assembly_stats") as mock_stats:
        result = runner.invoke(
            pre_main,
            [
                str(REF_FASTA),
                str(REF_PAIRS),
                "--enzyme",
                "DpnII,HinfI",
                "--output-dir",
                str(out),
            ],
        )
    assert result.exit_code == 0, f"instagraal-pre failed:\n{result.output}"
    mock_stats.assert_called_once()
    args, kwargs = mock_stats.call_args
    called_label = kwargs.get("label", args[1] if len(args) > 1 else "")
    assert "input" in called_label.lower()


# ---------------------------------------------------------------------------
# Tests: fragments_list.txt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fragments_out(pre_output_dir):
    return pd.read_csv(pre_output_dir / "fragments_list.txt", sep="\t")


def test_fragment_count(fragments_out):
    """Fragment table is non-empty.  Update EXPECTED_N_FRAGS once calibrated."""
    assert len(fragments_out) > 0  # TODO: tighten to == EXPECTED_N_FRAGS after calibration


def test_gc_content_range(fragments_out):
    """GC content values are in [0, 1] and look biologically plausible."""
    gc = fragments_out["gc_content"]
    assert gc.between(0.0, 1.0).all(), "GC values outside [0, 1]"
    # Mean GC for a bacterial genome should be in a reasonable range
    assert 0.2 <= gc.mean() <= 0.8


# ---------------------------------------------------------------------------
# Tests: abs_fragments_contacts_weighted.txt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pixels_out(pre_output_dir):
    return pd.read_csv(
        pre_output_dir / "abs_fragments_contacts_weighted.txt",
        sep="\t",
        skiprows=1,
        header=None,
        names=["bin1_id", "bin2_id", "count"],
    )


def test_pixel_count(pixels_out):
    """Pixel table is non-empty.  Update EXPECTED_N_PIXELS once calibrated."""
    assert len(pixels_out) > 0  # TODO: tighten to == EXPECTED_N_PIXELS after calibration


# ---------------------------------------------------------------------------
# Tests: instaGRAAL pyramid build (CUDA-free parsing step)
# ---------------------------------------------------------------------------


def test_pyramid_build_from_pre_output(pre_output_dir, tmp_path):
    """pyramid_sparse.build() successfully parses the instagraal-pre output.

    This exercises the exact file-loading code path that instaGRAAL uses
    before any GPU operations, confirming that the output files are
    structurally compatible with the scaffolder.
    """
    # Copy the three text files into a fresh temp dir so the pyramid
    # sub-directories don't bleed into pre_output_dir across test runs.
    work = tmp_path / "hic_folder"
    work.mkdir()
    for name in (
        "fragments_list.txt",
        "info_contigs.txt",
        "abs_fragments_contacts_weighted.txt",
    ):
        shutil.copy(pre_output_dir / name, work / name)

    pyramid_sparse.build(str(work), size_pyramid=1, factor=3, min_bin_per_contig=1)

    hdf5_path = work / "pyramids" / "pyramid_1_no_thresh" / "pyramid.hdf5"
    assert hdf5_path.exists(), "pyramid.hdf5 was not created"

    with h5py.File(hdf5_path, "r") as f:
        assert f.attrs.get("0") == "done", "Pyramid level 0 not marked as done"
        assert "0" in f, "Pyramid level 0 dataset missing from HDF5"


# ---------------------------------------------------------------------------
# Tests: run_instagraal dispatch (no GPU work)
# ---------------------------------------------------------------------------


def test_run_instagraal_calls_existing_methods(tmp_path, monkeypatch):
    """``run_instagraal`` only calls methods that ``instagraal_class`` defines.

    The class is replaced by an autospec mock, which raises AttributeError for
    any attribute the real class lacks, so a call to a removed method fails
    here without a GPU.
    """
    try:
        import pycuda.autoinit  # noqa: F401  (a no-op with the CPU CI pycuda stub)
    except Exception as exc:  # real pycuda without a usable CUDA device
        pytest.skip(f"instagraal.instagraal needs pycuda.autoinit: {exc}")
    from instagraal import instagraal as ig

    mock_cls = create_autospec(ig.instagraal_class)
    mock_stats = MagicMock()
    monkeypatch.setattr(ig, "instagraal_class", mock_cls)
    monkeypatch.setattr("instagraal.assembly_stats.print_assembly_stats", mock_stats)
    p2 = mock_cls.return_value
    p2.simulation = MagicMock()  # set in __init__, so not part of the autospec

    ig.run_instagraal(tmp_path, REF_FASTA, cycles=3, neighborhood=7, bomb=True, save_matrix=True)

    p2.full_em.assert_called_once_with(n_cycles=3, n_neighbours=7, bomb=True, id_start_sample_param=4, save_matrix=True)
    mock_stats.assert_called_once_with(p2.simulation.new_fasta, label="Scaffolded assembly")


# ---------------------------------------------------------------------------
# Tests: without pycuda (pip install instagraal, without the gpu extra)
# ---------------------------------------------------------------------------


def test_mcmc_without_pycuda_points_to_the_gpu_extra(no_pycuda, tmp_path):
    from click.testing import CliRunner

    from instagraal.cli.main import main

    result = CliRunner().invoke(main, [str(tmp_path), str(REF_FASTA)])
    assert result.exit_code == 1, result.output
    assert "pip install 'instagraal[gpu]'" in result.output
    assert "instagraal-endtoend --mode cpu" in result.output


def test_mcmc_reports_any_other_missing_module_as_it_is(monkeypatch, tmp_path):
    """Only a missing pycuda reads as the missing gpu extra."""
    from click.testing import CliRunner

    from instagraal.cli.main import main

    monkeypatch.setitem(sys.modules, "pycuda", types.ModuleType("pycuda"))
    monkeypatch.setitem(sys.modules, "pycuda.autoinit", types.ModuleType("pycuda.autoinit"))
    monkeypatch.delitem(sys.modules, "instagraal.instagraal", raising=False)
    monkeypatch.setitem(sys.modules, "instagraal.simu_single", None)

    result = CliRunner().invoke(main, [str(tmp_path), str(REF_FASTA)])
    assert isinstance(result.exception, ModuleNotFoundError)
    assert result.exception.name == "instagraal.simu_single"
    assert "instagraal[gpu]" not in result.output


def test_gpu_self_test_without_pycuda_points_to_the_gpu_extra(no_pycuda):
    import click

    from instagraal.cli.test import _check_gpu

    with pytest.raises(click.ClickException, match=r"pip install 'instagraal\[gpu\]'"):
        _check_gpu(0)
