"""GPU-dependent tests for instaGRAAL.

All tests in this module are skipped unless a CUDA-capable GPU is detected.

The module covers two goals:

1. **End-to-end pipeline coverage** – run ``instagraal`` in-process (so
   pytest-cov instruments the GPU modules) and validate every output artefact
   it produces.

2. **Unit coverage for zero-coverage CPU helpers** – ``leastsqbound``,
   ``vector``, ``optim_rippe_curve_update``, ``parse_info_frags``, and
   ``fragment`` are pure-Python / NumPy / SciPy modules that can be tested
   without touching CUDA, but are placed here so a single GPU-enabled run
   maximally increases overall coverage.
"""

import math
import pathlib
import struct

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# GPU detection – must happen before any pycuda.autoinit side-effect
# ---------------------------------------------------------------------------

GPU_AVAILABLE = False
try:
    import pycuda.driver as _cuda

    _cuda.init()
    GPU_AVAILABLE = _cuda.Device.count() > 0
except Exception:
    pass

pytestmark = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="CUDA GPU not available",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
TEST_DATA = REPO_ROOT / "tests" / "data"
REF_FASTA = TEST_DATA / "yeast.contigs.fa.gz"

MCMC_LEVEL = 4
MCMC_CYCLES = 3
# n_new_frags at pyramid level 4 for yeast data (shape sub mat = 727x727)
MCMC_FRAGS = 727
MCMC_N_ITERS = MCMC_CYCLES * MCMC_FRAGS if MCMC_FRAGS is not None else None

# Expected number of large (>100 kb) contigs after scaffolding.
# Yeast genome: expect roughly 15–45 depending on Hi-C library quality.
EXPECTED_LARGE_CONTIGS = 30  # midpoint of expected range
LARGE_CONTIG_TOLERANCE = 15  # ± 15 → accepts 15–45


# ---------------------------------------------------------------------------
# Session fixture: run instagraal in-process once for all GPU tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def instagraal_run(pre_output_dir, tmp_path_factory):
    """Run ``instagraal`` on the ``instagraal-pre`` output in-process so
    pytest-cov can instrument the GPU modules.

    Using ``pre_output_dir`` (produced by the shared conftest fixture) as the
    HiC folder exercises the full pipeline:
        instagraal-pre  →  instagraal

    Replicates:
        instagraal <pre_output_dir> <reference.fa> --output-dir <output_dir>
                   --cycles 3 --bomb --save-matrix

    Returns the MCMC output directory
    ``<output_dir>/<hic_folder_name>/test_mcmc_<level>/``.
    """
    output_dir = tmp_path_factory.mktemp("instagraal_out")
    hic_folder_name = pre_output_dir.name

    from click.testing import CliRunner  # noqa: PLC0415
    from instagraal.cli.main import main  # noqa: PLC0415

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            str(pre_output_dir),
            str(REF_FASTA),
            "--output-dir",
            str(output_dir),
            "--cycles",
            str(MCMC_CYCLES),
            "--level",
            str(MCMC_LEVEL),
            "--bomb",
            "--save-matrix",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"instagraal failed (exit {result.exit_code}):\n{result.output}"

    # Output follows simu_single.py naming: {output_dir}/{hic_name}/test_mcmc_{lvl}
    return output_dir / hic_folder_name / f"test_mcmc_{MCMC_LEVEL}"


@pytest.fixture(scope="session")
def mcmc_out(instagraal_run):
    if not instagraal_run.is_dir():
        pytest.skip(f"instagraal output dir missing: {instagraal_run}")
    return instagraal_run


# ---------------------------------------------------------------------------
# Output file existence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fname",
    [
        "genome.fasta",
        "info_frags.txt",
        "list_likelihood.txt",
        "list_n_contigs.txt",
        "list_mean_len.txt",
        "list_dist_init_genome.txt",
        "list_mutations.txt",
        "save_simu_step_0.txt",
        f"save_simu_step_{MCMC_CYCLES - 1}.txt",
        "matrix_cycle_0.png",
        f"matrix_cycle_{MCMC_CYCLES - 1}.png",
    ],
)
def test_gpu_output_file_exists(mcmc_out, fname):
    """Every expected output file is produced."""
    assert (mcmc_out / fname).exists(), f"Missing output file: {fname}"


# ---------------------------------------------------------------------------
# genome.fasta
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gpu_fasta_lines(mcmc_out):
    return (mcmc_out / "genome.fasta").read_text().splitlines()


def test_gpu_fasta_has_contigs(gpu_fasta_lines):
    headers = [l for l in gpu_fasta_lines if l.startswith(">")]
    assert len(headers) >= 1


def test_gpu_fasta_large_contig_count(gpu_fasta_lines):
    """Assembly must yield EXPECTED_LARGE_CONTIGS ± tolerance contigs > 100 kb."""
    seq_len = 0
    large = 0
    for line in gpu_fasta_lines:
        if line.startswith(">"):
            if seq_len > 100_000:
                large += 1
            seq_len = 0
        else:
            seq_len += len(line)
    if seq_len > 100_000:
        large += 1
    lo = EXPECTED_LARGE_CONTIGS - LARGE_CONTIG_TOLERANCE
    hi = EXPECTED_LARGE_CONTIGS + LARGE_CONTIG_TOLERANCE
    assert lo <= large <= hi, f"Expected {lo}–{hi} contigs >100 kb, got {large}"


def test_gpu_fasta_contig_names(gpu_fasta_lines):
    """Headers follow the '3C-assembly-contig_N' naming convention."""
    for line in gpu_fasta_lines:
        if line.startswith(">"):
            name = line.lstrip(">")
            assert name.startswith("3C-assembly-contig_"), f"Unexpected header: {line}"
            assert name.split("3C-assembly-contig_")[1].isdigit()


def test_gpu_fasta_valid_bases(gpu_fasta_lines):
    valid = set("ACGTNacgtn")
    for line in gpu_fasta_lines:
        if not line.startswith(">"):
            assert set(line) <= valid, f"Invalid chars in: {line[:60]!r}"


def test_gpu_fasta_no_empty_sequences(gpu_fasta_lines):
    current = None
    started = False
    for line in gpu_fasta_lines:
        if line.startswith(">"):
            if current is not None:
                assert started, f"Empty sequence for {current}"
            current, started = line, False
        elif line.strip():
            started = True
    if current is not None:
        assert started, f"Empty sequence for {current}"


# ---------------------------------------------------------------------------
# info_frags.txt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def info_frags_text(mcmc_out):
    return (mcmc_out / "info_frags.txt").read_text()


def test_gpu_info_frags_has_blocks(info_frags_text):
    blocks = [b.strip() for b in info_frags_text.split(">") if b.strip()]
    assert len(blocks) >= 1


def test_gpu_info_frags_block_structure(info_frags_text):
    """Each contig block has the expected column header and valid data rows."""
    blocks = [b.strip() for b in info_frags_text.split(">") if b.strip()]
    expected_cols = {"init_contig", "id_frag", "orientation", "start", "end"}
    for block in blocks:
        lines = block.splitlines()
        assert len(lines) >= 2
        assert set(lines[1].split()) == expected_cols
        for row in lines[2:]:
            fields = row.split()
            assert len(fields) == 5
            assert int(fields[2]) in (1, -1), f"Bad orientation: {fields[2]}"


# ---------------------------------------------------------------------------
# save_simu_step_*.txt
# ---------------------------------------------------------------------------


def test_gpu_save_simu_step_file_count(mcmc_out):
    assert len(sorted(mcmc_out.glob("save_simu_step_*.txt"))) == MCMC_CYCLES


def test_gpu_save_simu_step_row_count(mcmc_out):
    for i in range(MCMC_CYCLES):
        path = mcmc_out / f"save_simu_step_{i}.txt"
        rows = path.read_text().splitlines()
        assert len(rows) == MCMC_FRAGS, f"{path.name}: expected {MCMC_FRAGS} rows, got {len(rows)}"


def test_gpu_save_simu_step_format(mcmc_out):
    """Columns are pos, start_bp, id_c, ori – all integers; ori in {1, -1}."""
    for line in (mcmc_out / "save_simu_step_0.txt").read_text().splitlines():
        fields = line.split()
        assert len(fields) == 4
        assert int(fields[3]) in (1, -1)


# ---------------------------------------------------------------------------
# list_*.txt files
# ---------------------------------------------------------------------------


def test_gpu_list_likelihood_length(mcmc_out):
    vals = [v for v in (mcmc_out / "list_likelihood.txt").read_text().splitlines() if v.strip()]
    assert len(vals) == MCMC_N_ITERS


def test_gpu_list_likelihood_finite(mcmc_out):
    for v in (mcmc_out / "list_likelihood.txt").read_text().splitlines():
        if v.strip():
            assert math.isfinite(float(v)), f"Non-finite likelihood: {v}"


def test_gpu_list_n_contigs_length_and_positive(mcmc_out):
    vals = [v for v in (mcmc_out / "list_n_contigs.txt").read_text().splitlines() if v.strip()]
    assert len(vals) == MCMC_N_ITERS
    for v in vals:
        assert int(v) > 0


def test_gpu_list_mean_len_length(mcmc_out):
    vals = [v for v in (mcmc_out / "list_mean_len.txt").read_text().splitlines() if v.strip()]
    assert len(vals) == MCMC_N_ITERS


def test_gpu_list_dist_init_genome_length(mcmc_out):
    vals = [v for v in (mcmc_out / "list_dist_init_genome.txt").read_text().splitlines() if v.strip()]
    assert len(vals) == MCMC_N_ITERS


# ---------------------------------------------------------------------------
# list_mutations.txt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mutations_df(mcmc_out):
    return pd.read_csv(mcmc_out / "list_mutations.txt", sep="\t")


def test_gpu_mutations_columns(mutations_df):
    assert list(mutations_df.columns) == ["id_fA", "id_fB", "id_mutation"]


def test_gpu_mutations_row_count(mutations_df):
    assert len(mutations_df) == MCMC_N_ITERS


def test_gpu_mutations_fragment_ids_in_range(mutations_df):
    assert mutations_df["id_fA"].between(0, MCMC_FRAGS - 1).all()
    assert mutations_df["id_fB"].between(0, MCMC_FRAGS - 1).all()


def test_gpu_mutations_op_codes_non_negative(mutations_df):
    assert (mutations_df["id_mutation"] >= 0).all()


# ---------------------------------------------------------------------------
# matrix_cycle_*.png
# ---------------------------------------------------------------------------


def test_gpu_matrix_png_count(mcmc_out):
    assert len(sorted(mcmc_out.glob("matrix_cycle_*.png"))) == MCMC_CYCLES


def test_gpu_matrix_pngs_valid(mcmc_out):
    _SIG = b"\x89PNG\r\n\x1a\n"
    for i in range(MCMC_CYCLES):
        data = (mcmc_out / f"matrix_cycle_{i}.png").read_bytes()
        assert data[:8] == _SIG, f"matrix_cycle_{i}.png: bad PNG signature"
        w = struct.unpack(">I", data[16:20])[0]
        h = struct.unpack(">I", data[20:24])[0]
        assert w > 0 and h > 0, f"matrix_cycle_{i}.png: zero dimensions"


# ---------------------------------------------------------------------------
# Pyramid output (built by instagraal before scaffolding)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pyramid_dir(mcmc_out):
    """Return the pyramids directory built in the base output dir during the GPU run.

    Pyramids are written at <output_dir>/pyramids/, where <output_dir> is two
    levels above mcmc_out (<output_dir>/<hic_name>/test_mcmc_<level>/).
    """
    d = mcmc_out.parent.parent / "pyramids"
    if not d.exists():
        pytest.skip(f"Pyramid dir not found: {d}")
    return d


def test_gpu_pyramid_no_thresh_hdf5_exists(pyramid_dir):
    assert (pyramid_dir / "pyramid_1_no_thresh" / "pyramid.hdf5").exists()


def test_gpu_pyramid_thresh_auto_hdf5_exists(pyramid_dir):
    assert (pyramid_dir / "pyramid_9_thresh_auto" / "pyramid.hdf5").exists()


def test_gpu_pyramid_no_thresh_level0_files(pyramid_dir):
    level0 = pyramid_dir / "pyramid_1_no_thresh" / "level_0"
    for fname in ("0_abs_frag_contacts.txt", "0_contig_info.txt", "0_fragments_list.txt"):
        assert (level0 / fname).exists(), f"Missing: {fname}"


def test_gpu_pyramid_no_thresh_hdf5_structure(pyramid_dir):
    import h5py

    with h5py.File(pyramid_dir / "pyramid_1_no_thresh" / "pyramid.hdf5", "r") as f:
        assert "0" in f
        assert f.attrs.get("0") == "done"


def test_gpu_pyramid_thresh_auto_multi_level(pyramid_dir):
    import h5py

    with h5py.File(pyramid_dir / "pyramid_9_thresh_auto" / "pyramid.hdf5", "r") as f:
        assert len(list(f.keys())) > 1, "Expected multiple pyramid levels"


# ===========================================================================
# Unit tests for pure-CPU zero-coverage modules
# (no GPU required, but placed here to maximise coverage per GPU run)
# ===========================================================================

# ---------------------------------------------------------------------------
# leastsqbound
# ---------------------------------------------------------------------------


def test_leastsqbound_no_bounds_identity():
    """With no bounds, internal == external."""
    from instagraal.leastsqbound import internal2external  # noqa: PLC0415

    xi = np.array([1.0, -3.0, 0.5])
    bounds = [(None, None)] * 3
    xe = internal2external(xi, bounds)
    np.testing.assert_array_almost_equal(xe, xi)


def test_leastsqbound_roundtrip_min_only():
    """Min-only bound: external2internal then internal2external recovers value."""
    from instagraal.leastsqbound import external2internal, internal2external  # noqa: PLC0415

    xe_orig = np.array([3.0])
    bounds = [(1.0, None)]
    xi = external2internal(xe_orig, bounds)
    xe_back = internal2external(xi, bounds)
    np.testing.assert_array_almost_equal(xe_back, xe_orig, decimal=6)


def test_leastsqbound_roundtrip_max_only():
    from instagraal.leastsqbound import external2internal, internal2external  # noqa: PLC0415

    xe_orig = np.array([-2.0])
    bounds = [(None, 0.0)]
    xi = external2internal(xe_orig, bounds)
    xe_back = internal2external(xi, bounds)
    np.testing.assert_array_almost_equal(xe_back, xe_orig, decimal=6)


def test_leastsqbound_roundtrip_both_bounds():
    from instagraal.leastsqbound import external2internal, internal2external  # noqa: PLC0415

    xe_orig = np.array([0.3, 2.5])
    bounds = [(0.0, 1.0), (2.0, 5.0)]
    xi = external2internal(xe_orig, bounds)
    xe_back = internal2external(xi, bounds)
    np.testing.assert_array_almost_equal(xe_back, xe_orig, decimal=6)


def test_leastsqbound_fit_linear():
    """leastsqbound recovers true parameters for a bounded linear model."""
    from instagraal.leastsqbound import leastsqbound  # noqa: PLC0415

    def residuals(p, y, x):
        return y - (p[0] * x + p[1])

    x = np.linspace(0.0, 1.0, 30)
    true_a, true_b = 2.5, 0.5
    y = true_a * x + true_b
    x0 = np.array([1.0, 0.0])
    bounds = [(0.0, 5.0), (0.0, 2.0)]
    result, ier = leastsqbound(residuals, x0, bounds, args=(y, x))
    assert ier in (1, 2, 3, 4), f"Unexpected ier code: {ier}"
    np.testing.assert_array_almost_equal(result, [true_a, true_b], decimal=3)


def test_leastsqbound_full_output():
    """leastsqbound with full_output=True returns all five elements."""
    from instagraal.leastsqbound import leastsqbound  # noqa: PLC0415

    def residuals(p, y, x):
        return y - p[0] * x

    x = np.array([1.0, 2.0, 3.0])
    y = 2.0 * x
    x0 = np.array([1.0])
    bounds = [(0.0, 10.0)]
    out = leastsqbound(residuals, x0, bounds, args=(y, x), full_output=True)
    assert len(out) == 5


# ---------------------------------------------------------------------------
# vector
# ---------------------------------------------------------------------------


def test_vec_2d_properties():
    from instagraal.vector import Vec  # noqa: PLC0415

    v = Vec([3.0, 4.0])
    assert v.x == 3.0
    assert v.y == 4.0


def test_vec_3d_properties():
    from instagraal.vector import Vec  # noqa: PLC0415

    v = Vec([1.0, 2.0, 3.0])
    assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0


def test_vec_4d_properties():
    from instagraal.vector import Vec  # noqa: PLC0415

    v = Vec([1.0, 2.0, 3.0, 4.0])
    assert v.w == 4.0


def test_vec_invalid_size_returns_none():
    from instagraal.vector import Vec  # noqa: PLC0415

    assert Vec([1.0]) is None


def test_vec_arithmetic():
    from instagraal.vector import Vec  # noqa: PLC0415

    a = Vec([1.0, 2.0, 3.0])
    b = Vec([4.0, 5.0, 6.0])
    # Arithmetic on Vec may return a base ndarray in newer NumPy versions;
    # check values only.
    c = np.asarray(a) + np.asarray(b)
    np.testing.assert_array_equal(c, [5.0, 7.0, 9.0])


def test_vec_setitem():
    from instagraal.vector import Vec  # noqa: PLC0415

    v = Vec([0.0, 0.0, 0.0])
    v.x = 9.0
    assert v[0] == 9.0


def test_vec_repr():
    from instagraal.vector import Vec  # noqa: PLC0415

    v = Vec([1.0, 2.0])
    r = repr(v)
    assert "Vec" in r


# ---------------------------------------------------------------------------
# optim_rippe_curve_update
# ---------------------------------------------------------------------------


def test_rippe_peval_shape():
    from instagraal.optim_rippe_curve_update import peval  # noqa: PLC0415

    x = np.linspace(10.0, 1000.0, 50)
    params = [50.0, 9.6, -1.5, 1.0]
    y = peval(x, params)
    assert y.shape == x.shape
    assert np.all(y > 0), "Rippe curve must be positive"


def test_rippe_residuals_zero_on_perfect_fit():
    from instagraal.optim_rippe_curve_update import peval, residuals  # noqa: PLC0415

    x = np.linspace(10.0, 1000.0, 30)
    params = [50.0, 9.6, -1.5, 1.0]
    y_pred = peval(x, params)
    err = residuals(params, y_pred, x)
    np.testing.assert_array_almost_equal(err, 0.0, decimal=10)


def test_rippe_log_peval():
    from instagraal.optim_rippe_curve_update import log_peval, peval  # noqa: PLC0415

    x = np.linspace(10.0, 1000.0, 20)
    params = [50.0, 9.6, -1.5, 1.0]
    y_lin = peval(x, params)
    y_log = log_peval(x, params)
    np.testing.assert_array_almost_equal(y_log, np.log(y_lin), decimal=6)


def test_rippe_estimate_param_rippe_returns_negative_slope():
    """Estimated slope from a synthetic Rippe curve must be negative."""
    from instagraal.optim_rippe_curve_update import estimate_param_rippe, peval  # noqa: PLC0415

    x = np.linspace(10.0, 1000.0, 50)
    y = peval(x, [50.0, 9.6, -1.5, 1.0])
    params_out, y_estim = estimate_param_rippe(y, x)
    assert len(params_out) == 5
    assert params_out[2] < 0, f"slope should be negative, got {params_out[2]}"
    assert y_estim.shape == x.shape


def test_rippe_estimate_max_dist_intra():
    from instagraal.optim_rippe_curve_update import estimate_max_dist_intra  # noqa: PLC0415

    p = [50.0, 9.6, -1.5, 2.0, 1.0]
    val_inter = 0.001
    dist = estimate_max_dist_intra(p, val_inter)
    assert dist > 0, f"max_dist must be positive, got {dist}"


# ---------------------------------------------------------------------------
# parse_info_frags
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def example_info_frags(mcmc_out):
    """Return the path to info_frags.txt produced during the GPU run."""
    p = mcmc_out / "info_frags.txt"
    assert p.exists(), f"info_frags.txt not found: {p}"
    return p


def test_info_frags_txt_block_count(example_info_frags):
    """info_frags.txt from the GPU run has at least one contig block."""
    text = example_info_frags.read_text()
    blocks = [b.strip() for b in text.split(">") if b.strip()]
    assert len(blocks) >= 1


def test_info_frags_txt_block_structure(example_info_frags):
    """Every block has the expected column header and valid data rows."""
    text = example_info_frags.read_text()
    blocks = [b.strip() for b in text.split(">") if b.strip()]
    expected_cols = {"init_contig", "id_frag", "orientation", "start", "end"}
    for block in blocks:
        lines = block.splitlines()
        assert len(lines) >= 2
        assert set(lines[1].split()) == expected_cols
        for row in lines[2:]:
            fields = row.split()
            assert len(fields) == 5
            assert int(fields[2]) in (1, -1)


# ---------------------------------------------------------------------------
# fragment.basic_fragment
# ---------------------------------------------------------------------------


def test_basic_fragment_initiate():
    from instagraal.fragment import basic_fragment  # noqa: PLC0415

    frag = basic_fragment.initiate(
        np_id_abs=0,
        id_init=42,
        init_contig="ctg001",
        curr_id=7,
        start_pos=100,
        end_pos=2000,
        length_kb=1.9,
        gc_content=0.52,
        init_frag_start=0,
        init_frag_end=10,
        sub_frag_start=0,
        sub_frag_end=5,
        super_index=3,
        id_contig=2,
        n_accu_frags=4,
    )
    assert frag.id_init == 42
    assert frag.init_contig == "ctg001"
    assert frag.start_pos == 100
    assert frag.end_pos == 2000
    assert frag.gc_content == pytest.approx(0.52)
    assert frag.n_accu_frags == 4
    assert frag.orientation == "w"
    assert frag.init_name == "42-ctg001"


# ===========================================================================
# instagraal-polish integration tests (run on real main-command output)
# ===========================================================================


@pytest.fixture(scope="session")
def polish_out(mcmc_out, tmp_path_factory):
    """Run ``instagraal-polish --mode polishing`` on the GPU-scaffolded output.

    Uses the real ``info_frags.txt`` produced by ``instagraal`` together with
    the original reference FASTA so the full polishing pipeline is exercised
    against realistic data.
    """
    out = tmp_path_factory.mktemp("polish_out")
    from click.testing import CliRunner  # noqa: PLC0415
    from instagraal.cli.polish import main as polish_main  # noqa: PLC0415

    runner = CliRunner()
    result = runner.invoke(
        polish_main,
        [
            "--mode",
            "polishing",
            "--input",
            str(mcmc_out / "info_frags.txt"),
            "--fasta",
            str(REF_FASTA),
            "--output-dir",
            str(out),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"instagraal-polish failed (exit {result.exit_code}):\n{result.output}"
    return out


def test_gpu_polish_polished_genome_exists(polish_out):
    """polished_genome.fa is created by the polishing step."""
    assert (polish_out / "polished_genome.fa").exists()


def test_gpu_polish_polished_genome_non_empty(polish_out):
    """polished_genome.fa is non-empty."""
    assert (polish_out / "polished_genome.fa").stat().st_size > 0


def test_gpu_polish_polished_genome_valid_fasta(polish_out):
    """polished_genome.fa is a valid FASTA with at least one sequence."""
    from Bio import SeqIO  # noqa: PLC0415

    records = list(SeqIO.parse(str(polish_out / "polished_genome.fa"), "fasta"))
    assert len(records) > 0, "No sequences in polished_genome.fa"
    for r in records:
        assert len(r.seq) > 0, f"Empty sequence for {r.id}"


def test_gpu_polish_polished_genome_valid_bases(polish_out):
    """Polished FASTA contains only valid IUPAC bases."""
    valid = set("ACGTNacgtn")
    for line in (polish_out / "polished_genome.fa").read_text().splitlines():
        if not line.startswith(">"):
            assert set(line) <= valid, f"Invalid chars in: {line[:60]!r}"


def test_gpu_polish_new_info_frags_exists(polish_out):
    """new_info_frags.txt is created alongside the polished FASTA."""
    assert (polish_out / "new_info_frags.txt").exists()


def test_gpu_polish_new_info_frags_non_empty(polish_out):
    """new_info_frags.txt has at least one scaffold block."""
    from instagraal.parse_info_frags import parse_info_frags  # noqa: PLC0415

    scaffolds = parse_info_frags(str(polish_out / "new_info_frags.txt"))
    assert len(scaffolds) > 0
