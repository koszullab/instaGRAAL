# Using the Python API

Each command of instaGRAAL runs functions of its library, which you can call
from Python to run a step, read the outputs, or edit a layout. The
[Python API reference](../api/pre.md) lists them.

## The modules

| Module | What it holds | Command |
|---|---|---|
| `instagraal.pre` | `run_pre`: the contact map of the restriction fragments, from the contigs and the pairs | `instagraal-pre` |
| `instagraal.instagraal` | `run_instagraal`: the MCMC scaffolder (needs pycuda and a GPU) | `instagraal` |
| `instagraal.refine` | `run_refine`, and the scoring of contig ends it is built on | `instagraal-refine` |
| `instagraal.parse_info_frags` | Reading and writing layouts and FASTA files, and the polishing steps | `instagraal-polish` |
| `instagraal.post` | `run_post`: the contact maps of the scaffolds | `instagraal-post` |
| `instagraal.assembly_stats` | The statistics of an assembly | `instagraal-stats` |

## Running the steps

The pipeline of `instagraal-endtoend --mode cpu`, from Python. `run_pre` and
`run_post` take `pathlib.Path` objects:

```python
from pathlib import Path

from instagraal.post import run_post
from instagraal.pre import run_pre
from instagraal.refine import run_refine

run_pre(Path("contigs.fa"), Path("pairs.gz"), ["DpnII", "HinfI"], Path("pre"))
summary = run_refine(
    info_frags=None,
    cool_path="pre/pairs.cool",
    fasta="contigs.fa",
    output_dir="refined",
    junction="NNNNNN",
    extend="on",
)
run_post(Path("pairs.gz"), Path("refined/refined_info_frags.txt"), Path("post"),
         resolutions="1000,10000", junction_len=6)
```

`run_refine` returns a summary of the run (`summary["scaffolds"]`,
`summary["scaffold_n50"]`, `summary["circular"]`…). `run_instagraal`
needs a GPU: importing `instagraal.instagraal` initialises CUDA.

## Statistics

```python
from instagraal.assembly_stats import (
    compute_assembly_stats,
    format_comparison_table,
)

results = {
    "Contigs": compute_assembly_stats("contigs.fa"),
    "Scaffolds": compute_assembly_stats("refined/refined_genome.fa"),
}
results["Scaffolds"]["n50"]
print(format_comparison_table(results))
```

`compute_assembly_stats` returns a dictionary: `n_contigs`, `total_length`,
`largest_contig`, `shortest_contig`, `mean_length`, `median_length`, `n50`,
`l50`, `n90`, `l90` and `gc_content`.

## Layouts

`parse_info_frags` reads a layout ([info_frags](../formats.md#info_frags-files))
into a dictionary: for each scaffold, its pieces in order, as
`[contig, id, start, end, orientation]`. `write_info_frags` writes one, and
`write_fasta` the scaffolds of a layout, from the contigs. Here, the
scaffolds of 500 kb or more:

```python
from instagraal.parse_info_frags import (
    parse_info_frags,
    write_fasta,
    write_info_frags,
)

layout = parse_info_frags("refined/refined_info_frags.txt")
largest = {
    name: pieces
    for name, pieces in layout.items()
    if sum(end - start for _, _, start, end, _ in pieces) >= 500_000
}
write_info_frags(largest, output="largest_info_frags.txt")
write_fasta("contigs.fa", "largest_info_frags.txt", output="largest.fa",
            junction="NNNNNN")
```

The steps of `instagraal-polish` take and return such dictionaries:
`rearrange_intra_scaffolds`, `reorient_consecutive_blocks`,
`correct_spurious_inversions`, `remove_spurious_insertions`, and
`find_lost_dna` with `integrate_lost_dna` (see
[Polishing](../api/polishing.md)).

## Contact maps

The cool and mcool files are read with [cooler](https://github.com/open2c/cooler):

```python
import cooler

clr = cooler.Cooler("post/pairs_scaffolds_binned.mcool::resolutions/10000")
clr.chromsizes
clr.matrix(balance=True).fetch("refined_scaffold_1")
```
