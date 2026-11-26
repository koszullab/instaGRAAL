# Contact maps of the assembly

`instagraal-post` lifts the read pairs over from the contigs to the
scaffolds, and writes the contact maps of the new assembly.

```bash
instagraal-post pairs.gz refined/refined_info_frags.txt \
    --resolutions 1000,5000,25000 --junction 6 --output-dir post
```

- `pairs.gz` is the pairs file aligned on the contigs, the one given to
  `instagraal-pre`.
- The second argument is the layout of the scaffolds: `final_info_frags.txt`
  of `instagraal-endtoend`, `refined_info_frags.txt` of `instagraal-refine`,
  or `new_info_frags.txt` of `instagraal-polish`.

## What it writes

| File | What it holds |
|---|---|
| `<name>_scaffolds_binned.mcool` | The contacts of the scaffolds at each resolution of `--resolutions`, balanced |
| `<name>_scaffolds.cool` | One bin per scaffold: the contacts between scaffolds |
| `<name>_contigs.cool` | One bin per contig, the contigs in their new order (only if the pairs file has `#chromsize` lines) |
| `<name>_lifted.pairs.gz` | The pairs, lifted over to the scaffolds |
| `<name>_ps_curves.png` | The contact probability against the distance, P(s), on the contigs and on the scaffolds |
| `<name>_scaffolds_binned_<res>.png`, `<name>_scaffolds.png`, `<name>_contigs.png` | Pictures of the three maps |

`<name>` is `--cool-name`, or the name of the pairs file without `.pairs` and
`.gz`. A pair whose mates cannot both be lifted over is dropped, and counted
in the log.

## Resolutions

`--resolutions` takes bin sizes in bp, comma-separated: `1000,5000,25000`.
The smallest is the base: the lifted pairs are binned at it, and the others
are summed from it, so each must be a multiple of a smaller one
(`1000,2500` fails). One value makes a single resolution.
`instagraal-endtoend` uses `1000`.

## Balancing

Each resolution is balanced with cooler's ICE (`--balance`, the default),
which adds the `weight` column that most tools read. `--balance-args` passes
options to `cooler.balance_cooler`, as JSON:

```bash
instagraal-post pairs.gz final_info_frags.txt -r 1000,10000 \
    --balance-args '{"max_iters": 2000, "mad_max": 10}'
```

On a small library, balancing can stop before it converges, with a warning:
the weights are still written. `--no-balance` skips it.

## The junction { #the-junction }

The scaffolds hold a junction between contigs (`NNNNNN` by default), which
moves every coordinate after it. `instagraal-post` rebuilds the coordinates
of the scaffolds from their layout, so `--junction` must be the length of
the junction of the FASTA file:

| Made by | Junction | `instagraal-post --junction` |
|---|---|---|
| `instagraal-endtoend` | `--junction`, `NNNNNN` by default | passed on by the pipeline |
| `instagraal-polish` | `--junction`, `NNNNNN` by default | `6` (the default) |
| `instagraal-refine` | `--junction`, none by default | `0` |

With the wrong length, the contact maps drift from the FASTA file by that
many bp at each junction.

## Looking at the maps

The `.mcool` file is a standard multi-resolution cool file:
[cooler](https://github.com/open2c/cooler), [HiGlass](https://higlass.io),
[hicstuff](https://github.com/koszullab/hicstuff) and most Hi-C tools read it.

```bash
cooler info post/yeast_scaffolds_binned.mcool::resolutions/1000
cooler show post/yeast_scaffolds_binned.mcool::resolutions/1000 \
    refined_scaffold_1 -o scaffold_1.png
```

```python
import cooler

clr = cooler.Cooler("post/yeast_scaffolds_binned.mcool::resolutions/1000")
clr.chromsizes          # the scaffolds and their lengths
matrix = clr.matrix(balance=True).fetch("refined_scaffold_1")
```

[Checking an assembly](checking.md) says what to look for in them.
