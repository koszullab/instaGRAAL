# Polishing the scaffolds

The MCMC scaffolder moves bins, not contigs: its scaffolds can split a
contig, flip part of one, and leave out the DNA that the coverage filter drops
at the ends of contigs. `instagraal-polish` corrects these artefacts in a layout
(`info_frags.txt`), and writes the polished scaffolds. The `legacy` mode of
`instagraal-endtoend` polishes the MCMC scaffolds; the `default` mode refines
them instead (see [Refining the scaffolds](refine.md)).

```bash
instagraal-polish --input mcmc/pre/test_mcmc_4/info_frags.txt --fasta contigs.fa \
    --output-dir polished
```

- `--input` is the layout of the MCMC run, `info_frags.txt` (see
  [Scaffolding on a GPU](scaffolding.md#outputs)).
- `--fasta` is the FASTA file of the contigs given to `instagraal-pre`.

## What polishing does

Without `--mode`, polishing runs four steps, the modes `rearrange`,
`inversion2`, `reincorporation` and `fasta`:

1. **rearrange**: within each scaffold, the bins of each contig are brought
   back together, next to their longest run.
2. **inversion2**: each run of bins of one contig is put back in its order on
   the contig, and all its bins in the run's majority orientation
   (`--criterion blocks`, the default). `--criterion sequences` reorients the
   bins from whether the ids of their neighbours rise or fall, without
   reordering them.
3. **reincorporation**: the DNA that no bin covers, left out by the coverage
   filter or with the discarded scaffolds, is put back next to the bin it
   adjoins on its contig, in that bin's orientation. A region next to no bin,
   such as a contig the scaffolder left out whole, becomes a scaffold of its
   own. Every base of the contigs ends up in the assembly once.
4. **fasta**: the scaffolds are written as FASTA, with `--junction`
   (`NNNNNN`) between bins of different contigs.

Polishing keeps the order of the contigs that the MCMC chose.

## The modes

`--mode` runs one step alone:

| Mode | What it does | Writes |
|---|---|---|
| `polishing` (default) | The four steps above | `new_info_frags.txt`, `polished_genome.fa`, `contig_composition.png` |
| `rearrange` | Brings the bins of each contig together, within each scaffold | `new_info_frags.txt`, `contig_composition.png` |
| `inversion2` | Reorients and reorders runs of bins of one contig (`--criterion blocks` or `sequences`) | the same |
| `inversion` | Gives each neighbourhood of bins of one contig its majority orientation, with `--criterion` `cis` (bins of one contig), `colinear` (the default: in their order on the contig) or `contiguous` (exactly consecutive) | the same |
| `singleton` | Removes the bins inserted alone between two bins of another contig | the same |
| `reincorporation` | Puts back the DNA that no bin covers | the same |
| `fasta` | Writes the scaffolds as they are, with no polishing | `new_info_frags.txt`, `polished_genome.fa` |

`polishing`, `reincorporation` and `fasta` need `--fasta`. The `default` mode
of `instagraal-endtoend` runs `--mode fasta`, to hand the MCMC scaffolds as
they are to refine.

## Options

- `-s, --min-scaffold-size`: discard the scaffolds of this many bins or fewer
  (0 by default). In the `polishing` mode, their DNA goes back into the
  assembly next to its neighbours on the contigs, as any other DNA left out.
- `-l, --min-scaffold-length`: discard the scaffolds shorter than this, in bp.
- `-j, --junction`: the sequence between bins of different contigs (`NNNNNN`;
  `''` for none). `instagraal-post` must be given its length (see
  [Contact maps of the assembly](post.md#the-junction)).
- `-o, --output-dir`: where to write (`out` by default).

## Outputs

| File | What it holds |
|---|---|
| `new_info_frags.txt` | The polished layout ([info_frags](../formats.md#info_frags-files)), which `instagraal-post` and `instagraal-refine` read |
| `polished_genome.fa` | The polished scaffolds |
| `contig_composition.png` | For each scaffold, the length each input contig contributes to it |

`instagraal-polish` prints how many scaffolds the filters keep, then the
statistics of the polished assembly.
