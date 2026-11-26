# The end-to-end pipeline

```bash
## 1. Getting the assembly
# Assemble the long reads with hifiasm (PacBio HiFi reads; Flye for Nanopore
# reads), and write its primary contigs as FASTA.
hifiasm -o sample -t 16 hifi.fq.gz
awk '/^S/ {print ">"$2; print $3}' sample.bp.p_ctg.gfa > contigs.fa

## 2. Mapping the Hi-C pairs
# Align the Hi-C reads on the contigs and stop at the pairs:
# hicstuff/sample.valid_idx.pairs.gz
hicstuff pipeline \
    --genome contigs.fa \
    --enzyme DpnII,HinfI \
    --mapping cutsite \
    --matrix none \
    --threads 16 \
    --outdir hicstuff \
    --prefix sample \
    hic_R1.fq.gz hic_R2.fq.gz

## 3. Scaffolding
# Order and orient the contigs into chromosomes: instagraal/final_genome.fa
instagraal-endtoend \
    --enzyme DpnII,HinfI \
    --cool-name sample \
    --output-dir instagraal \
    contigs.fa hicstuff/sample.valid_idx.pairs.gz

## 4. Checking
# The contigs beside the scaffolds
instagraal-stats contigs.fa instagraal/final_genome.fa --labels "Contigs,Scaffolds"
```

This takes a genome from its reads to chromosomes. It needs reads to
assemble (long reads, here) and a Hi-C library of the same organism.

1. **Assembly.** hifiasm writes its primary contigs in
   `sample.bp.p_ctg.gfa`. instaGRAAL does not assemble: any assembler's
   contigs work, if their names hold no spaces or special characters (see
   [Troubleshooting](troubleshooting.md#keyerror-on-contig-names)).
2. **Pairs.** hicstuff (4.0.0 or later) aligns the reads and writes their
   pairs, sorted and compressed. Give it the enzymes of the library: DpnII and
   HinfI for the Arima kit. `--mapping cutsite` cuts the reads at their
   ligation sites before aligning them, which aligns more of them;
   `--matrix none` stops at the pairs, since instaGRAAL makes its own contact
   maps. The pairs of any other aligner work too, in the
   [pairs format](formats.md#pairs) (see [Preparing the contacts](guides/pre.md)).
3. **Scaffolding.** `instagraal-endtoend` cuts the contigs into restriction
   fragments, counts the contacts between them, runs the MCMC scaffolder on
   the GPU, then refines its layout, and writes the contact maps of the
   scaffolds. Without a GPU, add `--mode cpu`; for a genome of more than
   500 Mb, `--level 5` (see [Running the pipeline](guides/pipeline.md)). Its
   outputs, in `instagraal/`:

    | Output | What it holds |
    |---|---|
    | `final_genome.fa` | The scaffolds |
    | `final_info_frags.txt` | Their layout: the contigs of each scaffold, in order, with their orientation |
    | `refined/` | The scaffolds as `instagraal-refine` wrote them, the score of every junction (`junctions.tsv`) and a report |
    | `post/` | The contact maps of the scaffolds: `sample_scaffolds_binned.mcool`, cool files, the lifted pairs and pictures |
    | `pre/`, `mcmc/`, `polished/` | The intermediate steps: the contacts between restriction fragments, the MCMC run, its scaffolds |

4. **Checking.** `instagraal-stats` sets the contigs beside the scaffolds.
   The contact map of the scaffolds, the score of each junction and the GC
   skew of bacterial chromosomes tell more: see
   [Checking an assembly](guides/checking.md).

See [Running the pipeline](guides/pipeline.md) for the modes and the options,
and [GPUs, CUDA and Docker](guides/gpu.md) to run it on a cluster.
