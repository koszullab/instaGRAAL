#!/usr/bin/env bash
# =============================================================================
# generate_test_data.sh
# =============================================================================
# Generates the instaGRAAL test dataset:
#   • An in silico contig-level assembly of S. cerevisiae S288c
#     (yeast.contigs.fa.gz)
#   • Hi-C contact pairs mapped to that in silico assembly
#     (yeast.pairs.gz)
#
# The two output files should be uploaded to Zenodo and referenced
# by ``instagraal-test``.
#
# Requirements (install e.g. via micromamba / conda):
#   bwa >= 0.7, samtools >= 1.15, pairtools >= 1.0, biopython, numpy
#
# Hi-C dataset used:
#   SRR22130071  –  S. cerevisiae Hi-C (ENA / SRA)
#   Enzyme: DpnII + HinfI
#
# Reference genome:
#   S. cerevisiae S288c R64-1-1 (Ensembl release 115)
#   Downloaded from Ensembl FTP if not already present.
#
# Usage
# -----
#   bash scripts/generate_test_data.sh [--threads N] [--outdir DIR]
#
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/scripts"

# --- Defaults ----------------------------------------------------------------
THREADS=8
OUTDIR="${REPO_ROOT}/example/yeast"

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --outdir)  OUTDIR="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTDIR"
cd "$OUTDIR"

# --- File names --------------------------------------------------------------
FASTA_GZ="S288c.fa.gz"          # chromosome-level reference (downloaded)
ASSEMBLY="yeast.contigs.fa.gz"  # in silico contig assembly (output)
HIC_R1="SRR22130071_R1.fq.gz"
HIC_R2="SRR22130071_R2.fq.gz"
PAIRS="yeast.pairs.gz"

# =============================================================================
# Step 1 – Download reference genome (Ensembl S288c R64-1-1)
# =============================================================================

echo "==> Step 1: Download S. cerevisiae S288c reference genome"

if [[ ! -f "$FASTA_GZ" ]]; then
    curl -L \
        https://ftp.ensembl.org/pub/release-115/fasta/saccharomyces_cerevisiae/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz \
        -o "$FASTA_GZ"
else
    echo "    ${FASTA_GZ} already present, skipping."
fi

# =============================================================================
# Step 2 – Download Hi-C reads (SRR22130071)
# =============================================================================
# SRR22130071: S. cerevisiae Hi-C experiment (ENA accession), DpnII + HinfI.
# Both mates are fetched from the EBI FTP mirror.

echo "==> Step 2: Download Hi-C reads (SRR22130071)"

if [[ ! -f "$HIC_R1" ]]; then
    curl -L \
        ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR221/071/SRR22130071/SRR22130071_1.fastq.gz \
        -o "$HIC_R1"
else
    echo "    ${HIC_R1} already present, skipping."
fi

if [[ ! -f "$HIC_R2" ]]; then
    curl -L \
        ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR221/071/SRR22130071/SRR22130071_2.fastq.gz \
        -o "$HIC_R2"
else
    echo "    ${HIC_R2} already present, skipping."
fi

# =============================================================================
# Step 3 – Generate in silico contig-level assembly
# =============================================================================
# make_insilico_assembly.py fragments the reference using a Poisson break
# process (rate = 1.5 breaks/Mb, seed = 42) to mimic a typical ONT/PacBio
# HiFi assembly.  Expected output: ~30 contigs, N50 ~400–600 kb.

echo "==> Step 3: Generate in silico assembly"
python "${SCRIPTS_DIR}/make_insilico_assembly.py" "$FASTA_GZ" "$ASSEMBLY"

# =============================================================================
# Step 4 – Index the in silico assembly with BWA
# =============================================================================

echo "==> Step 4: Index assembly with BWA"
bwa index "$ASSEMBLY"

# =============================================================================
# Step 5 – Map Hi-C reads and build a pairs file
# =============================================================================
# Mapping strategy (standard Hi-C pipeline):
#   • bwa mem -5SP  : chimeric-read aware, no pairing, soft-clip supplementary
#   • pairtools parse: convert SAM → 4DN pairs format
#   • pairtools sort : coordinate sort (required for dedup)
#   • pairtools dedup: mark and remove PCR / optical duplicates
#   • pairtools select: keep valid ligation products (UU, UR, RU)
#   • pairtools split : write final pairs file (SAM discarded)

echo "==> Step 5: Map + parse Hi-C reads"

bwa mem -5SP -T0 -t "$THREADS" "$ASSEMBLY" "$HIC_R1" "$HIC_R2" \
    | pairtools parse \
          --min-mapq 40 \
          --walks-policy 5unique \
          --max-inter-align-gap 30 \
          --nproc-in 4 \
          --nproc-out 4 \
          --chroms-path "$ASSEMBLY" \
    | pairtools sort \
          --nproc "$THREADS" \
          --tmpdir /tmp \
    | pairtools dedup \
          --mark-dups \
          --output-stats dedup_stats.txt \
    | pairtools select \
          '(pair_type == "UU") or (pair_type == "UR") or (pair_type == "RU")' \
    | pairtools split \
          --output-pairs "$PAIRS" \
          --output-sam /dev/null

# =============================================================================
# Done
# =============================================================================

echo ""
echo "==> Done!  Upload the following files to Zenodo:"
echo "      ${OUTDIR}/${ASSEMBLY}"
echo "      ${OUTDIR}/${PAIRS}"
echo ""
echo "    Also keep dedup_stats.txt for QC reporting."
echo "    Then update ZENODO_RECORD_ID in src/instagraal/cli/test.py."
