# instaGRAAL developer Makefile
# Regenerates the full pipeline from the toy dataset in tests/data/yeast/pre/.
#
# Usage:
#   make all          Run the complete pipeline (pre → main → polish → post → stats)
#   make pre          Run instagraal-pre  (FASTA + pairs → HiC folder)
#   make main         Run instagraal      (requires CUDA GPU)
#   make polish       Run instagraal-polish
#   make post         Run instagraal-post (remap pairs + mcool)
#   make stats        Print assembly statistics for input + polished assembly
#   make clean        Remove all generated outputs
#   make help         Show this message
#
# NOTE: The 'main' step requires a CUDA-capable GPU.

FASTA      := tests/data/yeast.contigs.fa.gz
PAIRS      := tests/data/yeast.pairs.gz
ENZYMES    := DpnII,HinfI

LEVEL      := 4
CYCLES     := 3

HIC_DIR    := tests/data/yeast/main
OUT_BASE   := tests/data/yeast/out
MCMC_DIR   := $(OUT_BASE)/main/test_mcmc_$(LEVEL)
POLISH_DIR := tests/data/yeast/polish
POST_DIR   := tests/data/yeast/post

# Sentinel files used as Make dependency targets
HIC_SENTINEL    := $(HIC_DIR)/fragments_list.txt
MCMC_SENTINEL   := $(MCMC_DIR)/genome.fasta
POLISH_SENTINEL := $(POLISH_DIR)/polished_genome.fa
POST_SENTINEL   := $(POST_DIR)/yeast.mcool

.PHONY: all pre main polish post stats clean help

all: pre main polish post stats

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all     Run the complete pipeline"
	@echo "  pre     Run instagraal-pre  (FASTA + pairs → HiC folder)"
	@echo "  main    Run instagraal      (HiC folder → scaffolded assembly, requires CUDA)"
	@echo "  polish  Run instagraal-polish (info_frags → polished FASTA)"
	@echo "  post    Run instagraal-post  (remap pairs to new assembly, write mcool)"
	@echo "  stats   Print assembly statistics (input vs. polished)"
	@echo "  clean   Remove all generated outputs"

pre: $(HIC_SENTINEL)

$(HIC_SENTINEL): $(FASTA) $(PAIRS)
	instagraal-pre $(FASTA) $(PAIRS) --enzyme $(ENZYMES) --output-dir $(HIC_DIR)

main: $(MCMC_SENTINEL)

$(MCMC_SENTINEL): $(HIC_SENTINEL)
	instagraal $(HIC_DIR) $(FASTA) \
		--output-dir $(OUT_BASE) \
		--level $(LEVEL) --cycles $(CYCLES) --bomb --save-matrix

polish: $(POLISH_SENTINEL)

$(POLISH_SENTINEL): $(MCMC_SENTINEL)
	instagraal-polish \
		--input $(MCMC_DIR)/info_frags.txt \
		--fasta $(FASTA) \
		--output-dir $(POLISH_DIR)

post: $(POST_SENTINEL)

$(POST_SENTINEL): $(POLISH_SENTINEL)
	instagraal-post $(PAIRS) $(POLISH_DIR)/new_info_frags.txt \
		--resolutions 10000 \
		--output-dir $(POST_DIR)

stats: $(FASTA) $(POLISH_SENTINEL)
	instagraal-stats $(FASTA) $(POLISH_SENTINEL) --labels "Input,Polished"

clean:
	rm -rf $(HIC_DIR) $(OUT_BASE) $(POLISH_DIR) $(POST_DIR) instagraal-*.log sparsity_*.pdf
