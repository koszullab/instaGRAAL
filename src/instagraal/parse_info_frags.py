#!/usr/bin/env python
# coding: utf-8

"""Functions to manipulate info_frags.txt files and BED files.

This module started as a couple of scripts to flip back artefact inversions
introduced by the GRAAL assembler and its derivatives (see:
https://github.com/koszullab/GRAAL, https://github.com/koszullab/instaGRAAL).
It has greatly evolved since, and provides a range of functions to polish
assemblies and correct potential misassemblies.

The implementation is split across three internal modules:

* ``_scaffold_io``      — parsing / writing info_frags, BED, and FASTA files
* ``_scaffold_correct`` — inversion / rearrangement / DNA-recovery algorithms
* ``_scaffold_viz``     — visualisation helpers

All public symbols are re-exported here so that existing code importing from
``instagraal.parse_info_frags`` continues to work without modification.
"""

from ._scaffold_correct import (
    correct_scaffolds,
    correct_spurious_inversions,
    find_lost_dna,
    integrate_lost_dna,
    is_block,
    rearrange_intra_scaffolds,
    reorient_consecutive_blocks,
    remove_spurious_insertions,
)
from ._scaffold_io import (
    DEFAULT_CRITERION,
    DEFAULT_CRITERION_2,
    DEFAULT_JUNCTION_SEQUENCE,
    DEFAULT_MIN_SCAFFOLD_SIZE,
    DEFAULT_NEW_GENOME_NAME,
    DEFAULT_NEW_INFO_FRAGS_NAME,
    _parse_fasta,
    format_info_frags,
    parse_bed,
    parse_info_frags,
    write_fasta,
    write_info_frags,
)
from ._scaffold_viz import plot_contig_composition, plot_info_frags

__all__ = [
    # constants
    "DEFAULT_CRITERION",
    "DEFAULT_CRITERION_2",
    "DEFAULT_JUNCTION_SEQUENCE",
    "DEFAULT_MIN_SCAFFOLD_SIZE",
    "DEFAULT_NEW_GENOME_NAME",
    "DEFAULT_NEW_INFO_FRAGS_NAME",
    # I/O
    "_parse_fasta",
    # correction algorithms
    "correct_scaffolds",
    "correct_spurious_inversions",
    "find_lost_dna",
    "format_info_frags",
    "integrate_lost_dna",
    "is_block",
    "parse_bed",
    "parse_info_frags",
    # visualisation
    "plot_contig_composition",
    "plot_info_frags",
    "rearrange_intra_scaffolds",
    "remove_spurious_insertions",
    "reorient_consecutive_blocks",
    "write_fasta",
    "write_info_frags",
]


if __name__ == "__main__":
    from .cli.polish import main

    main()
