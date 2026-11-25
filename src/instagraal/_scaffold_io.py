#!/usr/bin/env python
# coding: utf-8

"""Parsing and writing helpers for info_frags.txt / BED / FASTA files.

This module is internal; import from ``instagraal.parse_info_frags`` instead.
"""

import gzip

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

DEFAULT_MIN_SCAFFOLD_SIZE = 0
DEFAULT_NEW_INFO_FRAGS_NAME = "new_info_frags.txt"
DEFAULT_NEW_GENOME_NAME = "new_genome.fa"
DEFAULT_JUNCTION_SEQUENCE = Seq("NNNNNN")
DEFAULT_CRITERION = "colinear"
DEFAULT_CRITERION_2 = "blocks"


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------


def _parse_fasta(path):
    """Yield SeqRecord objects from a FASTA file, transparently handling
    gzip-compressed inputs (.gz).
    """
    path = str(path)
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            yield from SeqIO.parse(handle, "fasta")
    else:
        yield from SeqIO.parse(path, "fasta")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_info_frags(info_frags):
    """Import an info_frags.txt file and return a dictionary where each key
    is a newly formed scaffold and each value is the list of bins and their
    origin on the initial scaffolding.
    """

    new_scaffolds = {}
    with open(info_frags, "r") as info_frags_handle:
        current_new_contig = None
        for raw_line in info_frags_handle:
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            if line.startswith(">"):
                current_new_contig = line[1:].strip()
                new_scaffolds[current_new_contig] = []
            elif line.startswith("init_contig"):
                pass
            else:
                init_contig, id_frag, orientation, pos_start, pos_end = line.split("\t")
                start = int(pos_start)
                end = int(pos_end)
                ori = int(orientation)
                fragid = int(id_frag)
                assert start < end
                assert ori in {-1, 1}
                new_scaffolds[current_new_contig].append([init_contig, fragid, start, end, ori])

    return new_scaffolds


def parse_bed(bed_file):
    """Import a BED file (where the data entries are analogous to what may be
    expected in an info_frags.txt file) and return a scaffold dictionary,
    similarly to parse_info_frags.
    """

    new_scaffolds = {}
    with open(bed_file) as bed_handle:
        for line in bed_handle:
            chrom, start, end, query, qual, strand = line.split()[:7]
            if strand == "+":
                ori = 1
            elif strand == "-":
                ori = -1
            else:
                raise ValueError("Error when parsing strand orientation: {}".format(strand))

            if int(qual) > 0:
                bed_bin = [query, -2, int(start), int(end), ori]
                try:
                    new_scaffolds[chrom].append(bed_bin)
                except KeyError:
                    new_scaffolds[chrom] = [bed_bin]

    return new_scaffolds


def format_info_frags(info_frags):
    """A function to seamlessly run on either scaffold dictionaries or
    info_frags.txt files without having to check the input first.
    """
    if isinstance(info_frags, dict):
        return info_frags
    else:
        try:
            scaffolds = parse_info_frags(info_frags)
            return scaffolds
        except OSError:
            print("Error when opening info_frags.txt")
            raise


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_info_frags(scaffolds, output="new_info_frags.txt"):

    scaffolds = format_info_frags(scaffolds)
    with open(output, "w") as info_frags_handle:
        for new_name, scaffold in scaffolds.items():
            info_frags_handle.write(">{}\n".format(new_name))
            header_line = "\t".join(["init_contig", "id_frag", "orientation", "start", "end"])
            info_frags_handle.write("{}\n".format(header_line))
            for my_bin in scaffold:
                init_contig, id_frag, pos_start, pos_end, orientation = my_bin
                assert orientation in {-1, 1}
                my_line = "\t".join(
                    [
                        str(init_contig),
                        str(id_frag),
                        str(orientation),
                        str(pos_start),
                        str(pos_end),
                    ]
                )
                info_frags_handle.write("{}\n".format(my_line))


def write_fasta(init_fasta, info_frags, output=DEFAULT_NEW_GENOME_NAME, junction=False):
    """Convert an info_frags.txt file (or a scaffold dictionary) into a fasta
    file given a reference.
    Optionally adds junction sequences to reflect the possibly missing base
    pairs between two newly joined scaffolds: one junction is inserted
    between consecutive bins coming from different initial contigs.
    """

    init_genome = {record.id: record.seq for record in _parse_fasta(init_fasta)}
    scaffolds = format_info_frags(info_frags)
    junction_seq = str(junction) if junction else ""
    my_new_records = []
    for name, scaffold in scaffolds.items():
        pieces = []
        previous_contig = None
        for init_contig, _, start, end, ori in scaffold:
            if junction_seq and previous_contig not in {None, init_contig}:
                pieces.append(junction_seq)
            seq_to_add = init_genome[init_contig][start:end]
            if ori == 1:
                pieces.append(str(seq_to_add))
            elif ori == -1:
                pieces.append(str(seq_to_add.reverse_complement()))
            else:
                raise ValueError("Invalid data in orientation field {}".format(ori))
            previous_contig = init_contig
        my_new_records.append(SeqRecord(Seq("".join(pieces)), id=name, description=""))
    SeqIO.write(my_new_records, output, "fasta")
