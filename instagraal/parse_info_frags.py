#!/usr/bin/env python
# coding: utf-8

"""
A couple of functions to manipulate info_frags.txt files and BED files.

This module started as a couple scripts to flip back artefact inversions
introduced by the GRAAL assembler and its derivatives (see:
https://github.com/koszullab/GRAAL, https://github.com/koszullab/instaGRAAL).
It has greatly evolved since, and provides a range of functions to polish
assemblies and correct potential missassemblies.


"""

import argparse
import copy
import itertools
import operator
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq, IUPAC
from matplotlib import pyplot as plt

# Defaults:
# -Scaffolds below DEFAULT_MIN_SCAFFOLD_SIZE are not considered for polishing
# -Defaut names for output files
# -Junction sequence between two bins that have been stitched together
# in order to reflect that some base pairs may be missing
# -Two default schemes ('criteria') for each inversion corrector
DEFAULT_MIN_SCAFFOLD_SIZE = 0
DEFAULT_NEW_INFO_FRAGS_NAME = "new_info_frags.txt"
DEFAULT_NEW_GENOME_NAME = "new_genome.fa"
DEFAULT_JUNCTION_SEQUENCE = Seq("NNNNNN", IUPAC.ambiguous_dna)
DEFAULT_CRITERION = "colinear"
DEFAULT_CRITERION_2 = "blocks"


def parse_info_frags(info_frags):
    """Import an info_frags.txt file and return a dictionary where each key
    is a newly formed scaffold and each value is the list of bins and their
    origin on the initial scaffolding.
    """

    new_scaffolds = {}
    with open(info_frags, "r") as info_frags_handle:
        current_new_contig = None
        for line in info_frags_handle:
            if line.startswith(">"):
                current_new_contig = str(line[1:-1])
                new_scaffolds[current_new_contig] = []
            elif line.startswith("init_contig"):
                pass
            else:
                (init_contig, id_frag, orientation, pos_start, pos_end) = str(
                    line[:-1]
                ).split("\t")
                start = int(pos_start)
                end = int(pos_end)
                ori = int(orientation)
                fragid = int(id_frag)
                assert start < end
                assert ori in {-1, 1}
                new_scaffolds[current_new_contig].append(
                    [init_contig, fragid, start, end, ori]
                )

    return new_scaffolds


# The format of info_frags.txt files looks a lot like the BED format, so this
# function may come in handy
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
                raise ValueError(
                    "Error when parsing strand "
                    "orientation: {}".format(strand)
                )

            if int(qual) > 0:
                bed_bin = [query, -2, int(start), int(end), ori]
                try:
                    new_scaffolds[chrom].append(bed_bin)
                except KeyError:
                    new_scaffolds[chrom] = [bed_bin]

    return new_scaffolds


def correct_scaffolds(scaffolds, corrector):
    """Unfinished
    """

    new_scaffolds = {}

    def are_overlapping(bin1, bin2):
        """Check for overlapping regions between two regions - necessary
        requirement before potentially merging
        """

        if bin2 is None:
            return False

        init1, _, start1, end1, _ = bin1
        init2, _, start2, end2, _ = bin2

        if init1 != init2:
            return False
        else:
            return (start2 <= start1 <= end2) or (start1 <= start2 <= end1)

    def merge_bins(bin1, bin2, ignore_ori=True):
        """Painstakingly check for every edge case in order to properly merge
        two overlapping bins.
        """
        init1, _, start1, end1, ori1 = bin1
        init2, _, start2, end2, ori2 = bin2

        assert init1 == init2
        start = min(start1, start2)
        end = max(end1, end2)
        if ignore_ori:
            ori = ori1
        else:
            len1 = end1 - start1
            len2 = end2 - start2
            if len2 > len1:
                ori = ori2
            else:
                ori = ori1

        new_bin = [init1, -3, start, end, ori]
        return new_bin

    corrector_bins = copy.deepcopy(corrector)

    for name, scaffold in scaffolds.items():
        new_scaffold = []

        for _, blocks in itertools.groupby(scaffold, operator.itemgetter(0)):
            merged_bin = None
            while "Reading blocks":
                try:
                    my_bin = next(blocks)
                    if are_overlapping(my_bin, merged_bin):
                        merged_bin = merge_bins(my_bin, merged_bin)
                        continue
                    else:
                        if merged_bin is not None:
                            new_scaffold.append(merged_bin)
                        merged_bin = my_bin
                        i = 0
                        for correct_bin in corrector_bins:
                            if are_overlapping(my_bin, correct_bin):
                                merged_bin = merge_bins(my_bin, correct_bin)
                                corrector_bins.pop(i)
                                i -= 1
                            i += 1
                except StopIteration:
                    if merged_bin is not None:
                        new_scaffold.append(merged_bin)
                    break

        new_scaffolds[name] = new_scaffold

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


def plot_info_frags(scaffolds):

    """A crude way to visualize new scaffolds according to their origin on the
    initial scaffolding. Each scaffold spawns a new plot. Orientations are
    represented by different colors.
    """

    scaffolds = format_info_frags(scaffolds)
    for name, scaffold in scaffolds.items():
        plt.figure()
        xs = range(len(scaffold))
        color = []
        names = {}
        ys = []
        for my_bin in scaffold:
            current_color = "r" if my_bin[4] > 0 else "g"
            color += [current_color]
            name = my_bin[0]
            if name in names:
                ys.append(names[name])
            else:
                names[name] = len(names)
                ys.append(names[name])
        plt.scatter(xs, ys, c=color)
    plt.show()


def remove_spurious_insertions(scaffolds):

    """Remove all bins whose left and right neighbors belong to the same,
    different scaffold.

    Example with three such insertions in two different scaffolds:

        >>> scaffolds = {
        ...     "scaffold1": [
        ...         ["contig1", 0, 0, 100, 1],
        ...         ["contig1", 1, 100, 200, 1],
        ...         ["contig23", 53, 1845, 2058, -1], # <-- insertion
        ...         ["contig1", 4, 254, 408, 1],
        ...         ["contig1", 7, 805, 1253, 1],
        ...         ["contig5", 23, 1500, 1605, -1],
        ...         ["contig65", 405, 32145, 45548, -1], # <-- insertion
        ...         ["contig5", 22, 1385, 1499, -1],
        ...     ],
        ...     "scaffold2": [
        ...         ["contig8", 0, 0, 250, 1],
        ...         ["contig17", 2454, 8754, -1], # <-- insertion
        ...         ["contig8", 2, 320, 480, 1],
        ...      ],
        ... }

        >>> new_scaffolds = remove_spurious_insertions(scaffolds)
        >>> for my_bin in new_scaffolds['scaffold1']:
        ...     print(my_bin)
        ...
        ['contig1', 0, 0, 100, 1]
        ['contig1', 1, 100, 200, 1]
        ['contig1', 4, 254, 408, 1]
        ['contig1', 7, 805, 1253, 1]
        ['contig5', 23, 1500, 1605, -1]
        ['contig5', 22, 1385, 1499, -1]

        >>> for my_bin in new_scaffolds['scaffold2']:
        ...     print(my_bin)
        ...
        ['contig8', 0, 0, 250, 1]
        ['contig8', 2, 320, 480, 1]


    """

    scaffolds = format_info_frags(scaffolds)
    new_scaffolds = {}
    for name, scaffold in scaffolds.items():
        new_scaffold = []
        if len(scaffold) > 2:
            for i in range(len(scaffold)):
                # First take care of edge cases: *-- or --*
                if i == 0:
                    if not (
                        scaffold[i][0] != scaffold[i + 1][0]
                        and scaffold[i + 1][0] == scaffold[i + 2][0]
                    ):
                        new_scaffold.append(scaffold[i])
                elif i == len(scaffold) - 1:
                    if not (
                        scaffold[i][0] != scaffold[i - 1][0]
                        and scaffold[i - 1][0] == scaffold[i - 2][0]
                    ):
                        new_scaffold.append(scaffold[i])
                # Otherwise, looking for -*-
                else:
                    if not (
                        scaffold[i - 1][0] == scaffold[i + 1][0]
                        and scaffold[i - 1][0] != scaffold[i][0]
                    ):
                        new_scaffold.append(scaffold[i])
        else:
            # Can't remove insertions if 2 bins or less
            new_scaffold = copy.deepcopy(scaffold)

        new_scaffolds[name] = new_scaffold

    return new_scaffolds


def correct_spurious_inversions(scaffolds, criterion="colinear"):
    """Invert bins based on orientation neighborhoods. Neighborhoods can be
    defined by three criteria:

    -a 'cis' neighborhood is a group of bins belonging to the same initial
    contig
    -a 'colinear' neighborhood is a 'cis' neighborhood where bins are
    ordered the same way they were on the initial contig
    -a 'contiguous' neighborhood is a 'colinear' neighborhood where all bins
    are exactly consecutive, i.e. the end position of each bin matches the
    starting position of the next bin

    This function looks for such neighborhoods and orients all bins in it
    according to the majority orientation.

    An example with three inversions, one for each criterion:

        >>> scaffolds = {
        ...     "scaffold1": [
        ...         ["contig1", 1, 100, 200, 1],
        ...         ["contig1", 2, 200, 300, 1],
        ...         ["contig1", 3, 300, 400, -1], # <-- inversion (contiguous)
        ...         ["contig1", 4, 400, 500, 1],
        ...         ["contig1", 10, 1500, 1605, 1],
        ...         ["contig1", 12, 1750, 1850, -1], # <-- inversion (colinear)
        ...         ["contig1", 23, 2100, 2499, 1],
        ...         ["contig1", 28, 2850, 3000, 1],
        ...         ["contig1", 0, 0, 100, -1], # <-- inversion (cis)
        ...         ["contig2", 554, 1850, 1900, -1],
        ...     ],
        ... }

    With the 'cis' criterion, pretty much all bins from "contig1" get inverted
    to the majority orientation (+):

        >>> sc_cis = correct_spurious_inversions(scaffolds, "cis")
        >>> for my_bin in sc_cis['scaffold1']:
        ...     print(my_bin)
        ...
        ['contig1', 1, 100, 200, 1]
        ['contig1', 2, 200, 300, 1]
        ['contig1', 3, 300, 400, 1]
        ['contig1', 4, 400, 500, 1]
        ['contig1', 10, 1500, 1605, 1]
        ['contig1', 12, 1750, 1850, 1]
        ['contig1', 23, 2100, 2499, 1]
        ['contig1', 28, 2850, 3000, 1]
        ['contig1', 0, 0, 100, 1]
        ['contig2', 554, 1850, 1900, -1]

    With the 'colinear' criterion, the bin ['contig1', 0, 0, 100, -1] is
    treated as a different neighborhood from the rest (as it is not colinear
    with the other bins from 'contig1') and remains untouched:

        >>> sc_colinear = correct_spurious_inversions(scaffolds, "colinear")
        >>> for my_bin in sc_colinear['scaffold1']:
        ...     print(my_bin)
        ...
        ['contig1', 1, 100, 200, 1]
        ['contig1', 2, 200, 300, 1]
        ['contig1', 3, 300, 400, 1]
        ['contig1', 4, 400, 500, 1]
        ['contig1', 10, 1500, 1605, 1]
        ['contig1', 12, 1750, 1850, 1]
        ['contig1', 23, 2100, 2499, 1]
        ['contig1', 28, 2850, 3000, 1]
        ['contig1', 0, 0, 100, -1]
        ['contig2', 554, 1850, 1900, -1]

    With the 'contiguous' criterion, the ['contig1', 12, 1750, 1850, -1] breaks
    with the contiguous region spanning from 100 to 400 bp on 'contig1' and
    so is treated as a different neighborhood as well:

        >>> sc_cont = correct_spurious_inversions(scaffolds, "contiguous")
        >>> for my_bin in sc_cont['scaffold1']:
        ...     print(my_bin)
        ...
        ['contig1', 1, 100, 200, 1]
        ['contig1', 2, 200, 300, 1]
        ['contig1', 3, 300, 400, 1]
        ['contig1', 4, 400, 500, 1]
        ['contig1', 10, 1500, 1605, 1]
        ['contig1', 12, 1750, 1850, -1]
        ['contig1', 23, 2100, 2499, 1]
        ['contig1', 28, 2850, 3000, 1]
        ['contig1', 0, 0, 100, -1]
        ['contig2', 554, 1850, 1900, -1]

    Note that 'contig2' remains untouched at all times since bins in it are
    never in the same neighborhood as those from 'contig1'.

    """

    scaffolds = format_info_frags(scaffolds)
    new_scaffolds = {}

    def is_cis(bin1, bin2):
        return bin1[0] == bin2[0]

    def is_contiguous(bin1, bin2):
        return is_cis(bin1, bin2) and bin1[3] == bin2[2]

    def is_colinear(bin1, bin2):
        return is_cis(bin1, bin2) and bin1[3] <= bin2[2]

    condition_callables = {
        "cis": is_cis,
        "colinear": is_colinear,
        "contiguous": is_contiguous,
    }
    block_test = condition_callables.get(criterion, "colinear")

    for name, scaffold in scaffolds.items():
        new_scaffold = []

        block_cumulative_ori = 0

        if len(scaffold) > 2:
            current_bin = scaffold[0]
            block_buffer = []
            for my_bin in scaffold:

                if not block_buffer:
                    new_bin = copy.deepcopy(my_bin)
                    block_buffer.append(new_bin)
                    block_cumulative_ori = my_bin[-1]
                    continue

                elif not block_test(current_bin, my_bin):
                    for my_buf_bin in block_buffer:
                        new_bin = copy.deepcopy(my_buf_bin)
                        if block_cumulative_ori >= 0:
                            new_bin[-1] = 1
                        else:
                            new_bin[-1] = -1
                        new_scaffold.append(new_bin)
                    block_cumulative_ori = my_bin[-1]
                    current_bin = copy.deepcopy(my_bin)
                    block_buffer = copy.deepcopy([my_bin])

                else:
                    block_cumulative_ori += my_bin[-1]
                    new_bin = copy.deepcopy(my_bin)
                    block_buffer.append(new_bin)
                    current_bin = my_bin

            for my_bin in block_buffer:
                new_bin = copy.deepcopy(my_bin)
                if block_cumulative_ori >= 0:
                    new_bin[-1] = 1
                else:
                    new_bin[-1] = -1
                new_scaffold.append(new_bin)
            new_scaffolds[name] = copy.deepcopy(new_scaffold)

        else:
            new_scaffolds[name] = copy.deepcopy(scaffold)

    return new_scaffolds


def rearrange_intra_scaffolds(scaffolds):

    """Rearranges all bins within each scaffold such that all bins belonging
    to the same initial contig are grouped together in the same order. When
    two such groups are found, the smaller one is moved to the larger one.
    """

    scaffolds = format_info_frags(scaffolds)
    new_scaffolds = {}

    ordering = dict()

    for name, scaffold in scaffolds.items():
        new_scaffold = []

        ordering = dict()
        order = 0

        my_blocks = []

        for _, my_block in itertools.groupby(scaffold, operator.itemgetter(0)):

            my_bins = list(my_block)
            my_blocks.append(my_bins)
            block_length = len(my_bins)
            block_name = my_bins[0][0]

            if block_name in ordering.keys():
                if block_length > ordering[block_name][1]:
                    ordering[block_name] = (order, block_length)
            else:
                ordering[block_name] = (order, block_length)
            order += 1

        def block_order(block):
            return ordering[block[0][0]][0]

        for my_block in sorted(my_blocks, key=block_order):
            for my_bin in my_block:
                new_scaffold.append(my_bin)

        new_scaffolds[name] = copy.deepcopy(new_scaffold)

    return new_scaffolds


def reorient_consecutive_blocks(scaffolds, mode="blocks"):

    scaffolds = format_info_frags(scaffolds)
    new_scaffolds = {}

    for name, scaffold in scaffolds.items():
        # print(scaffold)
        new_scaffold = []
        for _, my_block in itertools.groupby(scaffold, operator.itemgetter(0)):
            my_bins = list(my_block)

            if mode == "sequences":

                if len(my_bins) < 2:
                    new_scaffold.append(my_bins[0])
                    continue
                else:
                    previous_bin = []
                    end_bin = [-2, -2, -2, -2, -2]
                    current_ori = 0
                    # print(my_bins + end_bin)
                    for my_bin in my_bins + [end_bin]:
                        if not previous_bin:
                            # print("début")
                            previous_bin = copy.copy(my_bin)
                            continue
                        elif my_bin[1] == previous_bin[1] + 1:
                            current_ori = 1
                            previous_bin[-1] = 1
                            new_scaffold.append(previous_bin)
                            previous_bin = copy.copy(my_bin)
                        elif my_bin[1] == previous_bin[1] - 1:
                            current_ori = -1
                            previous_bin[-1] = -1
                            new_scaffold.append(previous_bin)
                            previous_bin = copy.copy(my_bin)
                        else:
                            if current_ori == 0:
                                new_scaffold.append(previous_bin)
                            else:
                                previous_bin[-1] = current_ori
                                new_scaffold.append(previous_bin)
                                current_ori = 0
                            previous_bin = copy.copy(my_bin)

                    assert previous_bin[0] == -2

            elif mode == "blocks":
                total_ori = sum([my_bin[-1] for my_bin in my_bins])

                if total_ori >= 0:
                    block_ori = 1
                    sorted_block = sorted(my_bins, key=operator.itemgetter(1))
                else:
                    block_ori = -1
                    sorted_block = sorted(
                        my_bins, key=operator.itemgetter(1), reverse=True
                    )

                for my_bin in sorted_block:
                    my_bin[-1] = block_ori
                    new_scaffold.append(my_bin)

        new_scaffolds[name] = copy.deepcopy(new_scaffold)

    return new_scaffolds


def write_info_frags(scaffolds, output="new_info_frags.txt"):

    scaffolds = format_info_frags(scaffolds)
    with open(output, "w") as info_frags_handle:
        for new_name, scaffold in scaffolds.items():
            info_frags_handle.write(">{}\n".format(new_name))
            header_line = "\t".join(
                ["init_contig", "id_frag", "orientation", "start", "end"]
            )
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


def write_fasta(
    init_fasta, info_frags, output=DEFAULT_NEW_GENOME_NAME, junction=False
):

    """Convert an info_frags.txt file into a fasta file given a reference.
    Optionally adds junction sequences to reflect the possibly missing base
    pairs between two newly joined scaffolds.
    """

    init_genome = {
        record.id: record.seq for record in SeqIO.parse(init_fasta, "fasta")
    }
    my_new_records = []
    with open(info_frags, "r") as info_frags_handle:
        current_seq = ""
        current_id = None
        previous_contig = None
        for line in info_frags_handle:
            if line.startswith(">"):
                previous_contig = None
                if current_id is not None:
                    new_record = SeqRecord(
                        current_seq, id=current_id, description=""
                    )
                    my_new_records.append(new_record)
                current_seq = ""
                current_id = str(line[1:])
            elif line.startswith("init_contig"):
                previous_contig = None
            else:
                (init_contig, _, orientation, pos_start, pos_end) = str(
                    line[:-1]
                ).split("\t")

                start = int(pos_start)
                end = int(pos_end)
                ori = int(orientation)

                assert start < end
                assert ori in {-1, 1}

                seq_to_add = init_genome[init_contig][start:end]
                if ori == 1:
                    current_seq += seq_to_add
                elif ori == -1:
                    current_seq += seq_to_add.reverse_complement()

                if junction and previous_contig not in {init_contig, None}:
                    error_was_raised = False
                    try:
                        extra_seq = Seq(junction, IUPAC.ambiguous_dna)
                        current_seq = extra_seq + current_seq
                    except TypeError:
                        if not error_was_raised:
                            print("Invalid junction sequence")
                            error_was_raised = True
                previous_contig = init_contig

        new_record = SeqRecord(current_seq, id=current_id, description="")
        my_new_records.append(new_record)
    SeqIO.write(my_new_records, output, "fasta")


def find_lost_dna(init_fasta, scaffolds, output_file=None):

    my_scaffolds = format_info_frags(scaffolds)
    my_records = sorted(
        SeqIO.parse(init_fasta, "fasta"), reverse=True, key=len
    )

    that_which_was_removed = {}
    fasta_dict = {}

    def consecutiveness(key_base_tuple):
        key, base = key_base_tuple
        return base - key

    for record in my_records:

        fasta_dict[record.id] = record.seq

        remaining_regions_ordered = range(len(record))
        remaining_regions = set(remaining_regions_ordered)
        regions = [
            my_bin
            for scaffold in my_scaffolds.values()
            for my_bin in scaffold
            if my_bin[0] == record.id
        ]
        for region in regions:
            start, end = region[2], region[3]
            remaining_regions -= set(
                remaining_regions_ordered[start : end + 1]
            )

        sorted_regions = sorted(remaining_regions)

        for _, g in itertools.groupby(
            enumerate(sorted_regions), consecutiveness
        ):

            swath = list(map(operator.itemgetter(1), g))
            start = min(swath)
            end = max(swath) + 1
            ori = 1
            my_bin = [record.id, -1, start, end, ori]
            try:
                that_which_was_removed[record.id].append(my_bin)
            except KeyError:
                that_which_was_removed[record.id] = [my_bin]

    if output_file:

        try:
            with open(output_file, "w") as output_handle:
                for name, chunks in that_which_was_removed.items():
                    for chunk in chunks:
                        try:
                            start, end = chunk[2], chunk[3]
                        except ValueError:
                            continue
                        header_line = ">{}_{}_{}\n".format(name, start, end)
                        output_handle.write(header_line)
                        sequence = fasta_dict[name][start:end]
                        sequence_line = "{}\n".format(sequence)
                        output_handle.write(sequence_line)
        except OSError:
            print("Couldn't write fasta file.")

    return that_which_was_removed


def integrate_lost_dna(scaffolds, lost_dna_positions):

    scaffolds = format_info_frags(scaffolds)
    remaining_dna_positions = copy.deepcopy(lost_dna_positions)
    new_scaffolds = {}
    for name, scaffold in scaffolds.items():
        scaffold_to_modify = copy.deepcopy(scaffold)

        i = 0
        for my_bin in scaffold:
            init_name = my_bin[0]

            try:
                lost_dna_chunk = lost_dna_positions[init_name]
                start = my_bin[2]
                end = my_bin[3]
                ori = my_bin[4]
                for lost_bin in lost_dna_chunk:

                    lost_start = lost_bin[2]
                    lost_end = lost_bin[3]

                    if end == lost_start - 1:
                        # print(
                        #     "A bin was reintegrated after position {}"
                        #     " in scaffold {}".format(my_bin[3], init_name)
                        # )
                        bin_to_add = [
                            init_name,
                            -1,
                            lost_start - 1,
                            lost_end + 1,
                            ori,
                        ]
                        scaffold_to_modify.insert(
                            i + 1 - (ori < 0), bin_to_add
                        )
                        remaining_dna_positions.pop(init_name)
                        i += 1

                    elif start in (lost_end, lost_end - 1, lost_end + 1):
                        # print(
                        #     "A bin was reintegrated before position {}"
                        #     " in scaffold {}".format(my_bin[3], init_name)
                        # )
                        bin_to_add = [init_name, -1, lost_start, lost_end, ori]
                        scaffold_to_modify.insert(i - 1, bin_to_add)
                        remaining_dna_positions.pop(init_name)
                        i += 1
            except (ValueError, KeyError):
                i += 1
                continue

            i += 1

        new_scaffolds[name] = copy.deepcopy(scaffold_to_modify)
    print("Appending the rest...")
    for remaining_name, remaining_bins in remaining_dna_positions.items():
        for my_bin in remaining_bins:
            try:
                remaining_bin = [remaining_name, -1, my_bin[2], my_bin[3], 1]
                new_scaffolds[remaining_name] = [remaining_bin]
            except ValueError:
                continue

    return new_scaffolds


def is_block(bin_list):

    """Check if a bin list has exclusively consecutive bin ids.
    """
    id_set = set((my_bin[1] for my_bin in bin_list))
    start_id, end_id = min(id_set), max(id_set)
    return id_set == set(range(start_id, end_id + 1))


def main():
    parser = argparse.ArgumentParser(
        description="Process 3C bin " "formalized scaffolds."
    )

    parser.add_argument("-m", "--mode", help="Process mode", required=True)

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input info_frags.txt to process",
        required=True,
    )

    parser.add_argument(
        "-f",
        "--fasta",
        type=str,
        help="Reference FASTA file to generate "
        "new genome with info_frags.txt",
    )

    parser.add_argument("-o", "--output", help="Output file to generate")

    parser.add_argument(
        "-c", "--criterion", type=str, help="Block criterion stringency"
    )

    parser.add_argument(
        "-s",
        "--min-scaffold-size",
        type=int,
        help="Minimum scaffold size in bins",
        default=DEFAULT_MIN_SCAFFOLD_SIZE,
    )

    parser.add_argument(
        "-j", "--junction", type=str, help="Junction sequence", default=""
    )

    args = parser.parse_args()

    info_frags = args.input
    min_size = args.min_scaffold_size
    scaffolds = {
        name: scaffold
        for (name, scaffold) in parse_info_frags(info_frags).items()
        if len(scaffold) > min_size
    }

    if args.mode == "fasta":
        init_fasta = args.fasta
        output_file = args.output
        junction = args.junction

        if init_fasta is None:
            print(
                "Error! An initial FASTA file must be provided to write "
                "the bins into sequences."
            )

        write_fasta(
            init_fasta=init_fasta,
            info_frags=info_frags,
            junction=junction,
            output=output_file,
        )

    elif "singleton" in args.mode:
        output_file = args.output
        new_scaffolds = remove_spurious_insertions(scaffolds)
        write_info_frags(new_scaffolds, output=output_file)

    elif args.mode == "inversion":
        output_file = args.output or DEFAULT_NEW_INFO_FRAGS_NAME
        criterion = args.criterion or DEFAULT_CRITERION

        new_scaffolds = correct_spurious_inversions(
            scaffolds=scaffolds, criterion=criterion
        )

        write_info_frags(new_scaffolds, output=output_file)

    elif args.mode == "inversion2":
        output_file = args.output or DEFAULT_NEW_INFO_FRAGS_NAME
        criterion = args.criterion or DEFAULT_CRITERION_2

        new_scaffolds = reorient_consecutive_blocks(
            scaffolds=scaffolds, mode=criterion
        )

        write_info_frags(new_scaffolds, output=output_file)

    elif "rearrange" in args.mode:
        output_file = args.output or DEFAULT_NEW_INFO_FRAGS_NAME

        new_scaffolds = rearrange_intra_scaffolds(scaffolds=scaffolds)

        write_info_frags(new_scaffolds, output=output_file)

    elif "reincorporation" in args.mode:
        init_fasta = args.fasta
        output_file = args.output or DEFAULT_NEW_INFO_FRAGS_NAME

        if init_fasta is None:
            print(
                "Error! An initial FASTA file must be provided"
                " for bin reincorporation."
            )

        removed = find_lost_dna(init_fasta=init_fasta, scaffolds=scaffolds)

        new_scaffolds = integrate_lost_dna(
            scaffolds=scaffolds, lost_dna_positions=removed
        )

        write_info_frags(new_scaffolds, output=output_file)

    elif "polishing" in args.mode:
        init_fasta = args.fasta
        output_file = args.output

        arranged_scaffolds = rearrange_intra_scaffolds(scaffolds=scaffolds)
        reoriented_scaffolds = reorient_consecutive_blocks(arranged_scaffolds)
        removed = find_lost_dna(
            init_fasta=init_fasta, scaffolds=reoriented_scaffolds
        )
        new_scaffolds = integrate_lost_dna(
            scaffolds=reoriented_scaffolds, lost_dna_positions=removed
        )

        write_info_frags(new_scaffolds, output=DEFAULT_NEW_INFO_FRAGS_NAME)
        write_fasta(
            init_fasta=init_fasta,
            info_frags=DEFAULT_NEW_INFO_FRAGS_NAME,
            output=output_file,
        )

    elif args.mode == "plot":
        plot_info_frags(scaffolds)

    else:
        print(
            "Wrong mode. Available modes are: "
            "fasta, singletons, inversions, inversion2, rearrange, "
            "reincorporations, polishing, plot"
        )


if __name__ == "__main__":
    main()
