#!/usr/bin/env python
# coding: utf-8

"""Scaffold correction algorithms: inversion, rearrangement, DNA recovery.

This module is internal; import from ``instagraal.parse_info_frags`` instead.
"""

import copy
import itertools
import operator

from ._scaffold_io import _parse_fasta, format_info_frags

# ---------------------------------------------------------------------------
# Overlap / merge helpers (used internally by correct_scaffolds)
# ---------------------------------------------------------------------------


def _are_overlapping(bin1, bin2):
    """Check for overlapping regions between two bins."""
    if bin2 is None:
        return False
    init1, _, start1, end1, _ = bin1
    init2, _, start2, end2, _ = bin2
    if init1 != init2:
        return False
    return (start2 <= start1 <= end2) or (start1 <= start2 <= end1)


def _merge_bins(bin1, bin2, ignore_ori=True):
    """Merge two overlapping bins, picking the widest span."""
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
        ori = ori2 if len2 > len1 else ori1
    return [init1, -3, start, end, ori]


# ---------------------------------------------------------------------------
# Public correction functions
# ---------------------------------------------------------------------------


def correct_scaffolds(scaffolds, corrector):
    """Unfinished"""

    new_scaffolds = {}
    corrector_bins = copy.deepcopy(corrector)

    for name, scaffold in scaffolds.items():
        new_scaffold = []

        for _, blocks in itertools.groupby(scaffold, operator.itemgetter(0)):
            merged_bin = None
            while "Reading blocks":
                try:
                    my_bin = next(blocks)  # noqa: B031
                    if _are_overlapping(my_bin, merged_bin):
                        merged_bin = _merge_bins(my_bin, merged_bin)
                        continue
                    else:
                        if merged_bin is not None:
                            new_scaffold.append(merged_bin)
                        merged_bin = my_bin
                        i = 0
                        for correct_bin in corrector_bins:
                            if _are_overlapping(my_bin, correct_bin):
                                merged_bin = _merge_bins(my_bin, correct_bin)
                                corrector_bins.pop(i)
                                i -= 1
                            i += 1
                except StopIteration:
                    if merged_bin is not None:
                        new_scaffold.append(merged_bin)
                    break

        new_scaffolds[name] = new_scaffold

    return new_scaffolds


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
                    if not (scaffold[i][0] != scaffold[i + 1][0] and scaffold[i + 1][0] == scaffold[i + 2][0]):
                        new_scaffold.append(scaffold[i])
                elif i == len(scaffold) - 1:
                    if not (scaffold[i][0] != scaffold[i - 1][0] and scaffold[i - 1][0] == scaffold[i - 2][0]):
                        new_scaffold.append(scaffold[i])
                # Otherwise, looking for -*-
                else:
                    if not (scaffold[i - 1][0] == scaffold[i + 1][0] and scaffold[i - 1][0] != scaffold[i][0]):
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
    try:
        block_test = condition_callables[criterion]
    except KeyError:
        raise ValueError("Unknown criterion {!r}, expected one of {}".format(criterion, ", ".join(condition_callables))) from None

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

        def block_order(block, _ordering=ordering):
            return _ordering[block[0][0]][0]

        for my_block in sorted(my_blocks, key=block_order):
            for my_bin in my_block:
                new_scaffold.append(my_bin)

        new_scaffolds[name] = copy.deepcopy(new_scaffold)

    return new_scaffolds


REORIENT_MODES = ("blocks", "sequences")


def reorient_consecutive_blocks(scaffolds, mode="blocks"):
    """Reorient runs of consecutive bins that come from the same initial contig.

    In ``"blocks"`` mode, each run is sorted back into its order on the initial
    contig and all of its bins are given the run's majority orientation. In
    ``"sequences"`` mode, bins are reoriented according to whether their
    neighbours' ids increase or decrease, without reordering.
    """

    if mode not in REORIENT_MODES:
        raise ValueError("Unknown reorientation mode {!r}, expected one of {}".format(mode, ", ".join(REORIENT_MODES)))

    scaffolds = format_info_frags(scaffolds)
    new_scaffolds = {}

    for name, scaffold in scaffolds.items():
        new_scaffold = []
        for _, my_block in itertools.groupby(scaffold, operator.itemgetter(0)):
            my_bins = [list(my_bin) for my_bin in my_block]

            if mode == "sequences":
                if len(my_bins) < 2:
                    new_scaffold.append(my_bins[0])
                    continue
                else:
                    previous_bin = []
                    end_bin = [-2, -2, -2, -2, -2]
                    current_ori = 0
                    for my_bin in [*my_bins, end_bin]:
                        if not previous_bin:
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

                # Sort on the position on the initial contig rather than on the
                # bin id, which is -1 for bins put back by integrate_lost_dna.
                position = operator.itemgetter(2, 3)
                if total_ori >= 0:
                    block_ori = 1
                    sorted_block = sorted(my_bins, key=position)
                else:
                    block_ori = -1
                    sorted_block = sorted(my_bins, key=position, reverse=True)

                for my_bin in sorted_block:
                    my_bin[-1] = block_ori
                    new_scaffold.append(my_bin)

        new_scaffolds[name] = copy.deepcopy(new_scaffold)

    return new_scaffolds


# ---------------------------------------------------------------------------
# DNA recovery
# ---------------------------------------------------------------------------


def find_lost_dna(init_fasta, scaffolds, output_file=None):
    """Find the regions of the initial contigs that no scaffold bin covers.

    Bins are 0-based, half-open ``[start, end)`` intervals, so a bin ending
    at position ``p`` and a lost region starting at ``p`` are contiguous.

    Returns a dictionary mapping each initial contig with uncovered DNA to the
    list of its uncovered regions, sorted by position, as bins
    ``[contig, -1, start, end, 1]``. A contig that no bin covers at all is
    returned as a single ``[contig, -1, 0, length, 1]`` region. Contigs are
    listed from longest to shortest.

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as fa:
        ...     _ = fa.write(">ctgA\\n" + "A" * 1000 + "\\n>ctgB\\n" + "C" * 50 + "\\n")
        >>> scaffolds = {"s1": [["ctgA", 0, 100, 300, 1], ["ctgA", 2, 500, 900, -1]]}
        >>> for name, chunks in find_lost_dna(fa.name, scaffolds).items():
        ...     print(name, chunks)
        ctgA [['ctgA', -1, 0, 100, 1], ['ctgA', -1, 300, 500, 1], ['ctgA', -1, 900, 1000, 1]]
        ctgB [['ctgB', -1, 0, 50, 1]]
    """

    my_scaffolds = format_info_frags(scaffolds)

    covered = {}
    for scaffold in my_scaffolds.values():
        for my_bin in scaffold:
            covered.setdefault(my_bin[0], []).append((my_bin[2], my_bin[3]))

    fasta_dict = {}
    lengths = {}
    for record in _parse_fasta(init_fasta):
        lengths[record.id] = len(record)
        if output_file:
            fasta_dict[record.id] = record.seq

    that_which_was_removed = {}
    for name in sorted(lengths, key=lambda contig: lengths[contig], reverse=True):
        length = lengths[name]
        chunks = []
        cursor = 0
        for start, end in sorted(covered.get(name, [])):
            if start > cursor:
                chunks.append([name, -1, cursor, min(start, length), 1])
            cursor = max(cursor, end)
            if cursor >= length:
                break
        if cursor < length:
            chunks.append([name, -1, cursor, length, 1])
        if chunks:
            that_which_was_removed[name] = chunks

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
    """Put the DNA found by :func:`find_lost_dna` back into the assembly.

    Each lost region is used exactly once. It is attached to a bin it is
    contiguous with on its initial contig, in that bin's orientation: a region
    that starts where a bin ends follows that bin on the contig, and a region
    that ends where a bin starts precedes it. Regions contiguous with no bin
    (e.g. contigs that no bin covers at all) become new scaffolds, one per
    region, named after their contig (``<contig>_<start>_<end>`` when the
    contig is otherwise present in the assembly).

        >>> scaffolds = {
        ...     "s1": [["ctgA", 0, 100, 300, 1], ["ctgB", 0, 0, 50, 1]],
        ...     "s2": [["ctgA", 2, 500, 900, -1]],
        ... }
        >>> lost = {
        ...     "ctgA": [["ctgA", -1, 0, 100, 1], ["ctgA", -1, 300, 500, 1], ["ctgA", -1, 900, 1000, 1]],
        ...     "ctgC": [["ctgC", -1, 0, 80, 1]],
        ... }
        >>> for name, scaffold in integrate_lost_dna(scaffolds, lost).items():
        ...     print(name, scaffold)
        s1 [['ctgA', -1, 0, 100, 1], ['ctgA', 0, 100, 300, 1], ['ctgA', -1, 300, 500, 1], ['ctgB', 0, 0, 50, 1]]
        s2 [['ctgA', -1, 900, 1000, -1], ['ctgA', 2, 500, 900, -1]]
        ctgC [['ctgC', -1, 0, 80, 1]]
    """

    scaffolds = format_info_frags(scaffolds)

    starting_at = {}
    ending_at = {}
    for contig, chunks in lost_dna_positions.items():
        for chunk in chunks:
            key = (contig, chunk[2], chunk[3])
            starting_at[(contig, chunk[2])] = key
            ending_at[(contig, chunk[3])] = key
    used = set()

    def take(key, ori):
        if key is None or key in used:
            return []
        used.add(key)
        contig, start, end = key
        return [[contig, -1, start, end, ori]]

    new_scaffolds = {}
    for name, scaffold in scaffolds.items():
        new_scaffold = []
        for my_bin in scaffold:
            contig, _, start, end, ori = my_bin
            upstream = ending_at.get((contig, start))
            downstream = starting_at.get((contig, end))
            # Claim the region downstream on the contig first, then upstream.
            downstream_bins = take(downstream, ori)
            upstream_bins = take(upstream, ori)
            if ori == 1:
                new_scaffold += [*upstream_bins, list(my_bin), *downstream_bins]
            else:
                new_scaffold += [*downstream_bins, list(my_bin), *upstream_bins]
        new_scaffolds[name] = new_scaffold

    contigs_in_assembly = {my_bin[0] for scaffold in scaffolds.values() for my_bin in scaffold}
    for contig, chunks in lost_dna_positions.items():
        left = [chunk for chunk in chunks if (contig, chunk[2], chunk[3]) not in used]
        for chunk in left:
            if contig in contigs_in_assembly or len(chunks) > 1:
                new_name = "{}_{}_{}".format(contig, chunk[2], chunk[3])
            else:
                new_name = contig
            while new_name in new_scaffolds:
                new_name += "_lost"
            new_scaffolds[new_name] = [[contig, -1, chunk[2], chunk[3], 1]]

    return new_scaffolds


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def is_block(bin_list):
    """Check if a bin list has exclusively consecutive bin ids."""
    id_set = set((my_bin[1] for my_bin in bin_list))
    start_id, end_id = min(id_set), max(id_set)
    return id_set == set(range(start_id, end_id + 1))
