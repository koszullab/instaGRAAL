#!/usr/bin/env python
# coding: utf-8

"""Aggregate genetic linkage data with an existing assembly for correction and
polishing purposes.

Usage:
    linkage.py <linkage.csv> <info_frags.txt> [--output <output_file>]
                                              [--fasta <reference.fa>]

Options:
    -h, --help              Display this help message.
    --version               Display the program's current version.
    -o, --output            Prefix of output files
    -f, --fasta             If a fasta reference is supplied, also write
                            the corresponding genome from info_frags.txt

"""

import functools
import itertools
import operator
import collections
import copy
import docopt
import numpy as np
import parse_info_frags

VERSION_NUMBER = "0.1a"

parse_linkage_csv = functools.partial(
    np.genfromtxt, dtype=None, skip_header=1, encoding="utf-8"
)


def collapse_degenerate_markers(linkage_records):

    """Group all markers with no genetic distance as distinct features
    to generate a BED file with.

    Simple example with sixteen degenerate markers:

        >>> marker_features = [
        ... ['36915_sctg_207_31842', 1, 0, 207, 31842],
        ... ['36941_sctg_207_61615', 1, 0, 207, 61615],
        ... ['36956_sctg_207_77757', 1, 0, 207, 77757],
        ... ['36957_sctg_207_78332', 1, 0, 207, 78332],
        ... ['36972_sctg_207_94039', 1, 0, 207, 94039],
        ... ['36788_sctg_207_116303', 1, 0.652, 207, 116303],
        ... ['36812_sctg_207_158925', 1, 1.25, 207, 158925],
        ... ['36819_sctg_207_165424', 1, 1.25, 207, 165424],
        ... ['36828_sctg_207_190813', 1, 1.25, 207, 190813],
        ... ['36830_sctg_207_191645', 1, 1.25, 207, 191645],
        ... ['36834_sctg_207_195961', 1, 1.25, 207, 195961],
        ... ['36855_sctg_207_233632', 1, 1.25, 207, 233632],
        ... ['36881_sctg_207_258658', 1, 1.25, 207, 258658],
        ... ['82072_sctg_486_41893', 1, 3.756, 486, 41893],
        ... ['85634_sctg_516_36614', 1,	3.756, 516, 36614],
        ... ['85638_sctg_516_50582', 1, 3.756, 516, 50582]]

        >>> len(marker_features)
        16

        >>> collapsed_features = collapse_degenerate_markers(marker_features)
        >>> len(collapsed_features)
        5

    The degenerate features (identical linkage group, genetic distance and
    original scaffold) are collapsed into a region:

        >>> collapsed_features[0]
        [1, 31842, 94039, 207]

    The format is [linkage group, start, end, original scaffold].

    If a singleton (non-degenerate) feature is found, the region is simply
    a single point in the genome:

        >>> collapsed_features[1]
        [1, 116303, 116303, 207]

    so 'start' and 'end' are identical.

    Two markers are not considered degenerate if they belong to different
    original scaffolds, even if they are in terms of genetic linkage:

        >>> collapsed_features[2]
        [1, 158925, 258658, 207]
        >>> collapsed_features[3:]
        [[1, 41893, 41893, 486], [1, 36614, 50582, 516]]
    """

    def degeneracy(linkage_record):
        linkage_group, genetic_distance, scaffold = (
            linkage_record[1],
            linkage_record[2],
            linkage_record[3],
        )
        return (linkage_group, genetic_distance, scaffold)

    degenerate_records = []
    for _, degenerate_group in itertools.groupby(
        linkage_records, key=degeneracy
    ):
        group_list = list(degenerate_group)
        start_record, end_record = group_list[0], group_list[-1]

        assert (start_record[1], start_record[2], start_record[3]) == (
            end_record[1],
            end_record[2],
            end_record[3],
        )

        start_position = start_record[-1]
        end_position = end_record[-1]

        scaffold = start_record[3]
        linkage_group = start_record[1]
        record = [linkage_group, start_position, end_position, scaffold]

        degenerate_records.append(record)

    return degenerate_records


def write_bed(records, output_file):
    with open(output_file, "w") as bed_handle:
        for record in records:
            line = "\t".join((str(field) for field in record))
            bed_handle.write("{}\n".format(line))


def linkage_group_ordering(linkage_records):
    """Convert degenerate linkage records into ordered info_frags-like records
    for comparison purposes.

    Simple example:

        >>> linkage_records = [
        ... ['linkage_group_1', 31842, 94039, 'sctg_207'],
        ... ['linkage_group_1', 95303, 95303, 'sctg_207'],
        ... ['linkage_group_2', 15892, 25865, 'sctg_308'],
        ... ['linkage_group_2', 41893, 41893, 'sctg_486'],
        ... ['linkage_group_3', 36614, 50582, 'sctg_516'],
        ... ]
        >>> ordering = linkage_group_ordering(linkage_records)

    Each key of the record is a newly-formed 'scaffold' (linkage group):

        >>> sorted(ordering.keys())
        ['linkage_group_1', 'linkage_group_2', 'linkage_group_3']

    Records are in the form [init_contig, frag_id, start, end, orientation].
    Since fragment ids are meaningless in non-HiC contexts a negative
    identifier is set so it is understood that region was added due to
    linkage data (-1 is for recovering data after first-pass polishing and -2
    is for sequence insertions after long read based polishing).

        >>> ordering['linkage_group_1']
        [['sctg_207', -3, 31842, 94039, 1], ['sctg_207', -3, 95303, 95303, 1]]
        >>> ordering['linkage_group_2']
        [['sctg_308', -3, 15892, 25865, 1], ['sctg_486', -3, 41893, 41893, 1]]

    Orientations are always set to 1 by default.

        >>> ordering['linkage_group_3']
        [['sctg_516', -3, 36614, 50582, 1]]


    """

    new_records = dict()

    for lg_name, linkage_group in itertools.groupby(
        linkage_records, operator.itemgetter(0)
    ):
        new_records[lg_name] = []
        for record in linkage_group:
            init_contig = record[-1]
            start = record[1]
            end = record[2]

            new_record = [init_contig, -3, start, end, 1]

            new_records[lg_name].append(new_record)

    return new_records


def compare_orderings(info_frags_records, linkage_orderings):

    """Given linkage groups and info_frags records, link pseudo-chromosomes to
    scaffolds based on the initial contig composition of each group. Because
    info_frags records are usually richer and may contain contigs not found
    in linkage groups, those extra sequences are discarded.

    Example with two linkage groups and two chromosomes:

        >>> linkage_orderings = {
        ...     'linkage_group_1': [
        ...         ['sctg_516', -3, 36614, 50582, 1],
        ...         ['sctg_486', -3, 41893, 41893, 1],
        ...         ['sctg_486', -3, 50054, 62841, 1],
        ...         ['sctg_207', -3, 31842, 94039, 1],
        ...         ['sctg_558', -3, 51212, 54212, 1],
        ...     ],
        ...     'linkage_group_2': [
        ...         ['sctg_308', -3, 15892, 25865, 1],
        ...         ['sctg_842', -3, 0, 8974, 1],
        ...         ['sctg_994', -3, 0, 81213, 1],
        ...     ],
        ... }
        >>> info_frags = {
        ...     'scaffold_A': [
        ...         ['sctg_308', 996, 15892, 25865, 1],
        ...         ['sctg_778', 1210, 45040, 78112, -1],
        ...         ['sctg_842', 124, 0, 8974, 1],
        ...     ],
        ...     'scaffold_B': [
        ...         ['sctg_516', 5, 0, 38000, 1],
        ...         ['sctg_486', 47, 42050, 49000, 1],
        ...         ['sctg_1755', 878, 95001, 10844, -1],
        ...         ['sctg_842', 126, 19000, 26084, 1],
        ...         ['sctg_207', 705, 45500, 87056, 1],
        ...     ],
        ...     'scaffold_C': [
        ...        ['sctg_558', 745, 50045, 67851, 1],
        ...        ['sctg_994', 12, 74201, 86010, -1],
        ...     ],
        ... }
        >>> matching_pairs = compare_orderings(info_frags, linkage_orderings)
        >>> matching_pairs['scaffold_B']
        (3, 'linkage_group_1', {'sctg_558': 'sctg_207'})
        >>> matching_pairs['scaffold_A']
        (2, 'linkage_group_2', {'sctg_994': 'sctg_842'})

    """

    scaffolds = info_frags_records.keys()
    linkage_groups = linkage_orderings.keys()

    best_matching_table = dict()

    for scaffold, linkage_group in itertools.product(
        scaffolds, linkage_groups
    ):
        lg_ordering = [
            init_contig
            for init_contig, _ in itertools.groupby(
                linkage_orderings[linkage_group], operator.itemgetter(0)
            )
        ]
        scaffold_ordering = [
            init_contig
            for init_contig, bin_group in itertools.groupby(
                info_frags_records[scaffold], operator.itemgetter(0)
            )
            if init_contig in lg_ordering
        ]

        overlap = set(lg_ordering).intersection(set(scaffold_ordering))
        missing_locations = dict()
        for missing_block in sorted(set(lg_ordering) - set(overlap)):
            for i, init_contig in enumerate(lg_ordering):
                if init_contig == missing_block:
                    try:
                        block_before = lg_ordering[i - 1]
                    except IndexError:
                        block_before = "beginning"
                    missing_locations[missing_block] = block_before

        try:
            if len(overlap) > best_matching_table[scaffold][0]:
                best_matching_table[scaffold] = (
                    len(overlap),
                    linkage_group,
                    missing_locations,
                )

        except KeyError:
            best_matching_table[scaffold] = (
                len(overlap),
                linkage_group,
                missing_locations,
            )

    return best_matching_table


def get_missing_blocks(info_frags_records, matching_pairs, linkage_orderings):
    """[summary]

    [description]

    Parameters
    ----------
    info_frags_records : dict
        [description]
    linkage_orderings : dict
        [description]
    matching_pairs : dict
        [description]

    Example
    -------

        >>> linkage_orderings = {
        ...     'linkage_group_1': [
        ...         ['sctg_516', -3, 36614, 50582, 1],
        ...         ['sctg_486', -3, 41893, 41893, 1],
        ...         ['sctg_486', -3, 50054, 62841, 1],
        ...         ['sctg_207', -3, 31842, 94039, 1],
        ...         ['sctg_558', -3, 51212, 54212, 1],
        ...     ],
        ...     'linkage_group_2': [
        ...         ['sctg_308', -3, 15892, 25865, 1],
        ...         ['sctg_842', -3, 0, 8974, 1],
        ...         ['sctg_994', -3, 0, 81213, 1],
        ...     ],
        ... }
        >>> info_frags = {
        ...     'scaffold_A': [
        ...         ['sctg_308', 996, 15892, 25865, 1],
        ...         ['sctg_778', 1210, 45040, 78112, -1],
        ...         ['sctg_842', 124, 0, 8974, 1],
        ...     ],
        ...     'scaffold_B': [
        ...         ['sctg_516', 5, 0, 38000, 1],
        ...         ['sctg_486', 47, 42050, 49000, 1],
        ...         ['sctg_1755', 878, 95001, 10844, -1],
        ...         ['sctg_842', 126, 19000, 26084, 1],
        ...         ['sctg_207', 705, 45500, 87056, 1],
        ...     ],
        ...     'scaffold_C': [
        ...        ['sctg_558', 745, 50045, 67851, 1],
        ...        ['sctg_994', 12, 74201, 86010, -1],
        ...     ],
        ... }
        >>> matching_pairs = compare_orderings(info_frags, linkage_orderings)
        >>> new_records = get_missing_blocks(info_frags, matching_pairs,
        ...                                  linkage_orderings)
        >>> for my_bin in new_records['scaffold_A']:
        ...     print(list(my_bin))
        ...
        ['sctg_308', 996, 15892, 25865, 1]
        ['sctg_778', 1210, 45040, 78112, -1]
        ['sctg_842', 124, 0, 8974, 1]
        ['sctg_842', 126, 19000, 26084, 1]
        ['sctg_994', 12, 74201, 86010, -1]

        >>> for my_bin in new_records['scaffold_B']:
        ...     print(list(my_bin))
        ...
        ['sctg_516', 5, 0, 38000, 1]
        ['sctg_486', 47, 42050, 49000, 1]
        ['sctg_1755', 878, 95001, 10844, -1]
        ['sctg_207', 705, 45500, 87056, 1]
        ['sctg_558', 745, 50045, 67851, 1]

    """

    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    touched_lgs = set()

    def record_length(key_record_tuple):
        return len(key_record_tuple[1])

    new_scaffolds = copy.deepcopy(info_frags_records)

    for scaffold_name in collections.OrderedDict(
        sorted(new_scaffolds.items(), key=record_length, reverse=True)
    ):
        scaffold = new_scaffolds[scaffold_name]
        new_scaffold = []
        corresponding_lg = matching_pairs[scaffold_name][1]
        if corresponding_lg in touched_lgs:
            continue
        else:
            touched_lgs.add(corresponding_lg)
        scaffold_block_names = {my_bin[0] for my_bin in scaffold}
        lg_block_names = [
            my_block[0] for my_block in linkage_orderings[corresponding_lg]
        ]

        touched_bins = set()

        for lg_block_name in lg_block_names:
            if lg_block_name in scaffold_block_names:
                for my_bin in scaffold:
                    if tuple(my_bin) in new_scaffold:
                        continue
                    elif (
                        my_bin[0] == lg_block_name
                        or my_bin[0] not in lg_block_names
                    ):
                        new_scaffold.append(tuple(my_bin))
                        touched_bins.add(tuple(my_bin))
                    else:
                        break
            for other_name, other_scaffold in new_scaffolds.items():
                if other_name == scaffold_name:
                    continue
                i = 0
                for my_bin in other_scaffold:
                    if tuple(my_bin) in new_scaffold:
                        i += 1
                        continue
                    elif my_bin[0] == lg_block_name:
                        moved_bin = tuple(other_scaffold.pop(i))
                        new_scaffold.append(tuple(moved_bin))
                        touched_bins.add(tuple(moved_bin))
                        i -= 1
                    i += 1

        for remaining_bin in scaffold:
            if tuple(remaining_bin) not in touched_bins:
                new_scaffold.append(tuple(remaining_bin))
                touched_bins.add(tuple(remaining_bin))

        if len(new_scaffold) > 0:
            new_scaffolds[scaffold_name] = new_scaffold

    return new_scaffolds


def main():
    arguments = docopt.docopt(__doc__, version=VERSION_NUMBER)

    linkage_file = arguments["<linkage.csv>"]
    info_frags_file = arguments["<info_frags.txt>"]
    output = arguments["<output_file>"]
    fasta = arguments["--fasta"]

    output_fasta = "{}_linkage.fa".format(output)

    linkage_csv = parse_linkage_csv(linkage_file)
    info_frags_records = parse_info_frags.parse_info_frags(info_frags_file)
    collapsed_linkage = collapse_degenerate_markers(linkage_csv)
    group_ordering = linkage_group_ordering(collapsed_linkage)
    matching_table = compare_orderings(info_frags_records, group_ordering)
    new_scaffolds = get_missing_blocks(
        info_frags_records, matching_table, group_ordering
    )

    parse_info_frags.write_info_frags(new_scaffolds, output=output)

    if fasta:
        parse_info_frags.write_fasta(fasta, new_scaffolds, output=output_fasta)


if __name__ == "__main__":
    main()
