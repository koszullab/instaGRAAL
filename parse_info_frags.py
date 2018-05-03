#!/usr/bin/env python
# coding: utf-8

"""
A couple of functions to manipulate info_frags.txt files
"""

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq, IUPAC
from matplotlib import pyplot as plt
import argparse
import copy
import itertools
import operator

DEFAULT_MIN_SCAFFOLD_SIZE = 0
DEFAULT_NEW_INFO_FRAGS_NAME = "new_info_frags.txt"
DEFAULT_NEW_GENOME_NAME = "new_genome.fa"
DEFAULT_JUNCTION_SEQUENCE = Seq('NNNNNN', IUPAC.ambiguous_dna)
DEFAULT_CRITERION = 'colinear'
DEFAULT_CRITERION_2 = 'blocks'


def parse_info_frags(info_frags):

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
                (init_contig,
                 id_frag,
                 orientation,
                 pos_start,
                 pos_end) = str(line[:-1]).split("\t")
                start = int(pos_start)
                end = int(pos_end)
                ori = int(orientation)
                fragid = int(id_frag)
                assert start < end
                assert ori in {-1, 1}
                new_scaffolds[current_new_contig].append([init_contig,
                                                          fragid,
                                                          start,
                                                          end,
                                                          ori])

    return new_scaffolds


def parse_bed(bed_file):

    new_scaffolds = {}
    with open(bed_file) as bed_handle:
        for line in bed_handle:
            chrom, start, end, query, qual, strand = line.split()[:7]
            if strand == '+':
                ori = 1
            elif strand == '-':
                ori = -1
            else:
                raise TypeError("Error when parsing strand "
                                "orientation: {}".format(strand))

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

        if bin2 is None:
            return False

        init1, fragid1, start1, end1, ori1 = bin1
        init2, fragid2, start2, end2, ori2 = bin2

        if init1 != init2:
            return False
        else:
            return (start2 <= start1 <= end2) or (start1 <= start2 <= end1)

    def merge_bins(bin1, bin2, ignore_ori=True):
        init1, fragid1, start1, end1, ori1 = bin1
        init2, fragid2, start2, end2, ori2 = bin2

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

    for name, scaffold in scaffolds.iteritems():
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

    scaffolds = format_info_frags(scaffolds)
    for name, scaffold in scaffolds.iteritems():
        plt.figure()
        xs = range(len(scaffold))
        color = []
        names = {}
        ys = []
        for my_bin in scaffold:
            current_color = 'r' if my_bin[4] > 0 else 'g'
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

    scaffolds = format_info_frags(scaffolds)
    new_scaffolds = {}
    for name, scaffold in scaffolds.iteritems():
        new_scaffold = []
        if len(scaffold) > 2:
            for i in range(len(scaffold)):
                # First take care of edge cases: *-- or --*
                if i == 0:
                    if not (scaffold[i][0] != scaffold[i + 1][0] and
                            scaffold[i + 1][0] == scaffold[i + 2][0]):
                        new_scaffold.append(scaffold[i])
                elif i == len(scaffold) - 1:
                    if not (scaffold[i][0] != scaffold[i - 1][0] and
                            scaffold[i - 1][0] == scaffold[i - 2][0]):
                        new_scaffold.append(scaffold[i])
                # Otherwise, looking for -*-
                else:
                    if not (scaffold[i - 1][0] == scaffold[i + 1][0] and
                            scaffold[i - 1][0] != scaffold[i][0]):
                        new_scaffold.append(scaffold[i])
        else:
            # Can't remove insertions if 2 bins or less
            new_scaffold = copy.deepcopy(scaffold)

        new_scaffolds[name] = new_scaffold

    return new_scaffolds


def correct_spurious_inversions(scaffolds,
                                method="blocks",
                                criterion="colinear"):

    scaffolds = format_info_frags(scaffolds)
    new_scaffolds = {}

    def is_cis(bin1, bin2):
        return bin1[0] == bin2[0]

    def is_contiguous(bin1, bin2):
        return is_cis(bin1, bin2) and bin1[-3] == bin2[-2]

    def is_colinear(bin1, bin2):
        return is_cis(bin1, bin2) and bin1[-3] <= bin2[-2]

    condition_callables = {"cis": is_cis,
                           "colinear": is_colinear,
                           "contiguous": is_contiguous}
    block_test = condition_callables.get(criterion, "colinear")
    # print("You have chosen that a block "
    #       "only be composed of {} bins".format(criterion))

    for name, scaffold in scaffolds.iteritems():
        # print("This is scaffold %s" % name)
        new_scaffold = []

        block_cumulative_ori = 0
        if len(scaffold) > 2:
            current_bin = scaffold[0]
            block_buffer = []
            for my_bin in scaffold:
                # print(my_bin)
                if not block_buffer:
                    new_bin = copy.deepcopy(my_bin)
                    block_buffer.append(new_bin)
                    block_cumulative_ori = my_bin[-1]
                    continue
                elif not block_test(my_bin, current_bin):
                    # print(my_bin[0], current_bin[0])
                    for my_buf_bin in block_buffer:
                        # print("Writing new bin: %s" % my_bin)
                        new_bin = copy.deepcopy(my_buf_bin)
                        new_bin[-1] = ((block_cumulative_ori >= 0) -
                                       (block_cumulative_ori < 0))
                        new_scaffold.append(new_bin)
                    block_cumulative_ori = 0
                    current_bin = copy.deepcopy(my_bin)
                    block_buffer = copy.deepcopy([my_bin])
                else:
                    block_cumulative_ori += my_bin[-1]
                    new_bin = copy.deepcopy(my_bin)
                    block_buffer.append(new_bin)
                    current_bin = my_bin
            for my_bin in block_buffer:
                # print("Writing new bin: %s" % my_bin)
                new_bin = copy.deepcopy(my_bin)
                new_bin[-1] = ((block_cumulative_ori >= 0) -
                               (block_cumulative_ori < 0))
                new_scaffold.append(new_bin)
            new_scaffolds[name] = copy.deepcopy(new_scaffold)
        else:
            new_scaffolds[name] = copy.deepcopy(scaffold)

    return new_scaffolds


def rearrange_intra_scaffolds(scaffolds):

    scaffolds = format_info_frags(scaffolds)
    new_scaffolds = {}

    for name, scaffold in scaffolds.iteritems():
        new_scaffold = []

        ordering = dict()
        order = 0

        my_blocks = []

        for k, my_block in itertools.groupby(scaffold,
                                             operator.itemgetter(0)):

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


def reorient_consecutive_blocks(scaffolds, mode='blocks'):

    scaffolds = format_info_frags(scaffolds)
    new_scaffolds = {}

    for name, scaffold in scaffolds.iteritems():
        # print(scaffold)
        new_scaffold = []
        for k, my_block in itertools.groupby(scaffold,
                                             operator.itemgetter(0)):
            my_bins = list(my_block)

            if mode == 'sequences':

                if len(my_bins) < 2:
                    new_scaffold.append(my_bins[0])
                    continue
                else:
                    previous_bin = None
                    end_bin = [-2, -2, -2, -2, -2]
                    current_ori = 0
                    # print(my_bins + end_bin)
                    for my_bin in my_bins + [end_bin]:
                        if previous_bin is None:
                            # print("début")
                            previous_bin = copy.copy(my_bin)
                            continue
                        elif my_bin[1] == previous_bin[1] + 1:
                            # print("suite croissante")
                            current_ori = 1
                            previous_bin[-1] = 1
                            new_scaffold.append(previous_bin)
                            previous_bin = copy.copy(my_bin)
                        elif my_bin[1] == previous_bin[1] - 1:
                            # print("suite décroissante")
                            current_ori = -1
                            previous_bin[-1] = -1
                            new_scaffold.append(previous_bin)
                            previous_bin = copy.copy(my_bin)
                        else:
                            # print("fin de suite")
                            if current_ori == 0:
                                new_scaffold.append(previous_bin)
                            else:
                                previous_bin[-1] = current_ori
                                new_scaffold.append(previous_bin)
                                current_ori = 0
                            previous_bin = copy.copy(my_bin)

                    assert previous_bin[0] == -2

            elif mode == 'blocks':
                total_ori = sum([my_bin[-1] for my_bin in my_bins])

                if total_ori >= 0:
                    block_ori = 1
                    sorted_block = sorted(my_bins, key=operator.itemgetter(1))
                else:
                    block_ori = -1
                    sorted_block = sorted(my_bins,
                                          key=operator.itemgetter(1),
                                          reverse=True)

                for my_bin in sorted_block:
                    my_bin[-1] = block_ori
                    new_scaffold.append(my_bin)

        new_scaffolds[name] = copy.deepcopy(new_scaffold)

    return new_scaffolds


def write_info_frags(scaffolds, output="new_info_frags.txt"):

    scaffolds = format_info_frags(scaffolds)
    with open(output, "w") as info_frags_handle:
        for new_name, scaffold in scaffolds.iteritems():
            info_frags_handle.write(">{}\n".format(new_name))
            header_line = "\t".join(["init_contig",
                                     "id_frag",
                                     "orientation",
                                     "start",
                                     "end"])
            info_frags_handle.write("{}\n".format(header_line))
            for my_bin in scaffold:
                init_contig, id_frag, pos_start, pos_end, orientation = my_bin
                assert orientation in {-1, 1}
                my_line = "\t".join([str(init_contig),
                                     str(id_frag),
                                     str(orientation),
                                     str(pos_start),
                                     str(pos_end)])
                info_frags_handle.write("{}\n".format(my_line))


def write_fasta(init_fasta, info_frags,
                output=DEFAULT_NEW_GENOME_NAME,
                junction=False):

    init_genome = {record.id: record.seq
                   for record in SeqIO.parse(init_fasta, "fasta")}
    my_new_records = []
    with open(info_frags, "r") as info_frags_handle:
        current_seq = ""
        current_id = None
        previous_contig = None
        for line in info_frags_handle:
            if line.startswith(">"):
                previous_contig = None
                if current_id is not None:
                    new_record = SeqRecord(current_seq,
                                           id=current_id,
                                           description="")
                    my_new_records.append(new_record)
                current_seq = ""
                current_id = str(line[1:])
            elif line.startswith("init_contig"):
                previous_contig = None
                pass
            else:
                (init_contig,
                 id_frag,
                 orientation,
                 pos_start,
                 pos_end) = str(line[:-1]).split("\t")

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

                if junction and previous_contig not in {init_contig,
                                                        None}:
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
    my_records = sorted(SeqIO.parse(init_fasta, "fasta"),
                        reverse=True,
                        key=len)

    that_which_was_removed = {}
    fasta_dict = {}

    def consecutiveness(key_base_tuple):
        key, base = key_base_tuple
        return base - key

    for record in my_records:

        fasta_dict[record.id] = record.seq

        remaining_regions_ordered = range(len(record))
        remaining_regions = set(remaining_regions_ordered)
        regions = [my_bin
                   for scaffold in my_scaffolds.values()
                   for my_bin in scaffold if my_bin[0] == record.id]
        for region in regions:
            start, end = region[2], region[3]
            remaining_regions -= set(remaining_regions_ordered[start:end + 1])

        sorted_regions = sorted(remaining_regions)

        for k, g in itertools.groupby(enumerate(sorted_regions),
                                      consecutiveness):

            swath = list(itertools.imap(operator.itemgetter(1), g))
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
                for name, chunks in that_which_was_removed.iteritems():
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
    for name, scaffold in scaffolds.iteritems():
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
                        print("A bin was reintegrated after position {}"
                              " in scaffold {}".format(my_bin[3], init_name))
                        bin_to_add = [init_name, -1,
                                      lost_start - 1,
                                      lost_end + 1,
                                      ori]
                        scaffold_to_modify.insert(i + 1 - (ori < 0),
                                                  bin_to_add)
                        remaining_dna_positions.pop(init_name)
                        i += 1

                    elif start in (lost_end, lost_end - 1, lost_end + 1):
                        print("A bin was reintegrated before position {}"
                              " in scaffold {}".format(my_bin[3], init_name))
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
    for remaining_name, remaining_bins in remaining_dna_positions.iteritems():
        for my_bin in remaining_bins:
            try:
                remaining_bin = [remaining_name,
                                 -1,
                                 my_bin[2],
                                 my_bin[3],
                                 1]
                new_scaffolds[remaining_name] = [remaining_bin]
            except ValueError:
                continue

    return new_scaffolds


def is_block(bin_list):

    id_set = set((my_bin[1] for my_bin in bin_list))
    start_id, end_id = min(id_set), max(id_set)
    return id_set == set(range(start_id, end_id + 1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process 3C bin '
                                                 'formalized scaffolds.')

    parser.add_argument('-m', '--mode', help='Process mode', required=True)

    parser.add_argument('-i', '--input',
                        type=str,
                        help='Input info_frags.txt to process', required=True)

    parser.add_argument('-f', '--fasta',
                        type=str,
                        help='Reference FASTA file to generate '
                             'new genome with info_frags.txt')

    parser.add_argument('-o', '--output', help='Output file to generate')

    parser.add_argument('-c', '--criterion',
                        type=str,
                        help='Block criterion stringency')

    parser.add_argument('-s', '--min-scaffold-size',
                        type=int,
                        help='Minimum scaffold size in bins',
                        default=DEFAULT_MIN_SCAFFOLD_SIZE)

    parser.add_argument('-j', '--junction',
                        type=str,
                        help='Junction sequence',
                        default='')

    args = parser.parse_args()

    info_frags = args.input
    min_size = args.min_scaffold_size
    scaffolds = {name: scaffold
                 for (name,
                      scaffold) in parse_info_frags(info_frags).iteritems()
                 if len(scaffold) > min_size}

    if args.mode == "fasta":
        init_fasta = args.fasta
        output_file = args.output
        junction = args.junction

        if init_fasta is None:
            print("Error! An initial FASTA file must be provided to write "
                  "the bins into sequences.")

        write_fasta(init_fasta=init_fasta,
                    info_frags=info_frags,
                    junction=junction,
                    output=output_file)

    elif "singleton" in args.mode:
        output_file = args.output
        new_scaffolds = remove_spurious_insertions(scaffolds)
        write_info_frags(new_scaffolds, output=output_file)

    elif args.mode == "inversion":
        output_file = args.output or DEFAULT_NEW_INFO_FRAGS_NAME
        criterion = args.criterion or DEFAULT_CRITERION

        new_scaffolds = correct_spurious_inversions(scaffolds=scaffolds,
                                                    criterion=criterion)

        write_info_frags(new_scaffolds, output=output_file)

    elif args.mode == "inversion2":
        output_file = args.output or DEFAULT_NEW_INFO_FRAGS_NAME
        criterion = args.criterion or DEFAULT_CRITERION_2

        new_scaffolds = reorient_consecutive_blocks(scaffolds=scaffolds,
                                                    mode=criterion)

        write_info_frags(new_scaffolds, output=output_file)

    elif "rearrange" in args.mode:
        output_file = args.output or DEFAULT_NEW_INFO_FRAGS_NAME

        new_scaffolds = rearrange_intra_scaffolds(scaffolds=scaffolds)

        write_info_frags(new_scaffolds, output=output_file)

    elif "reincorporation" in args.mode:
        init_fasta = args.fasta
        output_file = args.output or DEFAULT_NEW_INFO_FRAGS_NAME

        if init_fasta is None:
            print("Error! An initial FASTA file must be provided"
                  " for bin reincorporation.")

        removed = find_lost_dna(init_fasta=init_fasta,
                                scaffolds=scaffolds)

        new_scaffolds = integrate_lost_dna(scaffolds=scaffolds,
                                           lost_dna_positions=removed)

        write_info_frags(new_scaffolds, output=output_file)

    elif "polishing" in args.mode:
        init_fasta = args.fasta
        output_file = args.output

        arranged_scaffolds = rearrange_intra_scaffolds(scaffolds=scaffolds)
        reoriented_scaffolds = reorient_consecutive_blocks(arranged_scaffolds)
        removed = find_lost_dna(init_fasta=init_fasta,
                                scaffolds=reoriented_scaffolds)
        new_scaffolds = integrate_lost_dna(scaffolds=reoriented_scaffolds,
                                           lost_positions=removed)

        write_info_frags(new_scaffolds,
                         output_file=DEFAULT_NEW_INFO_FRAGS_NAME)
        write_fasta(init_fasta=init_fasta,
                    info_frags=DEFAULT_NEW_INFO_FRAGS_NAME,
                    output_file=output_file)

    elif args.mode == "plot":
        plot_info_frags(scaffolds)

    else:
        print("Wrong mode. Available modes are: "
              "fasta, singletons, inversions, reincorporations, plot")
