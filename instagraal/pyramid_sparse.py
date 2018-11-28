#!/usr/bin/env python3
import os
import shutil
import h5py
import sys
import numpy as np
from instagraal.fragment import basic_fragment as B_frag
from matplotlib import pyplot as plt
import scipy.sparse as sp
from instagraal import log
from instagraal.log import logger

logger.setLevel(log.CURRENT_LOG_LEVEL)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f, 1):
            pass
    return i


def build_and_filter(base_folder, size_pyramid, factor, thresh_factor=1):
    """ build fragments pyramids for multi scale analysis and
    remove high sparsity fragments
    """

    min_bin_per_contig = 1
    fact_sub_sampling = factor

    all_pyramid_folder = os.path.join(base_folder, "pyramids")
    if not (os.path.exists(all_pyramid_folder)):
        os.mkdir(all_pyramid_folder)
    init_pyramid_folder = os.path.join(
        all_pyramid_folder, "pyramid_" + str(1) + "_no_thresh"
    )
    if not (os.path.exists(init_pyramid_folder)):
        init_size_pyramid = 1
        build(base_folder, init_size_pyramid, factor, min_bin_per_contig)
    init_pyramid_folder_level_0 = os.path.join(init_pyramid_folder, "level_0")
    contig_info = os.path.join(
        init_pyramid_folder_level_0, "0_contig_info.txt"
    )
    fragments_list = os.path.join(
        init_pyramid_folder_level_0, "0_fragments_list.txt"
    )
    init_abs_fragments_contacts = os.path.join(
        init_pyramid_folder_level_0, "0_abs_frag_contacts.txt"
    )
    ###########################################

    init_pyramid_file = os.path.join(init_pyramid_folder, "pyramid.hdf5")

    ###########################################

    pyramid_folder = os.path.join(
        all_pyramid_folder, "pyramid_" + str(size_pyramid) + "_thresh_auto"
    )
    if not (os.path.exists(pyramid_folder)):
        os.mkdir(pyramid_folder)
    level = 0
    pyramid_level_folder = os.path.join(pyramid_folder, "level_" + str(level))
    if not (os.path.exists(pyramid_level_folder)):
        os.mkdir(pyramid_level_folder)

    current_contig_info = os.path.join(
        pyramid_level_folder, str(level) + "_contig_info.txt"
    )
    current_frag_list = os.path.join(
        pyramid_level_folder, str(level) + "_fragments_list.txt"
    )
    current_abs_fragments_contacts = os.path.join(
        pyramid_level_folder, str(level) + "_abs_frag_contacts.txt"
    )
    if not (
        os.path.exists(current_contig_info)
        and os.path.exists(current_frag_list)
        and os.path.exists(current_abs_fragments_contacts)
    ):
        # if not(os.path.exists(current_contig_info) and
        # os.path.exists(current_frag_list)) :
        #
        logger.info("start filtering")
        pyramid_0 = h5py.File(init_pyramid_file)

        remove_problematic_fragments(
            contig_info,
            fragments_list,
            init_abs_fragments_contacts,
            current_contig_info,
            current_frag_list,
            current_abs_fragments_contacts,
            pyramid_0,
            thresh_factor=thresh_factor,
        )
        pyramid_0.close()
        #
    else:
        logger.info("filtering already done...")

    hdf5_pyramid_file = os.path.join(pyramid_folder, "pyramid.hdf5")

    pyramid_handle = h5py.File(hdf5_pyramid_file)

    pyramid_level_folder = os.path.join(pyramid_folder, "level_" + str(level))
    level_pyramid = str(level) + "_"
    sub_2_super_frag_index_file = os.path.join(
        pyramid_level_folder, level_pyramid + "sub_2_super_index_frag.txt"
    )
    for level in range(0, size_pyramid):
        pyramid_level_folder = os.path.join(
            pyramid_folder, "level_" + str(level)
        )
        if not (os.path.exists(pyramid_level_folder)):
            os.mkdir(pyramid_level_folder)
        level_pyramid = str(level) + "_"
        new_contig_list_file = os.path.join(
            pyramid_level_folder, level_pyramid + "contig_info.txt"
        )
        new_fragments_list_file = os.path.join(
            pyramid_level_folder, level_pyramid + "fragments_list.txt"
        )
        new_abs_fragments_contacts_file = os.path.join(
            pyramid_level_folder, level_pyramid + "abs_frag_contacts.txt"
        )

        if level > 0:
            if (
                os.path.exists(new_contig_list_file)
                and os.path.exists(new_fragments_list_file)
                and os.path.exists(new_abs_fragments_contacts_file)
                and os.path.exists((sub_2_super_frag_index_file))
            ):
                # if os.path.exists(new_contig_list_file) and
                # os.path.exists(new_fragments_list_file) and
                # os.path.exists((sub_2_super_frag_index_file)):
                logger.info("level already built")
                nfrags = file_len(new_fragments_list_file) - 1
            else:  # this should never append !!!
                logger.info("writing new_files..")
                nfrags = subsample_data_set(
                    current_contig_info,
                    current_frag_list,
                    fact_sub_sampling,
                    current_abs_fragments_contacts,
                    new_abs_fragments_contacts_file,
                    min_bin_per_contig,
                    new_contig_list_file,
                    new_fragments_list_file,
                    sub_2_super_frag_index_file,
                )
        else:
            if (
                os.path.exists(new_contig_list_file)
                and os.path.exists(new_fragments_list_file)
                and os.path.exists(new_abs_fragments_contacts_file)
            ):
                # if os.path.exists(new_contig_list_file) and
                # os.path.exists(new_fragments_list_file) :
                logger.info("level already built...")
                nfrags = file_len(new_fragments_list_file) - 1

        try:
            status = pyramid_handle.attrs[str(level)] == "done"
        except KeyError:
            pyramid_handle.attrs[str(level)] = "pending"
            status = False
        if not (status):
            logger.info("Start filling the pyramid")
            # level_to_fill = pyramid_handle.create_dataset(str(level),
            # (nfrags,nfrags), 'i')
            # fill_pyramid_level(level_to_fill,new_abs_fragments_contacts_file,
            # size_chunk,nfrags)
            fill_sparse_pyramid_level(
                pyramid_handle, level, new_abs_fragments_contacts_file, nfrags
            )
            pyramid_handle.attrs[str(level)] = "done"
            ################################################
        current_frag_list = new_fragments_list_file
        current_contig_info = new_contig_list_file
        current_abs_fragments_contacts = new_abs_fragments_contacts_file
        sub_2_super_frag_index_file = os.path.join(
            pyramid_level_folder, level_pyramid + "sub_2_super_index_frag.txt"
        )
    logger.info("pyramid built.")
    ###############################################
    obj_pyramid = pyramid(pyramid_folder, size_pyramid)
    pyramid_handle.close()
    return obj_pyramid


def build(base_folder, size_pyramid, factor, min_bin_per_contig):
    """ build fragments pyramids for multi scale analysis """
    fact_sub_sampling = factor
    contig_info = os.path.join(base_folder, "info_contigs.txt")
    fragments_list = os.path.join(base_folder, "fragments_list.txt")
    init_abs_fragments_contacts = os.path.join(
        base_folder, "abs_fragments_contacts_weighted.txt"
    )
    all_pyramid_folder = os.path.join(base_folder, "pyramids")
    pyramid_folder = os.path.join(
        all_pyramid_folder, "pyramid_" + str(size_pyramid) + "_no_thresh"
    )

    if not (os.path.exists(all_pyramid_folder)):
        os.mkdir(all_pyramid_folder)

    if not (os.path.exists(pyramid_folder)):
        os.mkdir(pyramid_folder)

    hdf5_pyramid_file = os.path.join(pyramid_folder, "pyramid.hdf5")
    pyramid_handle = h5py.File(hdf5_pyramid_file)
    level = 0
    pyramid_level_folder = os.path.join(pyramid_folder, "level_" + str(level))
    if not (os.path.exists(pyramid_level_folder)):
        os.mkdir(pyramid_level_folder)

    current_contig_info = os.path.join(
        pyramid_level_folder, str(level) + "_contig_info.txt"
    )
    current_frag_list = os.path.join(
        pyramid_level_folder, str(level) + "_fragments_list.txt"
    )
    current_abs_fragments_contacts = os.path.join(
        pyramid_level_folder, str(level) + "_abs_frag_contacts.txt"
    )
    for level in range(0, size_pyramid):
        pyramid_level_folder = os.path.join(
            pyramid_folder, "level_" + str(level)
        )
        if not (os.path.exists(pyramid_level_folder)):
            os.mkdir(pyramid_level_folder)
        level_pyramid = str(level) + "_"
        if level == 0:
            shutil.copyfile(contig_info, current_contig_info)
            shutil.copyfile(
                init_abs_fragments_contacts, current_abs_fragments_contacts
            )
            nfrags = init_frag_list(fragments_list, current_frag_list)
            new_abs_fragments_contacts_file = current_abs_fragments_contacts
            new_contig_list_file = current_contig_info
            new_fragments_list_file = current_frag_list
            sub_2_super_frag_index_file = os.path.join(
                pyramid_level_folder,
                level_pyramid + "sub_2_super_index_frag.txt",
            )

        else:

            new_contig_list_file = os.path.join(
                pyramid_level_folder, level_pyramid + "contig_info.txt"
            )
            new_fragments_list_file = os.path.join(
                pyramid_level_folder, level_pyramid + "fragments_list.txt"
            )
            new_abs_fragments_contacts_file = os.path.join(
                pyramid_level_folder, level_pyramid + "abs_frag_contacts.txt"
            )
            if (
                os.path.exists(new_contig_list_file)
                and os.path.exists(new_fragments_list_file)
                and os.path.exists(new_abs_fragments_contacts_file)
                and os.path.exists(sub_2_super_frag_index_file)
            ):
                logger.info("level already built...")
                nfrags = file_len(new_fragments_list_file) - 1
            else:
                logger.info("writing new_files..")
                nfrags = subsample_data_set(
                    current_contig_info,
                    current_frag_list,
                    fact_sub_sampling,
                    current_abs_fragments_contacts,
                    new_abs_fragments_contacts_file,
                    min_bin_per_contig,
                    new_contig_list_file,
                    new_fragments_list_file,
                    sub_2_super_frag_index_file,
                )
        ################################################

        try:
            status = pyramid_handle.attrs[str(level)] == "done"
        except KeyError:
            pyramid_handle.attrs[str(level)] = "pending"
            status = False
        if not (status):
            logger.info("Start filling the pyramid")
            fill_sparse_pyramid_level(
                pyramid_handle, level, new_abs_fragments_contacts_file, nfrags
            )
            pyramid_handle.attrs[str(level)] = "done"
        ################################################
        current_frag_list = new_fragments_list_file
        current_contig_info = new_contig_list_file
        current_abs_fragments_contacts = new_abs_fragments_contacts_file
        sub_2_super_frag_index_file = os.path.join(
            pyramid_level_folder, level_pyramid + "sub_2_super_index_frag.txt"
        )
    logger.info("pyramid built.")
    pyramid_handle.close()
    ###############################################
    # obj_pyramid =
    # pyramid(pyramid_folder,size_pyramid,use_gpu,default_level,use_gpu)
    #
    # return obj_pyramid


def abs_contact_2_coo_file(abs_contact_file, coo_file):

    sparse_dict = dict()
    h = open(abs_contact_file, "r")
    all_lines = h.readlines()
    n_lines = len(all_lines)
    # index start at
    for i in range(1, n_lines):

        line = all_lines[i]
        dat = line.split()
        mates = [int(dat[0]), int(dat[1])]
        mates.sort()
        f1 = mates[0] - 1
        f2 = mates[1] - 1
        if f1 in sparse_dict:
            if f2 in sparse_dict[f1]:
                sparse_dict[f1][f2] += 1
            else:
                sparse_dict[f1][f2] = 1
        else:
            sparse_dict[f1] = dict()
            sparse_dict[f1][f2] = 1

    keys = list(sparse_dict.keys())
    keys.sort()
    h.close()
    h_coo = open(coo_file, "w")
    h_coo.write("%s\t%s\t%s\n" % ("id_frag_a", "id_frag_b", "n_contact"))
    for fa in keys:
        d_fb = sparse_dict[fa]
        keys_b = list(d_fb.keys())
        keys_b.sort()
        for fb in keys_b:
            nc = d_fb[fb]
            h_coo.write("%s\t%s\t%s\n" % (str(fa), str(fb), str(nc)))
    h_coo.close()
    h.close()


def fill_sparse_pyramid_level(pyramid_handle, level, contact_file, nfrags):

    sparse_dict = dict()
    h = open(contact_file, "r")
    all_lines = h.readlines()
    n_lines = len(all_lines)
    # index start at
    for i in range(1, n_lines):

        line = all_lines[i]
        dat = line.split()
        mates = [int(dat[0]), int(dat[1])]
        nc = int(dat[2])
        mates.sort()
        f1 = mates[0]
        f2 = mates[1]

        if f1 in sparse_dict:
            if f2 in sparse_dict[f1]:
                sparse_dict[f1][f2] += nc
            else:
                sparse_dict[f1][f2] = nc
        else:
            sparse_dict[f1] = dict()
            sparse_dict[f1][f2] = nc

    keys = list(sparse_dict.keys())
    keys.sort()

    out_r = []
    out_c = []
    out_d = []

    for r in keys:
        data = sparse_dict[r]
        for c in list(data.keys()):
            out_r.append(r)
            out_c.append(c)
            out_d.append(data[c])

    n_on_pxls = len(out_d)
    level_hdf5 = pyramid_handle.create_group(str(level))
    data_2_sparse = level_hdf5.create_dataset("data", (3, n_on_pxls), "i")
    data_nfrags = level_hdf5.create_dataset("nfrags", (1, 1), "i")
    np_csr = np.zeros((3, n_on_pxls), dtype=np.int32)
    np_csr[0, :] = out_r
    np_csr[1, :] = out_c
    np_csr[2, :] = out_d
    data_2_sparse[0, :] = out_r
    data_2_sparse[1, :] = out_c
    data_2_sparse[2, :] = out_d
    data_nfrags[:] = nfrags


def init_frag_list(fragment_list, new_frag_list):
    """ adapt the original frag list to fit the build function requirements """
    handle_frag_list = open(fragment_list, "r")
    handle_new_frag_list = open(new_frag_list, "w")
    handle_new_frag_list.write(
        "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
        % (
            "id",
            "chrom",
            "start_pos",
            "end_pos",
            "size",
            "gc_content",
            "accu_frag",
            "frag_start",
            "frag_end",
        )
    )
    handle_frag_list.readline()
    i = 0
    while 1:
        line_frag = handle_frag_list.readline()
        if not line_frag:
            handle_frag_list.close()
            handle_new_frag_list.close()
            break
        i += 1
        data = line_frag.split("\t")
        id_init = data[0]
        contig_name = data[1]
        start_pos = data[2]
        end_pos = data[3]
        length_kb = data[4]
        gc_content = str(float(data[5]))
        accu_frag = str(1)
        frag_start = id_init
        frag_end = id_init
        handle_new_frag_list.write(
            "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
            % (
                id_init,
                contig_name,
                start_pos,
                end_pos,
                length_kb,
                gc_content,
                accu_frag,
                frag_start,
                frag_end,
            )
        )

    return i


def subsample_data_set(
    contig_info,
    fragments_list,
    fact_sub_sample,
    abs_fragments_contacts,
    new_abs_fragments_contacts_file,
    min_bin_per_contig,
    new_contig_list_file,
    new_fragments_list_file,
    old_2_new_file,
):

    logger.debug("fact sub sampling = {}".format(fact_sub_sample))
    logger.debug("minimum bin numer per contig = {}".format(min_bin_per_contig))
    if fact_sub_sample <= 1:
        logger.info("subsampling : nothing to do")
        shutil.copy(fragments_list, new_fragments_list_file)
        shutil.copy(contig_info, new_contig_list_file)
        shutil.copy(abs_fragments_contacts, new_abs_fragments_contacts_file)
        nfrags = file_len(fragments_list) - 1
        handle_old_2_new = open(old_2_new_file, "w")
        handle_old_2_new.write("%s\t%s\n" % ("current_id", "super_id"))
        for ind in range(0, nfrags):
            curr_id = str(ind + 1)
            super_id = curr_id
            handle_old_2_new.write("%s\t%s\n" % (curr_id, super_id))
        handle_old_2_new.close()
    else:
        logger.info("subsampling : start")
        old_2_new_frags = dict()
        spec_new_frags = dict()
        handle_new_contigs_list = open(new_contig_list_file, "w")
        handle_new_contigs_list.write(
            "%s\t%s\t%s\t%s\n"
            % ("contig", "length_kb", "n_frags", "cumul_length")
        )

        new_abs_id_frag = 0
        id_frag_abs = 0

        # reading contig info !!!! #######################################
        handle_contig_info = open(contig_info, "r")
        handle_contig_info.readline()
        sum_length_contigs = 0
        while 1:
            line_contig = handle_contig_info.readline()

            if not line_contig:
                handle_contig_info.close()
                handle_new_contigs_list.close()
                break

            data = line_contig.split("\t")
            init_contig = data[0]
            # id_frag_start = 1
            id_frag_end = int(data[2])
            length_kb = data[1]
            # orientation = "w"
            condition_sub_sample = (
                id_frag_end / np.float32(fact_sub_sample)
            ) >= min_bin_per_contig and not (fact_sub_sample == 1)
            accu_frag = 0
            new_rel_id_frag = 0
            id_frag_rel = 0
            sum_length_contigs += id_frag_end
            if condition_sub_sample:
                for arbind in range(0, id_frag_end):
                    #            for id_frag_rel in range(1,id_frag_end+1 ):
                    id_frag_rel += 1
                    id_frag_abs += 1
                    if id_frag_rel % fact_sub_sample == 1:
                        accu_frag = 0
                        new_abs_id_frag += 1
                        new_rel_id_frag += 1
                        spec_new_frags[new_abs_id_frag] = dict()
                        spec_new_frags[new_abs_id_frag][
                            "frag_start"
                        ] = id_frag_abs
                        spec_new_frags[new_abs_id_frag][
                            "frag_end"
                        ] = id_frag_abs

                    accu_frag += 1
                    old_2_new_frags[id_frag_abs] = new_abs_id_frag
                    spec_new_frags[new_abs_id_frag]["accu_frag"] = accu_frag
                    spec_new_frags[new_abs_id_frag]["id_rel"] = new_rel_id_frag
                    spec_new_frags[new_abs_id_frag][
                        "init_contig"
                    ] = init_contig
                    spec_new_frags[new_abs_id_frag]["gc_content"] = []
                    spec_new_frags[new_abs_id_frag]["size"] = []
                    spec_new_frags[new_abs_id_frag]["frag_end"] = id_frag_abs

            else:
                for arbind in range(0, id_frag_end):

                    id_frag_abs += 1
                    new_abs_id_frag += 1
                    new_rel_id_frag += 1
                    id_frag_rel += 1
                    old_2_new_frags[id_frag_abs] = new_abs_id_frag
                    spec_new_frags[new_abs_id_frag] = {
                        "frag_start": id_frag_abs,
                        "frag_end": id_frag_abs,
                        "accu_frag": 1,
                        "init_contig": init_contig,
                        "gc_content": [],
                        "size": [],
                        "id_rel": new_rel_id_frag,
                    }

            handle_new_contigs_list.write(
                "%s\t%s\t%s\t%s\n"
                % (
                    init_contig,
                    length_kb,
                    new_rel_id_frag,
                    new_abs_id_frag - new_rel_id_frag,
                )
            )
            # write new fragments list
        logger.info("size matrix before sub sampling = {}".format(id_frag_abs))
        logger.info(
            "size matrix after sub sampling = {}".format(new_abs_id_frag)
        )
        logger.info("sum length contigs = {}".format(sum_length_contigs))

        # reading fragments list !!!! #######################################
        handle_fragments_list = open(fragments_list, "r")
        handle_fragments_list.readline()
        id_abs = 0
        while 1:
            line_fragments = handle_fragments_list.readline()
            if not line_fragments:
                handle_fragments_list.close()
                break
            id_abs += 1
            #        print id_abs
            data = line_fragments.split("\t")
            id_init = int(data[0])
            # contig_name = data[1]
            start_pos = int(data[2])
            end_pos = int(data[3])
            length_kb = int(data[4])
            gc_content = float(data[5])
            # np_id_abs = id_abs
            curr_id = id_init
            init_frag_start = int(data[7])
            init_frag_end = int(data[8])
            id_new = old_2_new_frags[id_abs]
            spec_new_frags[id_new]["gc_content"].append(gc_content)
            #        spec_new_frags[id_new]['size'].append(length_kb)

            if id_abs == spec_new_frags[id_new]["frag_start"]:
                spec_new_frags[id_new]["start_pos"] = start_pos
                spec_new_frags[id_new][
                    "init_frag_start"
                ] = init_frag_start  # coord level 0
            if id_abs == spec_new_frags[id_new]["frag_end"]:
                spec_new_frags[id_new]["end_pos"] = end_pos
                spec_new_frags[id_new]["size"] = (
                    end_pos - spec_new_frags[id_new]["start_pos"]
                )
                spec_new_frags[id_new][
                    "init_frag_end"
                ] = init_frag_end  # coord level 0

        keys_new_frags = list(spec_new_frags.keys())
        keys_new_frags.sort()
        handle_new_fragments_list = open(new_fragments_list_file, "w")
        handle_new_fragments_list.write(
            "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
            % (
                "id",
                "chrom",
                "start_pos",
                "end_pos",
                "size",
                "gc_content",
                "accu_frag",
                "init_frag_start",
                "init_frag_end",
                "sub_frag_start",
                "sub_frag_end",
            )
        )
        # print "id problem",spec_new_frags[1]
        nfrags = len(keys_new_frags)
        logger.info("nfrags = {}".format(nfrags))
        for new_frag in keys_new_frags:
            id = str(spec_new_frags[new_frag]["id_rel"])

            gc_content = np.array(
                spec_new_frags[new_frag]["gc_content"]
            ).mean()
            size = spec_new_frags[new_frag]["size"]

            start_pos = spec_new_frags[new_frag]["start_pos"]
            end_pos = spec_new_frags[new_frag]["end_pos"]
            chrom = spec_new_frags[new_frag]["init_contig"]

            init_frag_start = spec_new_frags[new_frag]["init_frag_start"]
            init_frag_end = spec_new_frags[new_frag]["init_frag_end"]
            sub_frag_start = spec_new_frags[new_frag]["frag_start"]
            sub_frag_end = spec_new_frags[new_frag]["frag_end"]
            accu_frag = str(int(init_frag_end) - int(init_frag_start) + 1)
            ##########################
            handle_new_fragments_list.write(
                "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                % (
                    id,
                    chrom,
                    start_pos,
                    end_pos,
                    size,
                    gc_content,
                    accu_frag,
                    init_frag_start,
                    init_frag_end,
                    sub_frag_start,
                    sub_frag_end,
                )
            )

        handle_new_fragments_list.close()
        logger.info("new fragments list written...")

        # be carefull : le dictionnaire est base sur
        if not (abs_fragments_contacts == "SIMU"):
            logger.info("update sparse contacts file...")  # index 0 based
            handle_new_abs_fragments_contacts = open(
                new_abs_fragments_contacts_file, "w"
            )
            handle_abs_fragments_contacts = open(abs_fragments_contacts, "r")
            handle_new_abs_fragments_contacts.write(
                "%s\t%s\t%s\n" % ("id_frag_a", "id_frag_b", "n_contact")
            )
            handle_abs_fragments_contacts.readline()
            all_lines = handle_abs_fragments_contacts.readlines()
            sparse_dict = dict()
            for id_line in range(1, len(all_lines)):
                line = all_lines[id_line]
                dat = line.split("\t")
                fa = int(dat[0])
                fb = int(dat[1])
                nc = int(dat[2])
                new_abs_id_frag_a = old_2_new_frags[
                    fa + 1
                ]  # old_2_new_frags 1-based index
                new_abs_id_frag_b = old_2_new_frags[
                    fb + 1
                ]  # old_2_new_frags 1-based index
                mates = [new_abs_id_frag_a - 1, new_abs_id_frag_b - 1]
                mates.sort()
                f1 = mates[0]
                f2 = mates[1]
                if f1 in sparse_dict:
                    if f2 in sparse_dict[f1]:
                        sparse_dict[f1][f2] += nc
                    else:
                        sparse_dict[f1][f2] = nc
                else:
                    sparse_dict[f1] = dict()
                    sparse_dict[f1][f2] = nc
            keys = list(sparse_dict.keys())
            keys.sort()
            for fa in keys:
                d_fb = sparse_dict[fa]
                keys_b = list(d_fb.keys())
                keys_b.sort()
                for fb in keys_b:
                    nc = d_fb[fb]
                    handle_new_abs_fragments_contacts.write(
                        "%s\t%s\t%s\n" % (str(fa), str(fb), str(nc))
                    )
            handle_new_abs_fragments_contacts.close()

        logger.info("subsampling: done.")
        handle_old_2_new = open(old_2_new_file, "w")
        handle_old_2_new.write("%s\t%s\n" % ("current_id", "super_id"))
        for ind in list(old_2_new_frags.keys()):
            curr_id = str(ind)
            super_id = str(old_2_new_frags[ind])
            handle_old_2_new.write("%s\t%s\n" % (curr_id, super_id))

        handle_old_2_new.close()

    return nfrags


# PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM
# PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM #####


def remove_problematic_fragments(
    contig_info,
    fragments_list,
    abs_fragments_contacts,
    new_contig_list_file,
    new_fragments_list_file,
    new_abs_fragments_contacts_file,
    pyramid,
    thresh_factor=1,
):

    import numpy as np

    # full_resolution = pyramid["0"]

    level = pyramid["0"]
    np_2_scipy_sparse = level["data"]
    nfrags = int(level["nfrags"][0])
    logger.info("nfrags = {}".format(nfrags))
    sparse_mat_csr = sp.csr_matrix(
        (np_2_scipy_sparse[2, :], np_2_scipy_sparse[0:2, :]),
        shape=(nfrags, nfrags),
    )

    full_mat = sparse_mat_csr + sparse_mat_csr.transpose()
    collect_sparsity = np.float32(np.diff(full_mat.indptr)) / np.float32(
        nfrags
    )
    mean_spars = collect_sparsity.mean()
    std_spars = collect_sparsity.std()
    max_spars = collect_sparsity.max()
    logger.info("n init frags = {}".format(nfrags))
    logger.info("mean sparsity = {}".format(mean_spars))
    logger.info("median sparsity = {}".format(np.median(collect_sparsity)))
    logger.info("std sparsity = {}".format(std_spars))
    logger.info("max_sparsity = {}".format(max_spars))
    plt.figure()
    plt.scatter(list(range(len(collect_sparsity))), collect_sparsity)
    plt.savefig("sparsity_plot.pdf")
    plt.close()
    plt.figure()
    plt.hist(collect_sparsity, 100)
    plt.savefig("sparsity_hist.pdf")

    # SENSITIVE PARAMETER ########

    # print(collect_sparsity)
    # thresh = np.percentile(collect_sparsity,25)
    # thresh = max_spars + std_spars
    thresh_max = mean_spars + 50 * std_spars
    # thresh_max = np.percentile(collect_sparsity,np.float64(99.9))
    # thresh = mean_spars - 1.01 * std_spars # para g1

    thresh = mean_spars - thresh_factor * std_spars
    # thresh = 0.01 *mean_spars # for human data chr14 4 and de novo chr14
    # thresh = 0.0015 # for human data chr14 4 and de novo chr14
    # thresh = 0.001 *mean_spars # dde novo s1
    # thresh = max(mean_spars - 0.9 * std_spars,0) # para g1
    # SENSITIVE PARAMETER ########
    list_fragments_problem = list(np.nonzero(collect_sparsity == 0)[0])
    logger.info("thresh sparsity = {}".format(thresh))
    # plt.show()
    list_fragments_problem = list(np.nonzero(collect_sparsity <= thresh)[0])
    list_fragments_problem_too_large = list(
        np.nonzero(collect_sparsity > thresh_max)[0]
    )
    list_fragments_problem.extend(list_fragments_problem_too_large)
    # print(np.array(list_fragments_problem))

    logger.info("cleaning : start")
    import numpy as np

    handle_new_fragments_list = open(new_fragments_list_file, "w")
    handle_new_fragments_list.write(
        "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
        % (
            "id",
            "chrom",
            "start_pos",
            "end_pos",
            "size",
            "gc_content",
            "accu_frag",
            "frag_start",
            "frag_end",
        )
    )

    # build np_id_2_frag dictionary
    np_id_2_frag = get_frag_info_from_fil(fragments_list)
    # remove fragments which are too small #
    list_short_frags = []
    thresh_size = 50
    for np_index in np_id_2_frag:
        tmp_frag = np_id_2_frag[np_index]
        if tmp_frag["length"] < thresh_size:
            list_short_frags.append(np_index)
    list_fragments_problem = list(list_fragments_problem)
    list_fragments_problem.extend(list_short_frags)
    list_fragments_problem = np.unique(list_fragments_problem)
    logger.info(
        "number of fragments to remove = {}".format(
            len(list_fragments_problem)
        )
    )

    # build Contigs.init_contigs dictionary ( need to know the number of frags
    # per contig)
    init_contigs, list_init_contig = get_contig_info_from_file(contig_info)

    prob_frag = dict()
    for np_index in list_fragments_problem:
        tmp_frag = np_id_2_frag[np_index]
        id = tmp_frag["index"] + "-" + tmp_frag["init_contig"]
        prob_frag[id] = {
            "init_contig": tmp_frag["init_contig"],
            "index": tmp_frag["index"],
        }

    #    print prob_frag
    contig_info_dict = dict()

    # list_init_contig = init_contigs.keys()
    # list_init_contig.sort()

    for chrom in list_init_contig:
        contig_info_dict[chrom] = {
            "n_frags": init_contigs[chrom]["n_frags"],
            "n_new_frags": 0,
            "length_kb": 0,
        }

    new_id_frag_rel = 0
    new_id_frag_abs = 1
    init_id_frag_abs = 0
    old_2_new_frags = dict()
    handle_fragments_list = open(fragments_list, "r")
    handle_fragments_list.readline()
    spec_new_frags = dict()
    tmp_cumul = {
        "start_pos": 0,
        "end_pos": 0,
        "chrom": 0,
        "size": 0,
        "accu_frag": 0,
        "gc_content": [],
        "lock": False,
        "init_id_frags": [],
        "list_chrom": [],
    }
    ######################################
    step = 0
    while 1:
        line_fragment = handle_fragments_list.readline()
        if not line_fragment:
            if tmp_cumul["lock"]:
                for ele in tmp_cumul["init_id_frags"]:
                    old_2_new_frags[ele] = "destroyed"
                new_id_frag_abs -= 1
            handle_fragments_list.close()
            handle_new_fragments_list.close()
            break
        step += 1
        init_id_frag_abs += 1

        data = line_fragment.split("\t")
        id = int(data[0])
        if id == 1:
            new_id_frag_rel = 1
            if not tmp_cumul["lock"]:
                new_id_frag_abs += 0
            else:
                for ele in tmp_cumul["init_id_frags"]:
                    old_2_new_frags[ele] = "destroyed"
            tmp_cumul["gc_content"] = []
            tmp_cumul["start_pos"] = 0
            tmp_cumul["init_id_frags"] = []
            tmp_cumul["list_chrom"] = []
            tmp_cumul["frag_start"] = []
            tmp_cumul["frag_end"] = []
            tmp_cumul["size"] = 0  # debug
        chrom = data[1]
        start_pos = data[2]
        end_pos = data[3]
        size = int(data[4])
        gc_content = float(data[5])
        accu_frag = int(data[6])
        frag_start = int(data[7])
        frag_end = int(data[8])
        name_frag = str(id) + "-" + chrom
        lock = name_frag in prob_frag

        tmp_cumul["chrom"] = chrom
        tmp_cumul["list_chrom"].append(chrom)
        tmp_cumul["end_pos"] = end_pos
        tmp_cumul["size"] += size
        tmp_cumul["accu_frag"] += accu_frag
        tmp_cumul["lock"] = name_frag in prob_frag
        if size <= 1:
            tmp_cumul["lock"] = True
        tmp_cumul["frag_start"].append(frag_start)
        tmp_cumul["frag_end"].append(frag_end)

        tmp_cumul["gc_content"].append(gc_content)
        tmp_cumul["init_id_frags"].append(init_id_frag_abs)

        old_2_new_frags[init_id_frag_abs] = new_id_frag_abs
        if not lock:
            for ele in tmp_cumul["list_chrom"]:
                if not (ele == tmp_cumul["list_chrom"][0]):
                    logger.info(
                        "warning problem hetero fragments!!!!!!!!!!!!!!!"
                    )

            contig_info_dict[chrom]["n_new_frags"] += 1
            contig_info_dict[chrom]["length_kb"] += tmp_cumul["size"]

            #            str_frag_start = str(min(tmp_cumul["frag_start"]))
            #            str_frag_end = str(max(tmp_cumul["frag_end"]))
            str_frag_start = str(new_id_frag_rel)
            str_frag_end = str(new_id_frag_rel)
            spec_new_frags[new_id_frag_abs] = {
                "accu_frag": accu_frag,
                "gc_content": tmp_cumul["gc_content"],
                "chrom": chrom,
            }
            handle_new_fragments_list.write(
                "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                % (
                    str(new_id_frag_rel),
                    chrom,
                    tmp_cumul["start_pos"],
                    tmp_cumul["end_pos"],
                    str(tmp_cumul["size"]),
                    str(np.array(tmp_cumul["gc_content"]).mean()),
                    tmp_cumul["accu_frag"],
                    str_frag_start,
                    str_frag_end,
                )
            )
            tmp_cumul["start_pos"] = end_pos
            tmp_cumul["end_pos"] = start_pos
            tmp_cumul["size"] = 0
            tmp_cumul["accu_frag"] = 0
            tmp_cumul["lock"] = name_frag in prob_frag
            tmp_cumul["chrom"] = chrom
            tmp_cumul["gc_content"] = []
            tmp_cumul["init_id_frags"] = []
            tmp_cumul["list_chrom"] = []
            tmp_cumul["frag_start"] = []
            tmp_cumul["frag_end"] = []
            new_id_frag_rel += 1
            new_id_frag_abs += 1

    logger.debug("max new id = {}".format(new_id_frag_abs))
    handle_new_contigs_list = open(new_contig_list_file, "w")
    handle_new_contigs_list.write(
        "%s\t%s\t%s\t%s\n" % ("contig", "length_kb", "n_frags", "cumul_length")
    )

    handle_contig_info = open(contig_info, "r")
    handle_contig_info.readline()
    cumul_length = 0

    while 1:
        line_contig = handle_contig_info.readline()
        if not line_contig:
            handle_contig_info.close()
            handle_new_contigs_list.close()
            break
        data = line_contig.split("\t")
        contig = data[0]
        # length_kb = data[1]
        length_kb = contig_info_dict[contig]["length_kb"]
        n_frags = contig_info_dict[contig]["n_new_frags"]
        if n_frags > 0:
            handle_new_contigs_list.write(
                "%s\t%s\t%s\t%s\n"
                % (contig, str(length_kb), str(n_frags), str(cumul_length))
            )
            cumul_length += n_frags
        else:
            logger.info(contig + " has been deleted...")
    logger.info("update contacts files...")
    # write new contacts file
    # n_total_contacts = file_len(abs_fragments_contacts)
    handle_new_abs_fragments_contacts = open(
        new_abs_fragments_contacts_file, "w"
    )
    handle_abs_fragments_contacts = open(abs_fragments_contacts, "r")

    handle_new_abs_fragments_contacts.write(
        "%s\t%s\t%s\n" % ("id_frag_a", "id_frag_b", "n_contact")
    )
    all_lines = handle_abs_fragments_contacts.readlines()

    #     for id_line_contacts in xrange(1, len(all_lines_contact)):
    #         line_contacts = all_lines_contact[id_line_contacts]
    #         data = line_contacts.split()
    #         w_size = data[2]
    #         w_gc = data[3]
    #         abs_id_frag_a = int(data[0])
    #         abs_id_frag_b = int(data[1])
    #         new_abs_id_frag_a = old_2_new_frags[abs_id_frag_a]
    #         new_abs_id_frag_b = old_2_new_frags[abs_id_frag_b]
    # #        step += 1
    # #        pt = np.int32(step)/n_total_contacts
    # p.render(pt * 100, 'step %s\nProcessing...\nDescription: updating
    # contacts file.' % step) if not(new_abs_id_frag_a == 'destroyed' or
    # new_abs_id_frag_b == 'destroyed'):
    # w_sub_sample = (spec_new_frags[new_abs_id_frag_a]['accu_frag']*
    # spec_new_frags[new_abs_id_frag_b]['accu_frag'])
    # handle_new_abs_fragments_contacts.write("%s\t%s\t%s\t%s\t%s\n"
    # %(str(new_abs_id_frag_a),str(new_abs_id_frag_b),
    # w_size, w_gc, str(w_sub_sample)))

    sparse_dict = dict()
    for id_line in range(1, len(all_lines)):
        line = all_lines[id_line]
        dat = line.split("\t")
        fa = int(dat[0])
        fb = int(dat[1])
        nc = int(dat[2])
        new_abs_id_frag_a = old_2_new_frags[
            fa + 1
        ]  # old_2_new_frags 1-based index
        new_abs_id_frag_b = old_2_new_frags[
            fb + 1
        ]  # old_2_new_frags 1-based index
        if not (
            new_abs_id_frag_a == "destroyed"
            or new_abs_id_frag_b == "destroyed"
        ):
            mates = [new_abs_id_frag_a - 1, new_abs_id_frag_b - 1]
            mates.sort()
            f1 = mates[0]
            f2 = mates[1]
            if f1 in sparse_dict:
                if f2 in sparse_dict[f1]:
                    sparse_dict[f1][f2] += nc
                else:
                    sparse_dict[f1][f2] = nc
            else:
                sparse_dict[f1] = dict()
                sparse_dict[f1][f2] = nc
    keys = list(sparse_dict.keys())
    keys.sort()
    for fa in keys:
        d_fb = sparse_dict[fa]
        keys_b = list(d_fb.keys())
        keys_b.sort()
        for fb in keys_b:
            nc = d_fb[fb]
            handle_new_abs_fragments_contacts.write(
                "%s\t%s\t%s\n" % (str(fa), str(fb), str(nc))
            )
    handle_new_abs_fragments_contacts.close()

    return thresh


# PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM
# PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM #####
def get_contig_info_from_file(contig_info):

    handle_contig_info = open(contig_info, "r")
    handle_contig_info.readline()
    init_contigs = dict()
    list_contigs = []
    while 1:
        line_contig = handle_contig_info.readline()
        if not line_contig:
            handle_contig_info.close()
            break

        data = line_contig.split("\t")
        chr = data[0]
        list_contigs.append(chr)
        length_kb = int(data[1])
        n_frags = int(data[2])
        cumul_length = int(data[3])
        init_contigs[chr] = dict()
        init_contigs[chr]["n_frags"] = n_frags
        init_contigs[chr]["length_kb"] = length_kb
        init_contigs[chr]["cumul_length"] = cumul_length
    return init_contigs, list_contigs


def get_frag_info_from_fil(fragments_list):
    handle_list_fragments = open(fragments_list, "r")
    handle_list_fragments.readline()
    fragments_info = dict()
    id = 0
    while 1:
        line_contig = handle_list_fragments.readline()
        if not line_contig:
            handle_list_fragments.close()
            break

        data = line_contig.split("\t")
        fragments_info[id] = dict()
        start = int(data[2])
        end = int(data[3])
        fragments_info[id]["init_contig"] = data[1]
        fragments_info[id]["index"] = data[0]
        fragments_info[id]["length"] = end - start
        id += 1

    return fragments_info


def new_remove_problematic_fragments(
    contig_info,
    fragments_list,
    abs_fragments_contacts,
    new_contig_list_file,
    new_fragments_list_file,
    new_abs_fragments_contacts_file,
    pyramid,
):

    import numpy as np

    # full_resolution = pyramid["0"]
    level = pyramid["0"]
    np_2_scipy_sparse = level["data"]
    nfrags = level["nfrags"][0]
    logger.debug("nfrags = {}".format(nfrags))

    sparse_mat_csr = sp.csr_matrix(
        (np_2_scipy_sparse[2, :], np_2_scipy_sparse[0:2, :]),
        shape=(nfrags, nfrags),
    )
    full_mat = sparse_mat_csr + sparse_mat_csr.transpose()
    collect_sparsity = np.float32(np.diff(full_mat.indptr)) / np.float32(
        nfrags
    )
    mean_spars = collect_sparsity.mean()
    std_spars = collect_sparsity.std()
    max_spars = collect_sparsity.max()
    logger.debug("n init frags = {}".format(nfrags))
    logger.debug("mean sparsity = {}".format(mean_spars))
    logger.debug("std sparsity = {}".format(std_spars))
    logger.debug("max_sparsity = {}".format(max_spars))
    thresh = mean_spars - 1.01 * std_spars  # para g1
    thresh = mean_spars * 0.8  # for de novo chr14
    thresh = 0.0009  # for de novo chr14
    logger.debug("thresh sparsity = {}".format(thresh))
    plt.hist(collect_sparsity, 500)
    plt.show()
    list_fragments_problem = np.nonzero(collect_sparsity <= thresh)[0]

    logger.info(
        "number of fragments to remove = {}".format(
            len(list_fragments_problem)
        )
    )
    handle_new_fragments_list = open(new_fragments_list_file, "w")
    handle_new_fragments_list.write(
        "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
        % (
            "id",
            "chrom",
            "start_pos",
            "end_pos",
            "size",
            "gc_content",
            "accu_frag",
            "frag_start",
            "frag_end",
        )
    )
    # build np_id_2_frag dictionary
    np_id_2_frag = get_frag_info_from_fil(fragments_list)

    # build Contigs.init_contigs dictionary ( need to know the number of frags
    # per contig)
    init_contigs, list_init_contig = get_contig_info_from_file(contig_info)

    prob_frag = dict()
    for np_index in list_fragments_problem:
        tmp_frag = np_id_2_frag[np_index]
        id = tmp_frag["index"] + "-" + tmp_frag["init_contig"]
        prob_frag[id] = {
            "init_contig": tmp_frag["init_contig"],
            "index": tmp_frag["index"],
        }

    #    print prob_frag
    contig_info_dict = dict()

    # list_init_contig = init_contigs.keys()
    # list_init_contig.sort()

    for chrom in list_init_contig:
        contig_info_dict[chrom] = {
            "n_frags": init_contigs[chrom]["n_frags"],
            "n_new_frags": 0,
            "length_kb": 0,
        }

    new_id_frag_rel = 0
    new_id_frag_abs = 1
    init_id_frag_abs = 0
    old_2_new_frags = dict()
    handle_fragments_list = open(fragments_list, "r")
    handle_fragments_list.readline()
    spec_new_frags = dict()
    tmp_cumul = {
        "start_pos": 0,
        "end_pos": 0,
        "chrom": 0,
        "size": 0,
        "accu_frag": 0,
        "gc_content": [],
        "lock": False,
        "init_id_frags": [],
        "list_chrom": [],
    }
    step = 0
    while 1:
        line_fragment = handle_fragments_list.readline()
        if not line_fragment:
            if tmp_cumul["lock"]:
                for ele in tmp_cumul["init_id_frags"]:
                    old_2_new_frags[ele] = "destroyed"
                new_id_frag_abs -= 1
            handle_fragments_list.close()
            handle_new_fragments_list.close()
            break
        step += 1

        init_id_frag_abs += 1

        data = line_fragment.split("\t")
        id = int(data[0])
        if id == 1:
            new_id_frag_rel = 1
            if not tmp_cumul["lock"]:
                new_id_frag_abs += 0
            else:
                for ele in tmp_cumul["init_id_frags"]:
                    old_2_new_frags[ele] = "destroyed"
            tmp_cumul["gc_content"] = []
            tmp_cumul["start_pos"] = 0
            tmp_cumul["init_id_frags"] = []
            tmp_cumul["list_chrom"] = []
            tmp_cumul["frag_start"] = []
            tmp_cumul["frag_end"] = []
            tmp_cumul["size"] = 0  # debug
        chrom = data[1]
        start_pos = data[2]
        end_pos = data[3]
        size = int(data[4])
        gc_content = float(data[5])
        accu_frag = int(data[6])
        frag_start = int(data[7])
        frag_end = int(data[8])
        name_frag = str(id) + "-" + chrom
        lock = name_frag in prob_frag

        tmp_cumul["chrom"] = chrom
        tmp_cumul["list_chrom"].append(chrom)
        tmp_cumul["end_pos"] = end_pos
        tmp_cumul["size"] += size
        tmp_cumul["accu_frag"] += accu_frag
        tmp_cumul["lock"] = name_frag in prob_frag
        if size <= 1:
            tmp_cumul["lock"] = True
        tmp_cumul["frag_start"].append(frag_start)
        tmp_cumul["frag_end"].append(frag_end)

        tmp_cumul["gc_content"].append(gc_content)
        tmp_cumul["init_id_frags"].append(init_id_frag_abs)

        old_2_new_frags[init_id_frag_abs] = new_id_frag_abs
        if not lock:
            for ele in tmp_cumul["list_chrom"]:
                if not (ele == tmp_cumul["list_chrom"][0]):
                    logger.info(
                        "warning problem hetero fragments!!!!!!!!!!!!!!!"
                    )

            contig_info_dict[chrom]["n_new_frags"] += 1
            contig_info_dict[chrom]["length_kb"] += tmp_cumul["size"]

            #            str_frag_start = str(min(tmp_cumul["frag_start"]))
            #            str_frag_end = str(max(tmp_cumul["frag_end"]))
            str_frag_start = str(new_id_frag_rel)
            str_frag_end = str(new_id_frag_rel)
            spec_new_frags[new_id_frag_abs] = {
                "accu_frag": accu_frag,
                "gc_content": tmp_cumul["gc_content"],
                "chrom": chrom,
            }
            handle_new_fragments_list.write(
                "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                % (
                    str(new_id_frag_rel),
                    chrom,
                    tmp_cumul["start_pos"],
                    tmp_cumul["end_pos"],
                    str(tmp_cumul["size"]),
                    str(np.array(tmp_cumul["gc_content"]).mean()),
                    tmp_cumul["accu_frag"],
                    str_frag_start,
                    str_frag_end,
                )
            )
            tmp_cumul["start_pos"] = end_pos
            tmp_cumul["end_pos"] = start_pos
            tmp_cumul["size"] = 0
            tmp_cumul["accu_frag"] = 0
            tmp_cumul["lock"] = name_frag in prob_frag
            tmp_cumul["chrom"] = chrom
            tmp_cumul["gc_content"] = []
            tmp_cumul["init_id_frags"] = []
            tmp_cumul["list_chrom"] = []
            tmp_cumul["frag_start"] = []
            tmp_cumul["frag_end"] = []
            new_id_frag_rel += 1
            new_id_frag_abs += 1

    # Update contig info file
    logger.debug("max new id = {}".format(new_id_frag_abs))
    handle_new_contigs_list = open(new_contig_list_file, "w")
    handle_new_contigs_list.write(
        "%s\t%s\t%s\t%s\n" % ("contig", "length_kb", "n_frags", "cumul_length")
    )

    handle_contig_info = open(contig_info, "r")
    handle_contig_info.readline()
    cumul_length = 0

    while 1:
        line_contig = handle_contig_info.readline()
        if not line_contig:
            handle_contig_info.close()
            handle_new_contigs_list.close()
            break
        data = line_contig.split("\t")
        contig = data[0]
        # length_kb = data[1]
        length_kb = contig_info_dict[contig]["length_kb"]
        n_frags = contig_info_dict[contig]["n_new_frags"]
        if n_frags > 0:
            handle_new_contigs_list.write(
                "%s\t%s\t%s\t%s\n"
                % (contig, str(length_kb), str(n_frags), str(cumul_length))
            )
            cumul_length += n_frags
        else:
            logger.info(contig + " has been deleted...")

    logger.info("update contacts files...")
    handle_new_abs_fragments_contacts = open(
        new_abs_fragments_contacts_file, "w"
    )
    handle_abs_fragments_contacts = open(abs_fragments_contacts, "r")
    handle_new_abs_fragments_contacts.write(
        "%s\t%s\t%s\n" % ("id_frag_a", "id_frag_b", "n_contact")
    )
    all_lines = handle_abs_fragments_contacts.readlines()
    sparse_dict = dict()
    for id_line in range(1, len(all_lines)):
        line = all_lines[id_line]
        dat = line.split("\t")
        fa = int(dat[0])
        fb = int(dat[1])
        nc = int(dat[2])
        new_abs_id_frag_a = old_2_new_frags[
            fa + 1
        ]  # old_2_new_frags 1-based index
        new_abs_id_frag_b = old_2_new_frags[
            fb + 1
        ]  # old_2_new_frags 1-based index
        if not (
            new_abs_id_frag_a == "destroyed"
            or new_abs_id_frag_b == "destroyed"
        ):
            mates = [new_abs_id_frag_a - 1, new_abs_id_frag_b - 1]
            mates.sort()
            f1 = mates[0]
            f2 = mates[1]
            if f1 in sparse_dict:
                if f2 in sparse_dict[f1]:
                    sparse_dict[f1][f2] += nc
                else:
                    sparse_dict[f1][f2] = nc
            else:
                sparse_dict[f1] = dict()
                sparse_dict[f1][f2] = nc
    keys = list(sparse_dict.keys())
    keys.sort()
    for fa in keys:
        d_fb = sparse_dict[fa]
        keys_b = list(d_fb.keys())
        keys_b.sort()
        for fb in keys_b:
            nc = d_fb[fb]
            handle_new_abs_fragments_contacts.write(
                "%s\t%s\t%s\n" % (str(fa), str(fb), str(nc))
            )
    handle_new_abs_fragments_contacts.close()


class pyramid:
    def __init__(self, pyramid_folder, n_levels):
        logger.info("init pyramid")
        self.pyramid_folder = pyramid_folder

        self.n_levels = n_levels
        pyramid_file = "pyramid.hdf5"
        self.pyramid_file = os.path.join(pyramid_folder, pyramid_file)
        self.data = h5py.File(self.pyramid_file)

        self.spec_level = dict()
        # self.default_level = default_level
        self.struct_initiated = False
        self.resol_F_s_kb = 3  # size bin in kb
        self.dist_max_kb = 30 * 2 * self.resol_F_s_kb  # length histo in kb
        # self.resol_F_s_kb = 10 # size bin in kb
        # self.dist_max_kb = 10 * 2 * self.resol_F_s_kb # length histo in kb
        for i in range(0, n_levels):
            level_folder = os.path.join(pyramid_folder, "level_" + str(i))
            find_super_index = i < n_levels - 1
            self.spec_level[str(i)] = dict()
            self.spec_level[str(i)]["level_folder"] = level_folder
            self.spec_level[str(i)]["fragments_list_file"] = os.path.join(
                level_folder, str(i) + "_fragments_list.txt"
            )
            self.spec_level[str(i)]["contig_info_file"] = os.path.join(
                level_folder, str(i) + "_contig_info.txt"
            )
            (
                frag_dictionary,
                contig_dictionary,
                list_contigs,
                list_contigs_id,
            ) = self.build_frag_dictionnary(
                self.spec_level[str(i)]["fragments_list_file"], i
            )
            if i == 0:
                self.list_contigs_name = list_contigs
                self.list_contigs_id = list_contigs_id
            self.spec_level[str(i)]["fragments_dict"] = frag_dictionary
            self.spec_level[str(i)]["contigs_dict"] = contig_dictionary

            if find_super_index:
                # print "update super index"
                super_index_file = os.path.join(
                    level_folder, str(i) + "_sub_2_super_index_frag.txt"
                )
                self.update_super_index(
                    self.spec_level[str(i)]["fragments_dict"], super_index_file
                )
                self.update_super_index_in_dict_contig(
                    self.spec_level[str(i)]["fragments_dict"],
                    self.spec_level[str(i)]["contigs_dict"],
                )
            else:
                for contig_id in list(
                    self.spec_level[str(i)]["contigs_dict"].keys()
                ):
                    try:
                        int(contig_id)
                    except ValueError:
                        self.spec_level[str(i)]["contigs_dict"].pop(contig_id)

    def close(self):
        self.data.close()

    def get_level(self, level_id):
        lev = level(self, level_id)
        return lev

    def build_frag_dictionnary(self, fragments_list, level):
        handle_list_fragments = open(fragments_list, "r")
        handle_list_fragments.readline()
        fragments_info = dict()
        id = 1
        contig_dict = dict()
        id_contig = 0
        list_contigs = []
        list_contigs_id = []
        while 1:
            line_contig = handle_list_fragments.readline()
            if not line_contig:
                handle_list_fragments.close()
                break

            data = line_contig.split("\t")
            curr_id = int(data[0])
            tag = data[0] + "-" + data[1]
            start_pos = int(data[2])
            end_pos = int(data[3])
            size = int(data[4])
            gc_content = float(data[5])
            n_accu_frags = int(data[6])
            id_init_frag_start = int(data[7])
            id_init_frag_end = int(data[8])

            if level > 0:
                id_sub_frag_start = int(data[9])
                id_sub_frag_end = int(data[10])
            else:
                id_sub_frag_start = int(curr_id)
                id_sub_frag_end = int(curr_id)
            fragments_info[id] = dict()
            contig_name = data[1]
            fragments_info[id]["init_contig"] = contig_name
            fragments_info[id]["index"] = int(curr_id)
            fragments_info[id]["tag"] = tag
            fragments_info[id]["start_pos(bp)"] = start_pos
            fragments_info[id]["end_pos(bp)"] = end_pos
            fragments_info[id]["size(bp)"] = size
            fragments_info[id]["sub_low_index"] = id_sub_frag_start
            fragments_info[id]["sub_high_index"] = id_sub_frag_end
            fragments_info[id]["super_index"] = curr_id
            fragments_info[id]["n_accu_frags"] = n_accu_frags

            if not (contig_name in contig_dict):
                id_contig += 1
                list_contigs.append(contig_name)
                list_contigs_id.append(id_contig)
                contig_dict[contig_name] = dict()
                contig_dict[contig_name]["frag"] = []
                contig_dict[contig_name]["id_contig"] = id_contig
                contig_dict[id_contig] = []
            f = B_frag.initiate(
                id,
                curr_id,
                contig_name,
                curr_id,
                start_pos,
                end_pos,
                size,
                gc_content,
                id_init_frag_start,
                id_init_frag_end,
                id_sub_frag_start,
                id_sub_frag_end,
                curr_id,
                id_contig,
                n_accu_frags,
            )
            contig_dict[contig_name]["frag"].append(f)
            contig_dict[id_contig].append(f)
            id += 1
        return fragments_info, contig_dict, list_contigs, list_contigs_id

    def update_super_index(self, dict_frag, super_index_file):
        handle_super_index = open(super_index_file, "r")
        handle_super_index.readline()

        while 1:
            line_index = handle_super_index.readline()
            if not line_index:
                handle_super_index.close()
                break

            data = line_index.split("\t")
            dict_frag[int(data[0])]["super_index"] = int(data[1])

    def update_super_index_in_dict_contig(self, dict_frag, dict_contig):
        set_contig = set()
        for id in list(dict_frag.keys()):
            id_frag = id
            frag = dict_frag[id_frag]
            init_contig = dict_frag[id_frag]["init_contig"]
            set_contig.add(init_contig)
            id_contig = dict_contig[init_contig]["id_contig"]
            f = dict_contig[id_contig][frag["index"] - 1]
            f.super_index = frag["super_index"]
        # print "set conrigs!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # print set_contig
        for ele in set_contig:
            dict_contig.pop(ele)

    def zoom_in_frag(self, curr_frag):
        """
        :param curr_frag:
        """
        level = curr_frag[1]
        frag = curr_frag[0]
        output = []
        if level > 0:
            str_level = str(level)
            sub_low = self.spec_level[str_level]["fragments_dict"][frag][
                "sub_low_index"
            ]
            sub_high = self.spec_level[str_level]["fragments_dict"][frag][
                "sub_high_index"
            ]
            new_level = level - 1
            for i in range(sub_low, sub_high + 1):
                output.append((i, new_level))
        else:
            output.append(curr_frag)
        return output

    def zoom_out_frag(self, curr_frag):
        """
        :param curr_frag:
        """
        level = curr_frag[1]
        frag = curr_frag[0]
        output = []
        if level > 0:
            str_level = str(level)
            high_frag = self.spec_level[str_level]["fragments_dict"][frag][
                "super_index"
            ]
            new_level = level + 1
            output = (high_frag, new_level)
        else:
            output = curr_frag
        return output

    def full_zoom_in_frag(self, curr_frag):
        """
        :param curr_frag:
        """
        level = curr_frag[1]
        frag = curr_frag[0]
        output = []
        if level > 0:
            str_level = str(level)
            sub_low = self.spec_level[str_level]["fragments_dict"][frag][
                "sub_low_index"
            ]
            sub_high = self.spec_level[str_level]["fragments_dict"][frag][
                "sub_high_index"
            ]
            new_level = level - 1
            for i in range(sub_low, sub_high + 1):
                output.append((i, new_level))
        else:
            output.append(curr_frag)
        return output

    def zoom_in_pixel(self, curr_pixel):
        """ return the curr_frag at a higher resolution"""
        low_frag = curr_pixel[0]
        high_frag = curr_pixel[1]
        level = curr_pixel[2]
        if level > 0:
            str_level = str(level)
            low_sub_low = self.spec_level[str_level]["fragments_dict"][
                low_frag
            ]["sub_low_index"]
            low_sub_high = self.spec_level[str_level]["fragments_dict"][
                low_frag
            ]["sub_high_index"]
            high_sub_low = self.spec_level[str_level]["fragments_dict"][
                high_frag
            ]["sub_low_index"]
            high_sub_high = self.spec_level[str_level]["fragments_dict"][
                high_frag
            ]["sub_high_index"]
            vect = [low_sub_low, low_sub_high, high_sub_low, high_sub_high]
            new_pix_low = min(vect)
            new_pix_high = max(vect)
            new_level = level - 1
            new_pixel = [new_pix_low, new_pix_high, new_level]
        else:
            new_pixel = curr_pixel
        return new_pixel

    def zoom_out_pixel(self, curr_pixel):
        """ return the curr_frag at a lower resolution"""
        low_frag = curr_pixel[0]
        high_frag = curr_pixel[1]
        level = curr_pixel[2]
        str_level = str(level)
        if level < self.n_level - 1:
            low_super = self.spec_level[str_level]["fragments_dict"][low_frag][
                "super_index"
            ]
            high_super = self.spec_level[str_level]["fragments_dict"][
                high_frag
            ]["sub_index"]

            new_pix_low = min([low_super, high_super])
            new_pix_high = max([low_super, high_super])
            new_level = level + 1
            new_pixel = [new_pix_low, new_pix_high, new_level]
        else:
            new_pixel = curr_pixel
        return new_pixel

    def zoom_in_area(self, area):
        """ zoom in area"""
        x = area[0]
        y = area[1]
        level = x[2]
        logger.debug("x = {}".format(x))
        logger.debug("y = {}".format(y))
        logger.debug("level = {}".format(level))
        if level == y[2] and level > 0:
            new_level = level - 1
            high_x = self.zoom_in_pixel(x)
            high_y = self.zoom_in_pixel(y)
            new_x = [
                min([high_x[0], high_y[0]]),
                min([high_x[1], high_y[1]]),
                new_level,
            ]
            new_y = [
                max([high_x[0], high_y[0]]),
                max([high_x[1], high_y[1]]),
                new_level,
            ]
            new_area = [new_x, new_y]
        else:
            new_area = area

        return new_area

    def load_reference_sequence(self, genome_fasta):

        logger.info("import reference genome")
        f = open(genome_fasta, "r")
        self.dict_sequence_contigs = dict()
        all_lines = f.readlines()
        id_chrom = all_lines[0][1:-1].replace("\n", "").replace("\r", "")
        self.dict_sequence_contigs[id_chrom] = ""
        start = 1
        chrom_list = []
        for i in range(1, len(all_lines)):
            if all_lines[i][0] == ">":
                # print id_chrom
                chrom_list.append(id_chrom)
                self.dict_sequence_contigs[id_chrom] = "".join(
                    all_lines[start:i]
                )
                start = i + 1
                id_chrom = (
                    all_lines[i][1:-1].replace("\n", "").replace("\r", "")
                )
                self.dict_sequence_contigs[id_chrom] = ""

        chrom_list.append(id_chrom)
        self.dict_sequence_contigs[id_chrom] = "".join(all_lines[start:-1])

        chrom_list = list(self.dict_sequence_contigs.keys())
        # chrom_list.sort()
        for chrom in chrom_list:
            self.dict_sequence_contigs[chrom] = self.dict_sequence_contigs[
                chrom
            ].replace("\n", "")
            self.dict_sequence_contigs[chrom] = self.dict_sequence_contigs[
                chrom
            ].replace("\r", "")

        f.close()
        # print chrom_list


class level:
    def __init__(self, pyramid, level):
        self.level = level
        self.T_frag = np.dtype(
            [
                ("pos", np.int32),
                ("id_c", np.int32),
                ("start_bp", np.int32),
                ("len_bp", np.int32),
                ("circ", np.int32),
                ("id", np.int32),
                ("prev", np.int32),
                ("next", np.int32),
                ("l_cont", np.int32),
                ("l_cont_bp", np.int32),
                ("n_accu", np.int32),
            ],
            align=True,
        )
        self.float4 = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("w", np.float32),
            ],
            align=True,
        )
        self.S_o_A_frags = {}
        self.S_o_A_frags["pos"] = []
        self.S_o_A_frags["sub_pos"] = []
        self.S_o_A_frags["id_c"] = []
        self.S_o_A_frags["start_bp"] = []
        self.S_o_A_frags["len_bp"] = []
        self.S_o_A_frags["sub_len"] = []
        self.S_o_A_frags["circ"] = []
        self.S_o_A_frags["id"] = []
        self.S_o_A_frags["prev"] = []
        self.S_o_A_frags["next"] = []
        self.S_o_A_frags["l_cont"] = []
        self.S_o_A_frags["sub_l_cont"] = []
        self.S_o_A_frags["l_cont_bp"] = []
        self.S_o_A_frags["n_accu"] = []
        self.vect_frag_np = []
        self.frags_init_contigs = []
        # self.pos_vect_frags_4_GL = []
        # self.col_vect_frags_4_GL = []
        self.load_data(pyramid)
        self.pyramid = pyramid

    def load_data(self, pyramid):
        """
        :param pyramid: hic pyramid
        """
        import colorsys

        logger.info("loading data from level = {}".format(self.level))
        # self.im_init = np.array(pyramid.data[str(self.level)],
        # dtype=np.int32)
        # self.n_frags = self.im_init.shape[0]

        # start loading sparse matrix #####
        self.n_frags = int(np.copy(pyramid.data[str(self.level)]["nfrags"][0]))
        self.np_2_scipy_sparse = np.copy(pyramid.data[str(self.level)]["data"])
        self.sparse_mat_csr = sp.csr_matrix(
            (self.np_2_scipy_sparse[2, :], self.np_2_scipy_sparse[0:2, :]),
            shape=(self.n_frags, self.n_frags),
        )
        self.sparse_mat_csc = sp.csc_matrix(
            (self.np_2_scipy_sparse[2, :], self.np_2_scipy_sparse[0:2, :]),
            shape=(self.n_frags, self.n_frags),
        )
        # end loading sparse matrix #####

        int_n_frags = np.int32(self.n_frags)
        self.pos_vect_frags_4_GL = np.ndarray(
            (int_n_frags, 4), dtype=np.float32
        )
        self.col_vect_frags_4_GL = np.ndarray(
            (int_n_frags, 4), dtype=np.float32
        )

        self.dict_contigs = dict()
        ContFrags = pyramid.spec_level[str(self.level)]["contigs_dict"]
        if str(self.level - 1) in pyramid.spec_level:
            subContFrags = pyramid.spec_level[str(self.level - 1)][
                "contigs_dict"
            ]
        else:
            subContFrags = ContFrags
        coord_cont = dict()

        n_contigs = len(list(ContFrags.keys()))
        # print "n init contigs = ", n_contigs
        HSV_tuples = [
            (x * 2.5 / n_contigs, 0.5, 0.5) for x in range(n_contigs)
        ]
        RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
        self.distri_frag = []

        # list_contigs = ContFrags.keys()
        # print "Cont frags keys = ", ContFrags.keys()
        list_contigs_id = pyramid.list_contigs_id
        # debugggg
        # list_contigs.sort()

        for i in range(0, self.n_frags):
            self.frags_init_contigs.append("")

        for id_cont in list_contigs_id:
            self.dict_contigs[id_cont] = dict()
            cont_frag = ContFrags[id_cont]
            sub_cont_frag = subContFrags[id_cont]
            coord_cont[id_cont] = []
            name_contig = cont_frag[0].init_contig
            tick_kb = []
            end_frags_kb = []

            id_f_prev = -1
            lim_curr_cont = len(cont_frag)
            sub_lim_curr_cont = len(sub_cont_frag)
            len_curr_contig_bp = 0
            curr_sub_pos = 0
            for id_f in range(0, lim_curr_cont):
                f = cont_frag[id_f]
                lkb = f.length_kb
                len_curr_contig_bp += lkb
            for id_f in range(0, lim_curr_cont):
                f = cont_frag[id_f]
                lkb = f.length_kb
                self.distri_frag.append(lkb)
                start = f.start_pos
                end = f.end_pos
                pos_kb = start + lkb / 2.
                tick_kb.append(pos_kb)
                end_frags_kb.append(end)
                id_f_curr = f.np_id_abs - 1
                n_accu_frags = f.n_accu_frags
                n_sub_frags = f.sub_frag_end - f.sub_frag_start + 1

                self.frags_init_contigs[id_f_curr] = f.init_contig

                if id_f < lim_curr_cont - 1:
                    id_f_next = cont_frag[id_f + 1].np_id_abs - 1
                else:
                    id_f_next = -1
                f_T_frag = (
                    np.int32(f.curr_id - 1),
                    np.int32(f.contig_id),
                    np.int32(start),
                    np.int32(lkb),
                    np.int32(0),
                    np.int32(id_f_curr),
                    np.int32(id_f_prev),
                    np.int32(id_f_next),
                    np.int32(lim_curr_cont),
                    np.int32(len_curr_contig_bp),
                    np.int32(n_accu_frags),
                )
                self.pos_vect_frags_4_GL[id_f_curr, 0] = (
                    np.float32(f.curr_id - 1) / 100.
                )
                self.pos_vect_frags_4_GL[id_f_curr, 1] = (
                    np.float32(f.contig_id) / 100.
                )
                # self.pos_vect_frags_4_GL[id_f_curr, 2] = np.float32(0.0)
                self.pos_vect_frags_4_GL[id_f_curr, 2] = 0.
                self.pos_vect_frags_4_GL[id_f_curr, 3] = np.float32(1.0)
                # GL_pos_frag = (np.float32(f.curr_id - 1),
                #            np.float32(f.contig_id),
                #            np.float32(lim_curr_cont),
                #            np.float32(0.0))
                # self.col_vect_frags_4_GL[id_f_curr,0] = np.float32(0.)
                # self.col_vect_frags_4_GL[id_f_curr,1] = np.float32(1.)
                # self.col_vect_frags_4_GL[id_f_curr,2] = np.float32(0.)
                # self.col_vect_frags_4_GL[id_f_curr,3] = np.float32(1.)
                self.col_vect_frags_4_GL[id_f_curr, 0] = np.float32(
                    RGB_tuples[id_cont - 1][0]
                )
                self.col_vect_frags_4_GL[id_f_curr, 1] = np.float32(
                    RGB_tuples[id_cont - 1][1]
                )
                self.col_vect_frags_4_GL[id_f_curr, 2] = np.float32(
                    RGB_tuples[id_cont - 1][2]
                )
                self.col_vect_frags_4_GL[id_f_curr, 3] = np.float32(1.0)

                # GL_col_frag = (np.float32(RGB_tuples[id_cont - 1][0],
                #                np.float32(RGB_tuples[id_cont - 1][1]),
                #                np.float32(RGB_tuples[id_cont - 1][2]),
                #                np.float32(1.0))
                self.S_o_A_frags["pos"].append(f.curr_id - 1)
                self.S_o_A_frags["sub_pos"].append(curr_sub_pos)  # TO DO !!!!
                self.S_o_A_frags["id_c"].append(f.contig_id)
                self.S_o_A_frags["start_bp"].append(start)
                self.S_o_A_frags["len_bp"].append(lkb)
                self.S_o_A_frags["sub_len"].append(n_sub_frags)
                self.S_o_A_frags["circ"].append(0)
                self.S_o_A_frags["id"].append(id_f_curr)
                self.S_o_A_frags["prev"].append(id_f_prev)
                self.S_o_A_frags["next"].append(id_f_next)
                self.S_o_A_frags["l_cont"].append(lim_curr_cont)
                self.S_o_A_frags["sub_l_cont"].append(sub_lim_curr_cont)
                self.S_o_A_frags["l_cont_bp"].append(len_curr_contig_bp)
                self.S_o_A_frags["n_accu"].append(n_accu_frags)

                id_f_prev = id_f_curr
                self.vect_frag_np.append(f_T_frag)
                curr_sub_pos += n_sub_frags

                # self.pos_vect_frags_4_GL.append(GL_pos_frag)
                # self.col_vect_frags_4_GL.append(GL_col_frag)
            for f in cont_frag:
                coord_cont[id_cont].append(f.np_id_abs - 1)
            self.dict_contigs[id_cont]["intra_coord"] = coord_cont[id_cont]
            self.dict_contigs[id_cont]["frags"] = cont_frag
            self.dict_contigs[id_cont]["name"] = name_contig
            self.dict_contigs[id_cont]["tick_kb"] = np.array(tick_kb)
            self.dict_contigs[id_cont]["end_frags_kb"] = np.array(end_frags_kb)
            #######################

        self.vect_frag_np = np.array(self.vect_frag_np, dtype=self.T_frag)

        # self.pos_vect_frags_4_GL = np.array(self.pos_vect_frags_4_GL,
        # dtype=self.float4) self.col_vect_frags_4_GL =
        # np.array(self.col_vect_frags_4_GL, dtype=self.float4)

        for key in list(self.S_o_A_frags.keys()):
            self.S_o_A_frags[key] = np.array(
                self.S_o_A_frags[key], dtype=np.int32
            )

        total_trans = 0
        n_tot = 0
        n_tot_intra = 0
        for id_cont in range(1, len(self.dict_contigs) + 1):
            sub_sp_mat_tmp = self.sparse_mat_csr[coord_cont[id_cont], :]
            n_full_contact = sub_sp_mat_tmp.sum()
            sub_sp_mat_tmp_csc = sub_sp_mat_tmp.tocsc()
            sub_sp_mat_tmp_intra = sub_sp_mat_tmp_csc[:, coord_cont[id_cont]]
            shape_test = sub_sp_mat_tmp_intra.shape
            test = shape_test[0] == shape_test[1]
            if not test:
                logger.info("MIERDA!!!!")
            n_intra = sub_sp_mat_tmp_intra.sum()

            total_trans += n_full_contact
            total_trans -= n_intra
            # n_tot += (sub_sp_mat_tmp.shape[0] * sub_sp_mat_tmp.shape[1]) -
            # sub_sp_mat_tmp_intra.shape[0]*sub_sp_mat_tmp_intra.shape[1]
            n_tot_intra += shape_test[0] * (shape_test[0] - 1) / 2

        # for id_cont in xrange(1, len(self.dict_contigs)+1):
        #    full = self.im_init[coord_cont[id_cont], :]
        #    intra = self.im_init[np.ix_(coord_cont[id_cont],
        #    coord_cont[id_cont])]
        #    total_trans += full.sum()
        #    total_trans -= intra.sum()
        #    n_tot += (full.shape[0]*full.shape[1]) -
        #    intra.shape[0]*intra.shape[1]
        n_tot = (
            self.sparse_mat_csr.shape[0]
            * (self.sparse_mat_csr.shape[0] - 1)
            / 2
            - n_tot_intra
        )
        self.mean_value_trans = total_trans / np.float32(n_tot)
        if np.isnan(self.mean_value_trans):
            self.mean_value_trans = np.amin(self.sparse_mat_csr.data) / 10.0
        logger.debug(
            "computed mean trans value ... = {}".format(self.mean_value_trans)
        )

        self.distri_frag = np.array(self.distri_frag)

        # self.define_inter_chrom_coord()
        # self.init_data()

        self.n_contigs = len(self.dict_contigs)

    def init_data(self,):
        logger.info("init data {}".format(self.level))
        if np.__version__ == "1.7.1" or np.__version__ == "1.8.0.dev-1a9aa5a":
            tmp = np.empty_like(self.im_init)
            np.copyto(tmp, self.im_init)
        else:
            tmp = np.copy(self.im_init)
        tmp = np.float32(tmp)
        self.im_curr = tmp

    def define_inter_chrom_coord(self):
        """
        """
        self.inter_coord = dict()
        self.all_data = dict()
        logger.info("define inter chrom coord ...")
        for id_cont_X in self.dict_contigs:
            if not (id_cont_X == 17):
                self.inter_coord[id_cont_X] = dict()
                self.all_data[id_cont_X] = dict()
                coord_intra_X = self.dict_contigs[id_cont_X]["intra_coord"]
                self.inter_coord[id_cont_X]["all"] = np.ix_(
                    coord_intra_X, np.arange(0, self.n_frags)
                )
                for id_cont_Y in self.dict_contigs:
                    if not (id_cont_Y == 17):
                        self.all_data[id_cont_X][id_cont_Y] = dict()
                        coord_intra_Y = self.dict_contigs[id_cont_Y][
                            "intra_coord"
                        ]
                        self.inter_coord[id_cont_X][id_cont_Y] = np.ix_(
                            coord_intra_X, coord_intra_Y
                        )
        logger.info("done!")

    def build_seq_per_bin(self, genome_fasta):
        self.pyramid.load_reference_sequence(genome_fasta)
        ContFrags = self.pyramid.spec_level[str(self.level)]["contigs_dict"]
        list_contigs = list(ContFrags.keys())
        list_contigs.sort()

        list_contigs_tmp = list(
            self.pyramid.spec_level[str(self.level)]["contigs_dict"].keys()
        )
        # print "list contigs = ", list_contigs
        list_contigs = []
        for ele in list_contigs_tmp:
            if not isinstance(ele, str):
                list_contigs.append(ele)

        list_contigs.sort()
        self.list_seq = []
        for cont in list_contigs:
            for frag in ContFrags[cont]:
                start = frag.start_pos
                end = frag.end_pos
                # print "init contig = ", frag.init_contig
                init_contig = frag.init_contig
                seq_frag = self.pyramid.dict_sequence_contigs[init_contig][
                    start:end
                ]
                # print "len seq per frag = ", len(seq_frag)
                self.list_seq.append(seq_frag)

    def generate_new_fasta(self, vect_frags, new_fasta, info_frags):
        logger.info("generate new fasta file...")
        id_c_frag = vect_frags.id_c
        pos_frag = vect_frags.pos
        ori_frag = vect_frags.ori
        activ_frag = vect_frags.activ
        handle_new_fasta = open(new_fasta, "w")
        handle_info_frags = open(info_frags, "w")

        list_id_contigs = np.unique(id_c_frag)
        # n_new_contigs = len(list_id_contigs)
        list_seq_new_contigs = dict()
        list_contigs_ok = []
        for id_cont in list_id_contigs:
            list_frags = np.nonzero(id_c_frag == id_cont)[0]
            if np.all(activ_frag[list_frags] == 1):
                # print "id contig = ", id_cont
                list_contigs_ok.append(id_cont)
                header = ">3C-assembly|contig_" + str(id_cont)
                handle_info_frags.write("%s\n" % (header))
                handle_info_frags.write(
                    "%s\t%s\t%s\t%s\t%s\n"
                    % ("init_contig", "id_frag", "orientation", "start", "end")
                )

                new_positions = pos_frag[list_frags]
                ordered_id_frags = list_frags[np.argsort(new_positions)]
                list_seq_new_contigs[id_cont] = ""
                for f in ordered_id_frags:
                    ori = ori_frag[f]
                    init_frag_id = vect_frags.id_d[f]
                    init_contig = self.frags_init_contigs[init_frag_id]
                    start_bp = self.pyramid.spec_level[str(self.level)][
                        "fragments_dict"
                    ][init_frag_id + 1]["start_pos(bp)"]
                    end_bp = self.pyramid.spec_level[str(self.level)][
                        "fragments_dict"
                    ][init_frag_id + 1]["end_pos(bp)"]
                    extract_seq = self.pyramid.dict_sequence_contigs[
                        init_contig
                    ][start_bp:end_bp]
                    if ori == -1:
                        seq = extract_seq[::-1]
                        seq = seq.translate(
                            str.maketrans("TAGCtagc", "ATCGATCG")
                        )
                    else:
                        seq = extract_seq

                    handle_info_frags.write(
                        "%s\t%s\t%s\t%s\t%s\n"
                        % (
                            init_contig,
                            str(init_frag_id),
                            str(ori),
                            str(start_bp),
                            str(end_bp),
                        )
                    )
                    list_seq_new_contigs[id_cont] += seq
        for id_cont in list_contigs_ok:
            cont_seq = list_seq_new_contigs[id_cont]
            header = ">3C-assembly|contig_" + str(id_cont)
            # print header
            handle_new_fasta.write("%s\n" % (header))
            len_line = 61
            len_seq = len(cont_seq)
            if len_seq > 0:
                idx_cut_EOL = list(range(0, len_seq, len_line))
                for id_s in range(1, len(idx_cut_EOL)):
                    line = cont_seq[idx_cut_EOL[id_s - 1] : idx_cut_EOL[id_s]]
                    handle_new_fasta.write("%s\n" % (line))
                if idx_cut_EOL[-1] != len_seq - 1:
                    line = cont_seq[idx_cut_EOL[-1] :]
                    handle_new_fasta.write("%s\n" % (line))

        handle_new_fasta.close()
        handle_info_frags.close()


def main():
    try:
        in_file, out_file = sys.argv[1:]

    except IndexError:
        print("Usage: ./pyramid_sparse.py in_file out_file")
        quit()

    abs_contact_2_coo_file(in_file, out_file)


if __name__ == "__main__":
    main()
