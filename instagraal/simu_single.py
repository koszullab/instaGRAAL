#!/usr/bin/env python3

# import pp
import os
import instagraal.pyramid_sparse as pyr

# import Image
# from OpenGL.GL import *
import OpenGL.GL
from OpenGL.arrays import vbo
import numpy as np
from instagraal.cuda_lib_gl_single import sampler as sampler_lib

# from cuda_lib_gl import sampler as sampler_lib
import matplotlib.pyplot as plt

from instagraal import log
from instagraal.log import logger

logger.setLevel(log.CURRENT_LOG_LEVEL)

# cuda.init()


def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[:k], cols[-k:]
    elif k > 0:
        return rows[k:], cols[:-k]
    else:
        return rows, cols


class simulation:
    def __init__(
        self,
        name,
        folder_path,
        fasta,
        level,
        n_iterations,
        is_simu,
        gl_window,
        use_rippe,
        gl_size_im,
        thresh_factor=1,
        output_folder=None,
    ):
        self.name = name
        self.use_rippe = use_rippe
        self.str_sub_level = str(level - 1)
        self.str_level = str(level)
        self.thresh_factor = thresh_factor

        self.data_set = name

        toolbox_directory = os.path.dirname(os.path.abspath(__file__))

        self.data_set_root = toolbox_directory
        self.dir_home = toolbox_directory
        self.data_set_root = toolbox_directory

        self.fasta = fasta
        # default_level = size_pyramid - 1
        # self.base_folder = os.path.join(
        #     self.data_set_root, self.data_set
        # )
        self.base_folder = folder_path

        if output_folder is None:
            self.output_folder = os.path.join(self.data_set_root, "results")
        else:
            self.output_folder = output_folder

        self.select_data_set(name)
        self.n_iterations = n_iterations
        self.gl_size_im = gl_size_im
        self.int4 = np.dtype(
            [
                ("x", np.int32),
                ("y", np.int32),
                ("z", np.int32),
                ("w", np.int32),
            ],
            align=True,
        )
        self.float3 = np.dtype(
            [("x", np.float32), ("y", np.float32), ("z", np.float32)],
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
        self.int3 = np.dtype(
            [("x", np.int32), ("y", np.int32), ("z", np.int32)], align=True
        )
        self.int2 = np.dtype([("x", np.int32), ("y", np.int32)], align=True)

        self.level = self.hic_pyr.get_level(level)

        self.level.build_seq_per_bin(genome_fasta=self.fasta)
        self.sub_level = self.hic_pyr.get_level(level - 1)

        self.n_frags = self.level.n_frags
        self.create_sub_frags()

        self.mean_squared_frags_per_bin = np.float32(
            (self.collect_accu_frags.mean()) ** 2
        )
        logger.info(
            "mean frag area = {}".format(self.mean_squared_frags_per_bin)
        )
        self.gl_window = gl_window
        # DEFINE REPEATED SEQ ####
        (
            self.candidate_dup,
            self.data_candidate_dup,
            self.sub_candidates_dup,
            self.sub_candidates_output_data,
        ) = self.select_repeated_frags()

        self.modify_vect_frags()
        self.blacklist_contig()
        # self.discard_low_coverage_frags()
        self.load_gl_buffers()
        self.create_new_sub_frags()
        self.modify_sub_vect_frags()

        self.gl_size_im = min(gl_size_im, self.n_frags)
        self.init_gl_image()

        self.sampler = sampler_lib(
            use_rippe,
            self.new_S_o_A_frags,
            self.collector_id_repeats,
            self.frag_dispatcher,
            self.candidate_dup,
            self.frag_blacklisted,
            self.init_n_frags,
            self.n_frags,
            self.init_n_sub_frags,
            self.n_new_sub_frags,
            self.rep_sub_frags_id,
            self.level.sparse_mat_csr,
            self.np_sub_frags_len_bp,
            self.np_sub_frags_id,
            self.np_sub_frags_accu,
            self.np_sub_frags_2_frags,
            self.mean_squared_frags_per_bin,
            self.norm_vect,
            self.sub_candidates_dup,
            self.sub_candidates_output_data,
            self.new_sub_S_o_A_frags,
            self.sub_collector_id_repeats,
            self.sub_frag_dispatcher,
            self.sub_level.sparse_mat_csr,
            self.sub_level.mean_value_trans,
            n_iterations,
            is_simu,
            self.gl_window,
            self.pos_vbo,
            self.col_vbo,
            self.vel,
            self.pos,
            self.raw_im_init,
            self.pbo_im_buffer,
            self.gl_size_im,
        )

        # display_graph = False
        display_graph = True

        id_start = np.nonzero(self.sampler.gpu_vect_frags.start_bp == 0)[0]
        max_dist_kb = (
            self.sampler.gpu_vect_frags.l_cont_bp[id_start].max() / 1000.
        )
        logger.info("max dist kb = {}".format(max_dist_kb))
        # size_bin_kb = self.sampler.gpu_vect_frags.len_bp.mean() / 1000.0
        mean_size_bin_kb = self.new_sub_S_o_A_frags["len_bp"].mean() / 1000.0

        logger.info("mean size kb = {}".format(mean_size_bin_kb))

        logger.info(
            "min fragment length =  {}".format(
                self.new_sub_S_o_A_frags["len_bp"].min() / 1000.0
            )
        )
        if is_simu:
            self.sampler.simulate_rippe_contacts(
                100, 9.6, -1.5, 0.5, 1, 800, 200
            )
        else:
            if self.use_rippe:
                self.sampler.estimate_parameters_rippe(
                    max_dist_kb, mean_size_bin_kb / 2., display_graph
                )
            else:
                self.sampler.estimate_parameters(
                    max_dist_kb, mean_size_bin_kb / 2, display_graph
                )

        # self.sampler.setup_texture()

    def blacklist_contig(self):
        # list_blacklist_manual = raw_input("ids (separated by space): ")
        list_blacklist_manual = ""
        if list_blacklist_manual != "":
            list_blacklist_manual = list_blacklist_manual.split(" ")
            candidates_blacklist = [int(i) for i in list_blacklist_manual]
        else:
            candidates_blacklist = []
        init_vect_frags = self.level.S_o_A_frags
        list_id_c = init_vect_frags["id_c"]

        # candidates_blacklist =  []

        frag_blacklisted = []

        for id_c_black in candidates_blacklist:
            id_black_list = np.nonzero(list_id_c == id_c_black)[0]
            for init_f in id_black_list:
                dis = self.frag_dispatcher[init_f]
                ids = self.collector_id_repeats[dis["x"] : dis["y"]]
                frag_blacklisted.extend(list(ids))

        self.frag_blacklisted = frag_blacklisted
        for id_f_black in self.frag_blacklisted:
            self.col_vect_frags_4_GL[id_f_black, 0] = np.float32(0)
            self.col_vect_frags_4_GL[id_f_black, 1] = np.float32(0)
            self.col_vect_frags_4_GL[id_f_black, 2] = np.float32(0)
            self.col_vect_frags_4_GL[id_f_black, 3] = np.float32(0)

    def discard_low_coverage_frags(self):
        mat = np.copy(self.level.im_curr)
        mat_norm = np.array(
            self.norm_vect.T * self.norm_vect, dtype=np.float32
        )
        self.matrix_normalized = mat / mat_norm
        coverage = self.matrix_normalized.sum(axis=1)
        mean_coverage = coverage.mean()
        std_coverage = coverage.std()
        mean_coverage_ext = mean_coverage - 0.1 * std_coverage
        candidates_low = np.nonzero(coverage < mean_coverage_ext)[0]
        logger.info(
            "n discarded frag of low coverage = {}".format(
                candidates_low.shape[0]
            )
        )
        for init_f in candidates_low:
            dis = self.frag_dispatcher[init_f]
            ids = self.collector_id_repeats[dis["x"] : dis["y"]]
            self.frag_blacklisted.extend(list(ids))

    def modify_vect_frags(self):

        "include repeated frags"
        modified_vect_frags = dict()
        init_vect_frags = self.level.S_o_A_frags

        # init_max_id_d = init_vect_frags["id"].max()
        max_id_F = len(init_vect_frags["id"])
        max_id_C = init_vect_frags["id_c"].max() + 1

        # HSV_tuples = [(x*1.0/(max_id_C - 1), 0.5, 0.5) for x in range(0,
        # (max_id_C-1))]
        # cmap = plt.cm.gist_ncar
        cmap = plt.cm.prism
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        id_smple = np.linspace(0, cmap.N, num=max_id_C)
        RGB_tuples = []
        for i in range(0, max_id_C - 1):
            RGB_tuples.append(cmaplist[int(id_smple[i])])

        # RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

        self.init_n_frags = len(init_vect_frags["id"])

        modified_vect_frags["pos"] = list(init_vect_frags["pos"])
        modified_vect_frags["sub_pos"] = list(init_vect_frags["sub_pos"])
        modified_vect_frags["id_c"] = list(init_vect_frags["id_c"])
        modified_vect_frags["start_bp"] = list(init_vect_frags["start_bp"])
        modified_vect_frags["len_bp"] = list(init_vect_frags["len_bp"])
        modified_vect_frags["sub_len"] = list(init_vect_frags["sub_len"])
        modified_vect_frags["circ"] = list(init_vect_frags["circ"])
        modified_vect_frags["id"] = list(init_vect_frags["id"])
        modified_vect_frags["prev"] = list(init_vect_frags["prev"])
        modified_vect_frags["next"] = list(init_vect_frags["next"])
        modified_vect_frags["l_cont"] = list(init_vect_frags["l_cont"])
        modified_vect_frags["sub_l_cont"] = list(init_vect_frags["sub_l_cont"])
        modified_vect_frags["l_cont_bp"] = list(init_vect_frags["l_cont_bp"])
        modified_vect_frags["n_accu"] = list(init_vect_frags["n_accu"])
        modified_vect_frags["rep"] = list(np.zeros(max_id_F, dtype=np.int32))
        modified_vect_frags["activ"] = list(np.ones(max_id_F, dtype=np.int32))
        modified_vect_frags["id_d"] = list(init_vect_frags["id"])

        for data_dup in self.data_candidate_dup:
            n_dup = int(data_dup[1])
            id_f = data_dup[0]
            for k in range(0, n_dup):
                modified_vect_frags["pos"].append(0)
                modified_vect_frags["sub_pos"].append(0)
                modified_vect_frags["id_c"].append(max_id_C)
                modified_vect_frags["start_bp"].append(0)
                modified_vect_frags["len_bp"].append(
                    init_vect_frags["len_bp"][id_f]
                )
                modified_vect_frags["sub_len"].append(
                    init_vect_frags["sub_len"][id_f]
                )
                modified_vect_frags["circ"].append(
                    init_vect_frags["circ"][id_f]
                )
                modified_vect_frags["id"].append(max_id_F)
                modified_vect_frags["prev"].append(-1)
                modified_vect_frags["next"].append(-1)
                modified_vect_frags["l_cont"].append(1)
                modified_vect_frags["sub_l_cont"].append(
                    init_vect_frags["sub_len"][id_f]
                )
                modified_vect_frags["l_cont_bp"].append(
                    init_vect_frags["len_bp"][id_f]
                )
                modified_vect_frags["n_accu"].append(
                    init_vect_frags["n_accu"][id_f]
                )
                modified_vect_frags["rep"].append(1)
                modified_vect_frags["activ"].append(1)
                modified_vect_frags["id_d"].append(init_vect_frags["id"][id_f])
                max_id_F += 1
                max_id_C += 1

        modified_vect_frags["pos"] = np.array(
            modified_vect_frags["pos"], dtype=np.int32
        )
        modified_vect_frags["sub_pos"] = np.array(
            modified_vect_frags["sub_pos"], dtype=np.int32
        )
        modified_vect_frags["id_c"] = np.array(
            modified_vect_frags["id_c"], dtype=np.int32
        )
        modified_vect_frags["start_bp"] = np.array(
            modified_vect_frags["start_bp"], dtype=np.int32
        )
        modified_vect_frags["len_bp"] = np.array(
            modified_vect_frags["len_bp"], dtype=np.int32
        )
        modified_vect_frags["sub_len"] = np.array(
            modified_vect_frags["sub_len"], dtype=np.int32
        )
        modified_vect_frags["circ"] = np.array(
            modified_vect_frags["circ"], dtype=np.int32
        )
        modified_vect_frags["id"] = np.array(
            modified_vect_frags["id"], dtype=np.int32
        )
        modified_vect_frags["prev"] = np.array(
            modified_vect_frags["prev"], dtype=np.int32
        )
        modified_vect_frags["next"] = np.array(
            modified_vect_frags["next"], dtype=np.int32
        )
        modified_vect_frags["l_cont"] = np.array(
            modified_vect_frags["l_cont"], dtype=np.int32
        )
        modified_vect_frags["sub_l_cont"] = np.array(
            modified_vect_frags["sub_l_cont"], dtype=np.int32
        )
        modified_vect_frags["l_cont_bp"] = np.array(
            modified_vect_frags["l_cont_bp"], dtype=np.int32
        )
        modified_vect_frags["n_accu"] = np.array(
            modified_vect_frags["n_accu"], dtype=np.int32
        )
        modified_vect_frags["rep"] = np.array(
            modified_vect_frags["rep"], dtype=np.int32
        )
        modified_vect_frags["activ"] = np.array(
            modified_vect_frags["activ"], dtype=np.int32
        )
        modified_vect_frags["id_d"] = np.array(
            modified_vect_frags["id_d"], dtype=np.int32
        )

        id_x = 0
        collector_id_repeats = []
        frag_dispatcher = []
        # BUILD LINK BETWEEN FRAGS AND SUB FRAGS
        for id_f in range(0, self.init_n_frags):
            if id_f in self.candidate_dup:
                id_start = id_x
                id_dup = np.nonzero(modified_vect_frags["id_d"] == id_f)[0]
                collector_id_repeats.extend(list(id_dup))
                n_rep = len(id_dup)
                frag_dispatcher.append(
                    (np.int32(id_start), np.int32(id_start + n_rep))
                )
                id_x += n_rep
            else:
                id_start = id_x
                n_rep = 1
                frag_dispatcher.append(
                    (np.int32(id_start), np.int32(id_start + n_rep))
                )
                collector_id_repeats.append(id_f)
                id_x += 1

        self.collector_id_repeats = np.array(
            collector_id_repeats, dtype=np.int32
        )
        self.frag_dispatcher = np.array(frag_dispatcher, dtype=self.int2)

        self.n_frags = len(modified_vect_frags["id"])

        pos_vect_frags_4_GL = np.ndarray((self.n_frags, 4), dtype=np.float32)
        col_vect_frags_4_GL = np.ndarray((self.n_frags, 4), dtype=np.float32)

        for id_f_curr in range(0, self.n_frags):
            id_d = modified_vect_frags["id_d"][id_f_curr]
            id_c = init_vect_frags["id_c"][id_d]
            pos_vect_frags_4_GL[id_f_curr, 0] = modified_vect_frags["pos"][
                id_f_curr
            ]
            pos_vect_frags_4_GL[id_f_curr, 1] = modified_vect_frags["id_c"][
                id_f_curr
            ]
            pos_vect_frags_4_GL[id_f_curr, 2] = 0.
            pos_vect_frags_4_GL[id_f_curr, 3] = np.float32(1.0)

            col_vect_frags_4_GL[id_f_curr, 0] = np.float32(
                RGB_tuples[id_c - 1][0]
            )
            col_vect_frags_4_GL[id_f_curr, 1] = np.float32(
                RGB_tuples[id_c - 1][1]
            )
            col_vect_frags_4_GL[id_f_curr, 2] = np.float32(
                RGB_tuples[id_c - 1][2]
            )
            col_vect_frags_4_GL[id_f_curr, 3] = np.float32(1.0)

        self.col_vect_frags_4_GL = col_vect_frags_4_GL
        self.pos_vect_frags_4_GL = pos_vect_frags_4_GL
        self.new_S_o_A_frags = modified_vect_frags

    def modify_sub_vect_frags(self):
        "include repeated frags"
        modified_vect_frags = dict()
        init_vect_frags = self.sub_level.S_o_A_frags

        # init_max_id_d = init_vect_frags["id"].max()
        max_id_F = len(init_vect_frags["id"])
        max_id_C = init_vect_frags["id_c"].max() + 1

        # HSV_tuples = [(x*1.0/(max_id_C - 1), 0.5, 0.5) for x in range(0,
        # (max_id_C-1))]
        # cmap = plt.cm.gist_ncar
        cmap = plt.cm.prism
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        id_smple = np.linspace(0, cmap.N, num=max_id_C)
        RGB_tuples = []
        for i in range(0, max_id_C - 1):
            RGB_tuples.append(cmaplist[int(id_smple[i])])

        # RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

        self.init_n_sub_frags = len(init_vect_frags["id"])

        modified_vect_frags["pos"] = list(init_vect_frags["pos"])
        modified_vect_frags["sub_pos"] = list(init_vect_frags["sub_pos"])
        modified_vect_frags["id_c"] = list(init_vect_frags["id_c"])
        modified_vect_frags["start_bp"] = list(init_vect_frags["start_bp"])
        modified_vect_frags["len_bp"] = list(init_vect_frags["len_bp"])
        modified_vect_frags["sub_len"] = list(init_vect_frags["sub_len"])
        modified_vect_frags["circ"] = list(init_vect_frags["circ"])
        modified_vect_frags["id"] = list(init_vect_frags["id"])
        modified_vect_frags["prev"] = list(init_vect_frags["prev"])
        modified_vect_frags["next"] = list(init_vect_frags["next"])
        modified_vect_frags["l_cont"] = list(init_vect_frags["l_cont"])
        modified_vect_frags["sub_l_cont"] = list(init_vect_frags["sub_l_cont"])
        modified_vect_frags["l_cont_bp"] = list(init_vect_frags["l_cont_bp"])
        modified_vect_frags["n_accu"] = list(init_vect_frags["n_accu"])
        modified_vect_frags["rep"] = list(np.zeros(max_id_F, dtype=np.int32))
        modified_vect_frags["activ"] = list(np.ones(max_id_F, dtype=np.int32))
        modified_vect_frags["id_d"] = list(init_vect_frags["id"])
        # WARNING IMPLICT BREAKING OF THE CONTIGS
        for data_dup in self.sub_candidates_output_data:
            n_dup = int(data_dup[1])
            id_f = data_dup[0]
            for k in range(0, n_dup):
                modified_vect_frags["pos"].append(0)
                modified_vect_frags["sub_pos"].append(0)
                modified_vect_frags["id_c"].append(max_id_C)
                modified_vect_frags["start_bp"].append(0)
                modified_vect_frags["len_bp"].append(
                    init_vect_frags["len_bp"][id_f]
                )
                modified_vect_frags["sub_len"].append(
                    init_vect_frags["sub_len"][id_f]
                )
                modified_vect_frags["circ"].append(
                    init_vect_frags["circ"][id_f]
                )
                modified_vect_frags["id"].append(max_id_F)
                modified_vect_frags["prev"].append(-1)
                modified_vect_frags["next"].append(-1)
                modified_vect_frags["l_cont"].append(1)
                modified_vect_frags["sub_l_cont"].append(
                    init_vect_frags["sub_len"][id_f]
                )
                modified_vect_frags["l_cont_bp"].append(
                    init_vect_frags["len_bp"][id_f]
                )
                modified_vect_frags["n_accu"].append(
                    init_vect_frags["n_accu"][id_f]
                )
                modified_vect_frags["rep"].append(1)
                modified_vect_frags["activ"].append(1)
                modified_vect_frags["id_d"].append(init_vect_frags["id"][id_f])
                max_id_F += 1
                max_id_C += 1

        logger.info("MAX ID CONTIG = {}".format(max_id_C))

        modified_vect_frags["pos"] = np.array(
            modified_vect_frags["pos"], dtype=np.int32
        )
        modified_vect_frags["sub_pos"] = np.array(
            modified_vect_frags["sub_pos"], dtype=np.int32
        )
        modified_vect_frags["id_c"] = np.array(
            modified_vect_frags["id_c"], dtype=np.int32
        )
        modified_vect_frags["start_bp"] = np.array(
            modified_vect_frags["start_bp"], dtype=np.int32
        )
        modified_vect_frags["len_bp"] = np.array(
            modified_vect_frags["len_bp"], dtype=np.int32
        )
        modified_vect_frags["sub_len"] = np.array(
            modified_vect_frags["sub_len"], dtype=np.int32
        )
        modified_vect_frags["circ"] = np.array(
            modified_vect_frags["circ"], dtype=np.int32
        )
        modified_vect_frags["id"] = np.array(
            modified_vect_frags["id"], dtype=np.int32
        )
        modified_vect_frags["prev"] = np.array(
            modified_vect_frags["prev"], dtype=np.int32
        )
        modified_vect_frags["next"] = np.array(
            modified_vect_frags["next"], dtype=np.int32
        )
        modified_vect_frags["l_cont"] = np.array(
            modified_vect_frags["l_cont"], dtype=np.int32
        )
        modified_vect_frags["sub_l_cont"] = np.array(
            modified_vect_frags["sub_l_cont"], dtype=np.int32
        )
        modified_vect_frags["l_cont_bp"] = np.array(
            modified_vect_frags["l_cont_bp"], dtype=np.int32
        )
        modified_vect_frags["n_accu"] = np.array(
            modified_vect_frags["n_accu"], dtype=np.int32
        )
        modified_vect_frags["rep"] = np.array(
            modified_vect_frags["rep"], dtype=np.int32
        )
        modified_vect_frags["activ"] = np.array(
            modified_vect_frags["activ"], dtype=np.int32
        )
        modified_vect_frags["id_d"] = np.array(
            modified_vect_frags["id_d"], dtype=np.int32
        )

        id_x = 0
        collector_id_repeats = []
        frag_dispatcher = []
        for id_f in range(0, self.init_n_sub_frags):
            if id_f in self.sub_candidates_dup:
                id_start = id_x
                id_dup = np.nonzero(modified_vect_frags["id_d"] == id_f)[0]
                collector_id_repeats.extend(list(id_dup))
                n_rep = len(id_dup)
                frag_dispatcher.append(
                    (np.int32(id_start), np.int32(id_start + n_rep))
                )
                id_x += n_rep
            else:
                id_start = id_x
                n_rep = 1
                frag_dispatcher.append(
                    (np.int32(id_start), np.int32(id_start + n_rep))
                )
                collector_id_repeats.append(id_f)
                id_x += 1

        self.sub_collector_id_repeats = np.array(
            collector_id_repeats, dtype=np.int32
        )
        self.sub_frag_dispatcher = np.array(frag_dispatcher, dtype=self.int2)

        self.sub_n_frags = len(modified_vect_frags["id"])

        # pos_vect_frags_4_GL = np.ndarray((self.n_frags, 4), dtype=np.float32)
        # col_vect_frags_4_GL = np.ndarray((self.n_frags, 4), dtype=np.float32)
        #
        # for id_f_curr in xrange(0 , self.sub_n_frags):
        #     id_d = modified_vect_frags['id_d'][id_f_curr]
        #     id_c = init_vect_frags['id_c'][id_d]
        #     pos_vect_frags_4_GL[id_f_curr, 0] =
        #     modified_vect_frags['pos'][id_f_curr]
        #     pos_vect_frags_4_GL[id_f_curr, 1] =
        #     modified_vect_frags['id_c'][id_f_curr]
        #     pos_vect_frags_4_GL[id_f_curr, 2] = 0.
        #     pos_vect_frags_4_GL[id_f_curr, 3] = np.float32(1.0)
        #
        #     col_vect_frags_4_GL[id_f_curr, 0] =
        #     np.float32(RGB_tuples[id_c - 1][0])
        #     col_vect_frags_4_GL[id_f_curr, 1] =
        #     np.float32(RGB_tuples[id_c - 1][1])
        #     col_vect_frags_4_GL[id_f_curr, 2] =
        #     np.float32(RGB_tuples[id_c - 1][2])
        #     col_vect_frags_4_GL[id_f_curr, 3] =
        #     np.float32(1.0)
        #
        # self.sub_col_vect_frags_4_GL = col_vect_frags_4_GL
        # self.sub_pos_vect_frags_4_GL = pos_vect_frags_4_GL
        self.new_sub_S_o_A_frags = modified_vect_frags

    def select_repeated_frags(self):  #

        # collect_cov = []
        # step = 0
        # p = ProgressBar('green', width=20, block='▣', empty='□')
        # for i in range(0, self.level.n_frags):
        #     v_r = self.level.sparse_mat_csr[i, :]
        #     v_c = self.level.sparse_mat_csc[:, i]
        #     non_zeros = v_c.nnz + v_r.nnz
        #     collect_cov.append(non_zeros)
        #     step += 1
        #     if step%1000 == 0:
        #         pt = step * 100 / self.level.n_frags
        #         p.render(pt, 'step %s\nProcessing...\nDescription: computing
        #         coverage per frag.' % step)

        coverage = np.array(self.level.sparse_mat_csr.sum(axis=0))[0]
        coverage += np.array(
            self.level.sparse_mat_csr.transpose().sum(axis=0)
        )[0]
        mean_coverage = coverage.mean()

        std_coverage = coverage.std()
        mean_coverage_ext = mean_coverage + 3 * std_coverage
        candidates_dup = np.nonzero(coverage > mean_coverage_ext)[0]
        # plt.figure()
        # plt.hist(coverage, 100)
        # plt.figure()
        # plt.plot(coverage)
        # plt.axhline(mean_coverage_ext, color='g')
        # # plt.show()
        # plt.figure()
        # n, bins, patches = plt.hist(coverage, 10000, normed=1,
        # facecolor='blue', alpha=0.75)
        # # add a 'best fit' line
        # (mu, sigma) = norm.fit(coverage)
        # y = mlab.normpdf( bins, mu, sigma)
        # l = plt.plot(bins, y, 'r--', linewidth=2)
        # plt.axvline(mean_coverage_ext, color='g', linewidth=2)
        # #plot
        # plt.xlabel('Raw contacts frequency')
        # plt.ylabel('Probability')
        # plt.title(r'$\mathrm{Histogram\ of\ HiC\ contact\ (data\ %s):}\
        # \mu=%.3f,\ \sigma=%.3f$' %(self.name, mu, sigma))
        # plt.legend(["gaussian fit","duplication limit","exp distribution"],
        # prop={'size':15})
        # plt.grid(True)
        #
        # plt.show()
        # DEBUGGGG ###############s##########
        # candidates_dup = []
        # print "candidate frags for duplication = ", candidates_dup
        test = ""
        # test = raw_input("ok?")

        if not (test == ""):
            # print "---enter id duplicated frags--"
            list_dup_manual = input("ids (separated by space): ")
            if list_dup_manual != "":
                list_dup_manual = list_dup_manual.split(" ")
                candidates_dup = [int(i) for i in list_dup_manual]
            else:
                candidates_dup = []
        # DEBUGGGG #########################
        # candidates_dup = range(880, 890)
        # candidates_dup = range(663, 674)
        # candidates_dup = range(0, 800)
        # candidates_dup = range(1204, 1217)

        candidates_dup = []
        # candidates_dup = range(1978, 2009)
        # print "you have selected: ", candidates_dup
        output_data = []
        sub_candidates_dup = []
        sub_candidates_output_data = []
        for ele in candidates_dup:
            cov_ele = coverage[ele]
            estim_n_dup = np.max(
                [1, np.round(cov_ele / mean_coverage_ext) - 1]
            )
            # estim_n_dup = np.max([1, np.ceil(cov_ele / mean_coverage_ext) -
            # 1])
            output_data.append((ele, estim_n_dup))
            # print "duplicated data = ", output_data
            tmp_sub_ids = self.np_sub_frags_id[ele]
            n_subs = tmp_sub_ids[-1]
            for i in range(0, n_subs):
                sub_candidates_dup.append(tmp_sub_ids[i])
                sub_candidates_output_data.append(
                    (tmp_sub_ids[i], estim_n_dup)
                )
        logger.info("N frag duplicated = {}".format(len(candidates_dup)))
        return (
            candidates_dup,
            output_data,
            sub_candidates_dup,
            sub_candidates_output_data,
        )

    def select_data_set(self, name):

        size_pyramid = 9
        factor = 3

        self.hic_pyr = pyr.build_and_filter(
            self.base_folder,
            size_pyramid,
            factor,
            thresh_factor=self.thresh_factor,
        )
        logger.info("pyramid loaded")

        if not (os.path.exists(self.output_folder)):
            os.mkdir(self.output_folder)
        self.output_folder = os.path.join(self.output_folder, self.data_set)
        if not (os.path.exists(self.output_folder)):
            os.mkdir(self.output_folder)
        self.output_folder = os.path.join(
            self.output_folder, "test_mcmc_" + self.str_level
        )
        if not (os.path.exists(self.output_folder)):
            os.mkdir(self.output_folder)

        self.new_fasta = os.path.join(self.output_folder, "genome.fasta")
        self.info_frags = os.path.join(self.output_folder, "info_frags.txt")
        self.output_matrix_em = os.path.join(
            self.output_folder, "post_em.tiff"
        )
        self.output_matrix_mcmc = os.path.join(
            self.output_folder, "post_mcmc.tiff"
        )
        self.input_matrix = os.path.join(self.output_folder, "pre_simu.tiff")
        self.scrambled_input_matrix = os.path.join(
            self.output_folder, "scrambled_simu.tiff"
        )

    def load_gl_buffers(self):
        num = self.n_frags
        pos = np.ndarray((num, 4), dtype=np.float32)
        seed = np.random.rand(2, num)
        pos[:, 0] = seed[0, :]
        pos[:, 1] = 0.0
        pos[:, 2] = seed[1, :]  # z pos
        pos[:, 3] = 1.  # velocity

        # num = self.n_frags
        # pos = np.ndarray((num, 4), dtype=np.float32)
        # pos[:,1] = np.sin(np.arange(0., num) * 2.001 * np.pi / (10*num))
        # pos[:,1] *= np.random.random_sample((num,)) / 3. - 0.2
        # pos[:,2] = np.cos(np.arange(0., num) * 2.001 * np.pi /(10* num))
        # pos[:,2] *= np.random.random_sample((num,)) / 3. - 0.2
        # pos[:,0] = 0. # z pos
        # pos[:,3] = 1. # velocity
        self.pos = pos
        self.pos_vbo = vbo.VBO(
            data=self.pos,
            usage=OpenGL.GL.GL_DYNAMIC_DRAW,
            target=OpenGL.GL.GL_ARRAY_BUFFER,
        )

        # self.pos_vbo = vbo.VBO(data=self.pos_vect_frags_4_GL,
        # usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.pos_vbo.bind()

        self.col_vbo = vbo.VBO(
            data=self.col_vect_frags_4_GL,
            usage=OpenGL.GL.GL_DYNAMIC_DRAW,
            target=OpenGL.GL.GL_ARRAY_BUFFER,
        )

        self.col_vbo.bind()

        self.vel = np.ndarray((self.n_frags, 4), dtype=np.float32)
        self.vel[:, 2] = self.pos[:, 2] * 2.
        self.vel[:, 1] = self.pos[:, 1] * 2.
        self.vel[:, 0] = 3.
        self.vel[:, 3] = np.random.random_sample((self.n_frags,))

    def init_gl_image(self,):

        self.texid = 0
        self.pbo_im_buffer = OpenGL.GL.glGenBuffers(
            1
        )  # generate 1 buffer reference

        OpenGL.GL.glBindBuffer(
            OpenGL.GL.GL_PIXEL_UNPACK_BUFFER, self.pbo_im_buffer
        )  # binding to this buffer

        # self.raw_im_init = np.uint8(np.random.rand(self.gl_size_im,
        # self.gl_size_im))
        self.raw_im_init = np.zeros(
            (self.gl_size_im, self.gl_size_im), dtype=np.uint8
        )

        OpenGL.GL.glBufferData(
            OpenGL.GL.GL_PIXEL_UNPACK_BUFFER,
            self.gl_size_im * self.gl_size_im,
            self.raw_im_init,
            OpenGL.GL.GL_STREAM_DRAW,
        )  # Allocate the buffer

        OpenGL.GL.glGetBufferParameteriv(
            OpenGL.GL.GL_PIXEL_UNPACK_BUFFER, OpenGL.GL.GL_BUFFER_SIZE
        )  # Check allocated buffer size

        #        try:
        #            assert(bsize == self.gl_size_im * self.gl_size_im)
        #        except AssertionError as e:
        #            print str(e)
        OpenGL.GL.glBindBuffer(OpenGL.GL.GL_PIXEL_UNPACK_BUFFER, 0)  # Unbind

        OpenGL.GL.glGenTextures(1, self.texid)  # generate 1 texture reference

        OpenGL.GL.glBindTexture(
            OpenGL.GL.GL_TEXTURE_2D, self.texid
        )  # binding to this texture

        OpenGL.GL.glTexParameteri(
            OpenGL.GL.GL_TEXTURE_2D,
            OpenGL.GL.GL_TEXTURE_MAG_FILTER,
            OpenGL.GL.GL_LINEAR,
        )
        OpenGL.GL.glTexParameteri(
            OpenGL.GL.GL_TEXTURE_2D,
            OpenGL.GL.GL_TEXTURE_MIN_FILTER,
            OpenGL.GL.GL_LINEAR,
        )

        # glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, self.gl_size_im,
        # self.gl_size_im,  0, GL_LUMINANCE, GL_FLOAT, self.raw_im_init) #
        # Allocate the texture
        OpenGL.GL.glTexImage2D(
            OpenGL.GL.GL_TEXTURE_2D,
            0,
            OpenGL.GL.GL_LUMINANCE,
            self.gl_size_im,
            self.gl_size_im,
            0,
            OpenGL.GL.GL_LUMINANCE,
            OpenGL.GL.GL_UNSIGNED_BYTE,
            None,
        )  # Allocate the texture

        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D, 0)  # Unbind

        OpenGL.GL.glPixelStorei(
            OpenGL.GL.GL_UNPACK_ALIGNMENT, 1
        )  # 1-byte row alignment
        OpenGL.GL.glPixelStorei(
            OpenGL.GL.GL_PACK_ALIGNMENT, 1
        )  # 1-byte row alignment

    def create_sub_frags(self):
        self.sub_frags_len_bp = []
        self.sub_frags_id = []
        self.sub_frags_accu = []
        unkb = np.float32(1000.0)
        self.collect_accu_frags = []
        self.norm_vect = []
        n_sub_frags = 0
        self.sub_frags_2_frags = []
        for i in range(0, self.n_frags):
            tmp = [
                self.hic_pyr.spec_level[self.str_level]["fragments_dict"][
                    i + 1
                ]["sub_low_index"]
                - 1,
                self.hic_pyr.spec_level[self.str_level]["fragments_dict"][
                    i + 1
                ]["sub_high_index"]
                - 1,
            ]

            # !!! warning!!! m
            n_sub = tmp[1] - tmp[0] + 1
            v_len = [0, 0, 0]
            v_accu = [0, 0, 0]
            v_id = [0, 0, 0, n_sub]
            n_sub_frags += n_sub
            for j in range(0, n_sub):
                v_len[j] = (
                    np.float32(
                        self.sub_level.vect_frag_np[tmp[0] + j]["len_bp"]
                    )
                    / unkb
                )
                v_id[j] = np.int32(tmp[0] + j)
                v_accu[j] = np.int32(
                    self.sub_level.vect_frag_np[tmp[0] + j]["n_accu"]
                )
                self.collect_accu_frags.append(v_accu[j])

            # create index to go from high res frags to low res frags
            tmp_len = np.array(v_len[0:n_sub], dtype=np.float32)
            for j in range(0, n_sub):
                w_d = np.sum(tmp_len[0:j]) + tmp_len[j] / 2.  # watson distance
                c_d = (
                    np.sum(tmp_len[list(range(n_sub - 1, j, -1))])
                    + tmp_len[j] / 2.
                )  # crick distance
                # dat = (i, w_d, c_d)
                dat = (i, w_d, c_d, j)
                self.sub_frags_2_frags.append(dat)

            self.norm_vect.append(np.sum(v_accu))
            self.sub_frags_len_bp.append(tuple(v_len))
            self.sub_frags_id.append(tuple(v_id))
            self.sub_frags_accu.append(tuple(v_accu))

        # self.np_sub_frags_2_frags = np.array(self.sub_frags_2_frags,
        # dtype=self.float3)
        self.np_sub_frags_2_frags = np.array(
            self.sub_frags_2_frags, dtype=self.float4
        )
        self.np_sub_frags_len_bp = np.array(
            self.sub_frags_len_bp, dtype=self.float3
        )
        self.np_sub_frags_accu = np.array(self.sub_frags_accu, dtype=self.int3)
        self.np_sub_frags_id = np.array(self.sub_frags_id, dtype=self.int4)
        self.collect_accu_frags = np.array(
            self.collect_accu_frags, dtype=np.float32
        )
        self.norm_vect = np.mat(self.norm_vect)
        self.init_n_sub_frags = n_sub_frags

    def create_new_sub_frags(self,):
        out = 0
        rep_sub_frags_id = []
        idx = 0
        for i in range(0, self.n_frags):
            id_d = self.new_S_o_A_frags["id_d"][i]
            n_sub = self.np_sub_frags_id[id_d]["w"]
            v_id = [0, 0, 0, n_sub]
            for j in range(0, n_sub):
                v_id[j] = np.int32(idx)
                idx += 1
            rep_sub_frags_id.append(tuple(v_id))
            out += n_sub
        self.n_new_sub_frags = out
        self.rep_sub_frags_id = np.array(rep_sub_frags_id, dtype=self.int4)

    def plot_info_simu(
        self,
        collect_likelihood_input,
        collect_n_contigs_input,
        file_plot,
        title_ax,
    ):
        collect_likelihood = np.array(collect_likelihood_input)
        collect_n_contigs = np.array(collect_n_contigs_input)
        len_collect = len(collect_likelihood)
        if len_collect > 1000:
            idx_2_plot = np.arange(1000, len_collect)
        else:
            idx_2_plot = np.arange(0, len_collect)

        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax1 = fig.add_subplot(111)
        ax1.plot(collect_likelihood[idx_2_plot], "r-")
        ax1.set_xlabel("iterations")
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel("likelihood", color="r")
        for tl in ax1.get_yticklabels():
            tl.set_color("r")
        ax2 = ax1.twinx()
        # if title_ax == "distance from init genome":
        #     ax2.semilogy(collect_n_contigs, 'b-')
        # else:
        ax2.plot(collect_n_contigs[idx_2_plot], "b-")
        ax2.set_ylabel(title_ax, color="b")
        for tl in ax2.get_yticklabels():
            tl.set_color("b")
        plt.show()
        fig.savefig(file_plot)

        if len_collect > 1000:
            plt.figure()
            plt.hist(collect_n_contigs[idx_2_plot], 100)
            plt.title("histogram " + title_ax)
            plt.xlabel(title_ax)
            plt.ylabel("counts")
            plt.show()

    # def plot_all_info_simu(self, all_data, headers):
    #     print "here we go!!"
    #     folder = self.output_folder
    #     ## generate files ##
    #     n_vals =
    #     # the main axes is subplot(111) by default
    #     plt.plot(t, s)
    #     plt.axis([0, 1, 1.1*amin(s), 2*amax(s) ])
    #     plt.xlabel('time (s)')
    #     plt.ylabel('current (nA)')
    #     plt.title('Gaussian colored noise')
    #
    #     # this is an inset axes over the main axes
    #     a = plt.axes([.65, .6, .2, .2], axisbg='w')
    #     n, bins, patches = plt.hist(s, 400, normed=1)
    #     plt.title('Counts')
    #     plt.setp(a, xticks=[], yticks=[])
    #     plt.show()

    def export_new_fasta(self):
        self.sampler.gpu_vect_frags.copy_from_gpu()
        self.level.generate_new_fasta(
            self.sampler.gpu_vect_frags, self.new_fasta, self.info_frags
        )

    def release(self):
        self.sampler.free_gpu()
