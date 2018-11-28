#!/usr/bin/env python3

import numpy as np

# import pyopencl as cl
import pycuda.tools
from pycuda import characterize
import pycuda.driver as cuda
import pycuda.compiler
from instagraal.gpustruct import GPUStruct
from pycuda import gpuarray as ga

# from pycuda.scan import InclusiveScanKernel

import pycuda.gl as cudagl

import codepy.cgen as cgen
from codepy.bpl import BoostPythonModule
from codepy.cuda import CudaModule
import codepy.jit
import codepy.toolchain

import time
import matplotlib.pyplot as plt

# import optim_rippe_curve as opti1
import instagraal.optim_rippe_curve_update as opti
import instagraal.init_nuisance as nuis

# from OpenGL.arrays import vbo
import scipy as scp

from instagraal import log
from instagraal.log import logger

import pkg_resources

logger.setLevel(log.CURRENT_LOG_LEVEL)

# from scipy import ndimage as ndi
# from scipy import stats
# import Image


class sampler:
    def __init__(
        self,
        use_rippe,
        S_o_A_frags,
        collector_id_repeats,
        frag_dispatcher,
        id_frag_duplicated,
        id_frags_blacklisted,
        n_frags,
        n_new_frags,
        init_n_sub_frags,
        n_new_sub_frags,
        np_rep_sub_frags_id,
        sub_sampled_sparse_matrix,
        np_sub_frags_len_bp,
        np_sub_frags_id,
        np_sub_frags_accu,
        np_sub_frags_2_frags,
        mean_squared_frags_per_bin,
        norm_vect_accu,
        sub_candidates_dup,
        sub_candidates_output_data,
        S_o_A_sub_frags,
        sub_collector_id_repeats,
        sub_frag_dispatcher,
        sparse_matrix,
        mean_value_trans,
        n_iterations,
        is_simu,
        gl_window,
        pos_vbo,
        col_vbo,
        vel,
        pos,
        raw_im_init,
        pbo_im_buffer,
        gl_size_im,
    ):
        self.o = 0

        self.log_e = 0.43429448190325182
        self.sparse_matrix = sparse_matrix + sparse_matrix.transpose()
        self.n_single_frags = np.int32(
            init_n_sub_frags - len(sub_candidates_dup)
        )
        self.sub_sampled_sparse_matrix = sub_sampled_sparse_matrix

        self.np_sub_frags_2_frags = np_sub_frags_2_frags
        self.gpu_sub_frags_2_frags = cuda.mem_alloc(
            self.np_sub_frags_2_frags.nbytes
        )
        cuda.memcpy_htod(self.gpu_sub_frags_2_frags, self.np_sub_frags_2_frags)

        # repeatead candidates info
        self.sub_candidates_dup = sub_candidates_dup
        self.sub_candidates_output_data = sub_candidates_output_data
        self.init_n_sub_frags = np.int32(init_n_sub_frags)
        self.sparse_data_2_gpu()
        self.use_rippe = use_rippe
        self.gl_window = gl_window
        self.ctx = gl_window.ctx_gl
        self.pos_vbo = pos_vbo
        self.col_vbo = col_vbo
        self.pos = pos
        self.vel = vel
        self.load_gl_cuda_vbo()

        self.id_frags_blacklisted = id_frags_blacklisted
        self.int4 = np.dtype(
            [
                ("x", np.int32),
                ("y", np.int32),
                ("z", np.int32),
                ("n", np.int32),
            ],
            align=True,
        )
        self.float3 = np.dtype(
            [("x", np.float32), ("y", np.float32), ("z", np.float32)],
            align=True,
        )
        self.int3 = np.dtype(
            [("x", np.int32), ("y", np.int32), ("z", np.int32)], align=True
        )

        self.np_id_frag_duplicated = np.int32(id_frag_duplicated)
        self.id_frag_duplicated = id_frag_duplicated
        self.n_frags = np.int32(n_frags)
        self.n_new_frags = np.int32(n_new_frags)

        self.n_new_sub_frags = np.int32(n_new_sub_frags)

        self.uniq_frags = np.int32(
            np.lib.arraysetops.setdiff1d(
                np.arange(0, self.n_frags, dtype=np.int32),
                self.np_id_frag_duplicated,
            )
        )
        self.n_frags_uniq = np.int32(len(self.uniq_frags))

        self.n_values_triu = np.int32(self.n_frags * (self.n_frags - 1) / 2)
        self.init_n_values_triu = np.int32(
            self.n_frags * (self.n_frags - 1) / 2
        )
        self.new_n_values_triu = np.int32(
            self.n_new_frags * (self.n_new_frags - 1) / 2
        )
        self.init_n_sub_values_triu = np.int32(
            self.init_n_sub_frags * (self.init_n_sub_frags - 1) / 2
        )
        self.new_n_sub_values_triu = np.int32(
            self.n_new_sub_frags * (self.n_new_sub_frags - 1) / 2
        )

        self.init_n_values_triu_extra = self.init_n_values_triu + self.n_frags
        self.init_n_sub_values_triu_extra = (
            self.init_n_sub_values_triu + self.init_n_sub_frags
        )

        self.n_insert_blocks = 6
        # self.n_insert_blocks = 0
        self.n_tmp_struct = 12 + self.n_insert_blocks * 2
        if self.n_insert_blocks == 0:
            self.active_insert_blocks = False
            self.size_block_4_sub = 128
        else:
            self.active_insert_blocks = True
            self.size_block_4_sub = 64

        self.is_simu = is_simu
        self.norm_vect_accu = norm_vect_accu
        self.np_sub_frags_len_bp = np_sub_frags_len_bp
        self.np_sub_frags_id = np_sub_frags_id
        self.np_sub_frags_accu = np_sub_frags_accu
        # self.hic_matrix_sub_sampled = hic_matrix_sub_sampled
        self.mean_squared_frags_per_bin = np.float32(
            mean_squared_frags_per_bin
        )
        # print "size hic matrix = ", hic_matrix.nbytes/10**6

        self.collector_id_repeats = collector_id_repeats
        self.gpu_collector_id_repeats = ga.to_gpu(
            ary=self.collector_id_repeats
        )
        self.frag_dispatcher = frag_dispatcher
        self.gpu_frag_dispatcher = cuda.mem_alloc(self.frag_dispatcher.nbytes)
        cuda.memcpy_htod(self.gpu_frag_dispatcher, self.frag_dispatcher)
        self.np_rep_sub_frags_id = np_rep_sub_frags_id
        self.gpu_rep_sub_frags_id = cuda.mem_alloc_like(
            self.np_rep_sub_frags_id
        )
        cuda.memcpy_htod(self.gpu_rep_sub_frags_id, self.np_rep_sub_frags_id)
        self.gpu_uniq_frags = ga.to_gpu(ary=self.uniq_frags)

        self.sub_collector_id_repeats = sub_collector_id_repeats
        self.sub_frag_dispatcher = sub_frag_dispatcher
        self.gpu_sub_frag_dispatcher = cuda.mem_alloc_like(
            self.sub_frag_dispatcher
        )
        cuda.memcpy_htod(
            self.gpu_sub_frag_dispatcher, self.sub_frag_dispatcher
        )
        self.gpu_sub_collector_id_repeats = cuda.mem_alloc_like(
            self.sub_collector_id_repeats
        )
        cuda.memcpy_htod(
            self.gpu_sub_collector_id_repeats, self.sub_collector_id_repeats
        )

        self.S_o_A_frags = S_o_A_frags
        self.S_o_A_sub_frags = S_o_A_sub_frags
        self.mean_n_accu = np.int32(
            np.round(self.S_o_A_frags["n_accu"].mean())
        )
        self.mean_len_bp_frags = self.S_o_A_sub_frags["len_bp"].mean()

        self.mean_value_trans = mean_value_trans

        self.param_simu_rippe = np.dtype(
            [
                ("kuhn", np.float32),
                ("lm", np.float32),
                ("c1", np.float32),
                ("slope", np.float32),
                ("d", np.float32),
                ("d_max", np.float32),
                ("fact", np.float32),
                ("v_inter", np.float32),
            ],
            align=True,
        )

        self.param_simu_exp = np.dtype(
            [
                ("d0", np.float32),
                ("d_max", np.float32),
                ("alpha_0", np.float32),
                ("alpha_1", np.float32),
                ("fact", np.float32),
                ("v_inter", np.float32),
            ],
            align=True,
        )
        if self.use_rippe:
            self.param_simu_T = self.param_simu_rippe
        else:
            self.param_simu_T = self.param_simu_exp

        self.setup_all_gpu_struct()

        self.n_iterations = n_iterations

        self.np_init_prev = np.copy(self.S_o_A_frags["prev"])
        self.np_init_next = np.copy(self.S_o_A_frags["next"])
        self.np_init_orientable = []
        for idf in range(0, self.n_new_frags):
            id_d = self.S_o_A_frags["id_d"][idf]
            self.np_init_orientable.append(self.np_sub_frags_id[id_d]["w"] > 1)
        self.np_init_orientable = np.array(
            self.np_init_orientable, dtype=np.int32
        )
        self.np_init_ori = np.ones((self.n_new_frags,), dtype=np.int32)
        # ########################
        self.n_generators = 100
        seed = 1
        self.rng_states = cuda.mem_alloc(
            self.n_generators
            * characterize.sizeof(
                "curandStateXORWOW", "#include <curand_kernel.h>"
            )
        )

        (free, total) = cuda.mem_get_info()
        logger.debug(
            (
                "Global memory occupancy after init:%f%% free"
                % (free * 100. / total)
            )
        )
        logger.debug(
            ("Global free memory after init:%i Mo free" % (free / 10 ** 6.))
        )

        logger.info("loading kernels ...")
        kernel_adapt_entry_point = pkg_resources.resource_filename(
            "instagraal", "kernels/kernel_sparse_adapt.cu"
        )
        kernel_entry_point = pkg_resources.resource_filename(
            "instagraal", "kernels/kernel_sparse.cu"
        )
        if self.active_insert_blocks:
            self.loadProgram(kernel_adapt_entry_point)
        else:
            self.loadProgram(kernel_entry_point)
        logger.info("kernels compiled")

        self.stride = 200

        seed = 1
        self.init_rng(
            np.int32(self.n_generators),
            self.rng_states,
            np.uint64(seed),
            np.uint64(0),
            block=(64, 1, 1),
            grid=(self.n_generators // 64 + 1, 1),
        )
        self.setup_distri_frags()

        # THRUST MODULE ##
        # Make a host_module, compiled for CPU
        self.setup_thrust_modules()

        # self.texref = self.module.get_texref("tex")
        self.raw_im_init = raw_im_init
        self.pbo_im_buffer = pbo_im_buffer
        self.gl_size_im = gl_size_im
        precision = 1
        self.sparse_data_4_gl(precision)
        self.load_gl_cuda_tex_buffer(self.raw_im_init)
        self.im_thresh = 50

    def setup_all_gpu_struct(self,):

        self.n_threads_mutations = int(
            np.power(2, np.floor(np.log2(self.n_tmp_struct)) + 1)
        )
        self.gpu_vect_frags = self.create_gpu_struct(data=self.S_o_A_frags)

        self.cpu_id_contigs = np.copy(self.S_o_A_frags["id_c"])
        self.gpu_id_contigs = ga.to_gpu(self.cpu_id_contigs)
        self.collector_gpu_vect_frags = []

        for k in range(0, self.n_tmp_struct):
            self.collector_gpu_vect_frags.append(
                self.create_gpu_struct(data=None)
            )

        sub_vect_dist = np.ones(
            (self.n_new_sub_frags * self.n_tmp_struct,), dtype=np.float32
        )
        self.collect_gpu_vect_dist = ga.to_gpu(sub_vect_dist)
        sub_vect_id_c = np.ones(
            (self.n_new_sub_frags * self.n_tmp_struct,), dtype=np.int32
        )
        self.collect_gpu_vect_id_c = ga.to_gpu(sub_vect_id_c)
        sub_vect_s_tot = np.ones(
            (self.n_new_sub_frags * self.n_tmp_struct,), dtype=np.float32
        )
        self.collect_gpu_vect_s_tot = ga.to_gpu(sub_vect_s_tot)
        sub_vect_pos = np.ones(
            (self.n_new_sub_frags * self.n_tmp_struct,), dtype=np.int32
        )
        self.collect_gpu_vect_pos = ga.to_gpu(sub_vect_pos)
        sub_vect_len = np.ones(
            (self.n_new_sub_frags * self.n_tmp_struct,), dtype=np.int32
        )
        self.collect_gpu_vect_len = ga.to_gpu(sub_vect_len)

        self.pop_gpu_vect_frags = self.create_gpu_struct(data=None)
        self.scrambled_gpu_vect_frags = self.create_gpu_struct(data=None)
        self.pop_cpu_id_contigs = np.copy(self.cpu_id_contigs)
        self.pop_gpu_id_contigs = ga.to_gpu(self.pop_cpu_id_contigs)
        self.trans1_gpu_vect_frags = self.create_gpu_struct(data=None)
        self.trans1_cpu_id_contigs = np.copy(self.cpu_id_contigs)
        self.trans1_gpu_id_contigs = ga.to_gpu(self.trans1_cpu_id_contigs)
        self.trans2_gpu_vect_frags = self.create_gpu_struct(data=None)
        self.trans2_cpu_id_contigs = np.copy(self.cpu_id_contigs)
        self.trans2_gpu_id_contigs = ga.to_gpu(self.trans2_cpu_id_contigs)

        self.gpu_counter_select = ga.zeros(1, dtype=np.int32)
        self.gpu_counter_select_gl = ga.zeros(1, dtype=np.int32)
        self.gpu_list_len_cont = ga.zeros(self.n_frags, dtype=np.int32)
        self.gpu_likelihood_on_zeros = ga.zeros(1, dtype=np.float64)
        self.gpu_likelihood_on_zeros_nuis = ga.zeros(1, dtype=np.float64)
        self.gpu_vect_likelihood_z = ga.zeros(
            self.n_tmp_struct, dtype=np.float64
        )
        self.gpu_n_vals_intra = ga.zeros(1, dtype=np.int32)
        self.gpu_all_n_vals_intra = ga.zeros(self.n_tmp_struct, dtype=np.int32)
        self.gpu_n_tot_sub_frags = ga.zeros(1, dtype=np.int32)
        self.n_pixl_sub_mat = (
            self.init_n_sub_frags * (self.init_n_sub_frags - 1) / 2
        )

        sub_vect_dist = np.ones((self.n_new_sub_frags,), dtype=np.float32)
        self.gpu_vect_dist = ga.to_gpu(sub_vect_dist)
        sub_vect_id_c = np.ones((self.n_new_sub_frags,), dtype=np.int32)
        self.gpu_vect_id_c = ga.to_gpu(sub_vect_id_c)
        sub_vect_s_tot = np.ones((self.n_new_sub_frags,), dtype=np.float32)
        self.gpu_vect_s_tot = ga.to_gpu(sub_vect_s_tot)
        sub_vect_pos = np.ones((self.n_new_sub_frags,), dtype=np.int32)
        self.gpu_vect_pos = ga.to_gpu(sub_vect_pos)
        sub_vect_len = np.ones((self.n_new_sub_frags,), dtype=np.int32)
        self.gpu_vect_len = ga.to_gpu(sub_vect_len)

        self.gpu_vect_dist_mut = ga.zeros(
            (self.n_new_sub_frags,), dtype=np.float32
        )
        self.gpu_vect_id_c_mut = ga.zeros(
            (self.n_new_sub_frags,), dtype=np.int32
        )
        self.gpu_vect_s_tot_mut = ga.zeros(
            (self.n_new_sub_frags,), dtype=np.float32
        )
        self.gpu_vect_pos_mut = ga.zeros(
            (self.n_new_sub_frags,), dtype=np.int32
        )
        self.gpu_vect_len_mut = ga.zeros(
            (self.n_new_sub_frags,), dtype=np.int32
        )

        self.gpu_list_uniq_mutations = ga.zeros(
            (self.n_tmp_struct,), dtype=np.int32
        )
        self.gpu_n_uniq = ga.zeros((1,), dtype=np.int32)
        self.gpu_n_pixl_sub_mat = ga.zeros((1,), dtype=np.float64)
        self.gpu_n_pixl_sub_mat.fill(np.float64(self.n_pixl_sub_mat))

        self.gpu_sub_sp_block_indptr = ga.to_gpu(
            np.zeros((self.n_non_zero,), dtype=np.int32)
        )
        self.gpu_info_blocks = ga.to_gpu(
            np.zeros(
                (int(self.n_non_zero / self.size_block_4_sub + 1),),
                dtype=self.int3,
            )
        )

        self.gpu_sub_vect_likelihood_nz = ga.to_gpu(
            np.zeros((self.n_tmp_struct,), dtype=np.float64)
        )
        self.gpu_curr_likelihood_nz_extract = ga.zeros(1, dtype=np.float64)

        self.gpu_all_scores = ga.to_gpu(
            np.zeros((self.n_tmp_struct,), dtype=np.float64)
        )

        self.gpu_curr_likelihood_nz = ga.to_gpu(
            np.zeros((1,), dtype=np.float64)
        )
        self.gpu_curr_likelihood_nz_nuis = ga.to_gpu(
            np.zeros((1,), dtype=np.float64)
        )
        self.gpu_val_likelihood_4_mut = ga.to_gpu(
            np.zeros((1,), dtype=np.float64)
        )

        self.gpu_uniq_id_c = ga.zeros((self.n_new_sub_frags,), dtype=np.int32)
        self.gpu_uniq_len = ga.zeros((self.n_new_sub_frags,), dtype=np.int32)
        self.gpu_old_2_new_id_c = ga.zeros(
            np.int32((self.n_new_sub_frags + self.n_new_sub_frags / 10,)),
            dtype=np.int32,
        )  # security length

        if self.active_insert_blocks:
            # self.gpu_list_bounds = ga.to_gpu(ary=np.array(range(1,
            # self.n_insert_blocks + 1), dtype=np.int32))
            list_size = np.array(
                [1, 3, 5, 10, 20, 50, 200, 200], dtype=np.int32
            )
            self.max_bounds_insert = list_size[
                : self.n_insert_blocks
            ].max() * np.int32(
                np.round(self.S_o_A_frags["sub_len"].mean()) + 1
            )  # approximation of sub size to extract in full matrix
            self.gpu_list_valid_insert = ga.zeros(
                (self.n_insert_blocks * 2), dtype=np.int32
            )
            self.gpu_list_bounds = ga.to_gpu(
                ary=np.array(list_size[: self.n_insert_blocks], dtype=np.int32)
            )
            self.gpu_list_f_upstream = ga.zeros(
                (self.n_insert_blocks,), dtype=np.int32
            )
            self.gpu_list_f_downstream = ga.zeros(
                (self.n_insert_blocks,), dtype=np.int32
            )

    def setup_thrust_modules(self):

        host_mod = BoostPythonModule()
        # Make a device module, compiled with NVCC
        nvcc_mod = CudaModule(host_mod)
        # Describe device module code
        # NVCC includes
        nvcc_includes = [
            "thrust/sort.h",
            "thrust/iterator/zip_iterator.h",
            "thrust/device_vector.h",
            "cuda.h",
            "thrust/scan.h",
        ]
        # Add includes to module
        nvcc_mod.add_to_preamble([cgen.Include(x) for x in nvcc_includes])
        # NVCC function sort by keys 3 cuda arrays
        nvcc_function_sort_by_keys_zip = cgen.FunctionBody(
            cgen.FunctionDeclaration(
                cgen.Value("void", "my_sort_zip"),
                [
                    cgen.Value("CUdeviceptr", "input_ptr_keys"),
                    cgen.Value("int", "length"),
                    cgen.Value("CUdeviceptr", "input_ptr_valsa"),
                    cgen.Value("CUdeviceptr", "input_ptr_valsb"),
                ],
            ),
            cgen.Block(
                [
                    cgen.Statement(
                        "thrust::device_ptr<int> "
                        "thrust_ptr((int*)input_ptr_keys)"
                    ),
                    cgen.Statement(
                        "thrust::device_ptr<int> "
                        "thrust_ptr_valsa((int*)input_ptr_valsa)"
                    ),
                    cgen.Statement(
                        "thrust::device_ptr<int> "
                        "thrust_ptr_valsb((int*)input_ptr_valsb)"
                    ),
                    cgen.Statement(
                        "thrust::tuple< thrust::device_ptr<int>, "
                        "thrust::device_ptr<int> > keytup_begin = "
                        "thrust::make_tuple(thrust_ptr_valsa,thrust_ptr_valsb)"
                    ),
                    cgen.Statement(
                        "thrust::zip_iterator<thrust::tuple"
                        "<thrust::device_ptr<int>, thrust::device_ptr<int> > >"
                        " first = thrust::make_zip_iterator(keytup_begin)"
                    ),
                    cgen.Statement(
                        "thrust::stable_sort_by_key(thrust_ptr, "
                        "thrust_ptr+length, first)"
                    ),
                ]
            ),
        )

        # Add declaration to nvcc_mod
        # Adds declaration to host_mod as well
        nvcc_mod.add_function(nvcc_function_sort_by_keys_zip)

        # NVCC function sort by keys 3 cuda arrays
        nvcc_function_sort_by_keys_zip_cmplex = cgen.FunctionBody(
            cgen.FunctionDeclaration(
                cgen.Value("void", "my_sort_zip_cmplex"),
                [
                    cgen.Value("CUdeviceptr", "input_ptr_keys_a"),
                    cgen.Value("CUdeviceptr", "input_ptr_keys_b"),
                    cgen.Value("int", "length"),
                    cgen.Value("CUdeviceptr", "input_ptr_vals"),
                ],
            ),
            cgen.Block(
                [
                    cgen.Statement(
                        "thrust::device_ptr<int> "
                        "thrust_ptr_v((int*)input_ptr_vals)"
                    ),
                    cgen.Statement(
                        "thrust::device_ptr<int> "
                        "thrust_ptr_keysa((int*)input_ptr_keys_a)"
                    ),
                    cgen.Statement(
                        "thrust::device_ptr<int> "
                        "thrust_ptr_keysb((int*)input_ptr_keys_b)"
                    ),
                    cgen.Statement(
                        "thrust::tuple< thrust::device_ptr<int>, "
                        "thrust::device_ptr<int> > keytup_begin = "
                        "thrust::make_tuple(thrust_ptr_keysa,thrust_ptr_keysb)"
                    ),
                    cgen.Statement(
                        "thrust::zip_iterator<thrust::tuple<thrust::device_ptr"
                        "<int>, thrust::device_ptr<int> > > first = "
                        "thrust::make_zip_iterator(keytup_begin)"
                    ),
                    cgen.Statement(
                        "thrust::stable_sort_by_key(first, first+length, "
                        "thrust_ptr_v)"
                    ),
                ]
            ),
        )

        # Add declaration to nvcc_mod
        # Adds declaration to host_mod as well
        nvcc_mod.add_function(nvcc_function_sort_by_keys_zip_cmplex)

        # NVCC function sort by keys a 2 cuda arrays
        nvcc_function_sort_by_keys_simple = cgen.FunctionBody(
            cgen.FunctionDeclaration(
                cgen.Value("void", "my_sort_simple"),
                [
                    cgen.Value("CUdeviceptr", "input_ptr_keys"),
                    cgen.Value("int", "length"),
                    cgen.Value("CUdeviceptr", "input_ptr_vals"),
                ],
            ),
            cgen.Block(
                [
                    cgen.Statement(
                        "thrust::device_ptr<int> "
                        "thrust_ptr((int*)input_ptr_keys)"
                    ),
                    cgen.Statement(
                        "thrust::device_ptr<int> "
                        "thrust_ptr_v((int*)input_ptr_vals)"
                    ),
                    cgen.Statement(
                        "thrust::stable_sort_by_key(thrust_ptr, "
                        "thrust_ptr+length, thrust_ptr_v, "
                        "thrust::greater<int>())"
                    ),
                ]
            ),
        )
        # Add declaration to nvcc_mod
        # Adds declaration to host_mod as well
        nvcc_mod.add_function(nvcc_function_sort_by_keys_simple)

        # NVCC function prefix sum
        nvcc_function_prefix_sum = cgen.FunctionBody(
            cgen.FunctionDeclaration(
                cgen.Value("void", "my_prefix_sum"),
                [
                    cgen.Value("CUdeviceptr", "input_ptr_vals"),
                    cgen.Value("int", "length"),
                ],
            ),
            cgen.Block(
                [
                    cgen.Statement(
                        "thrust::device_ptr<int> "
                        "thrust_ptr_v((int*)input_ptr_vals)"
                    ),
                    cgen.Statement(
                        "thrust::exclusive_scan(thrust_ptr_v, "
                        "thrust_ptr_v+length,thrust_ptr_v)"
                    ),
                ]
            ),
        )
        # Add declaration to nvcc_mod
        # Adds declaration to host_mod as well
        nvcc_mod.add_function(nvcc_function_prefix_sum)

        host_includes = ["boost/python/extract.hpp"]
        # Add host includes to module
        host_mod.add_to_preamble([cgen.Include(x) for x in host_includes])

        host_namespaces = ["using namespace boost::python"]

        # Add BPL using statement
        host_mod.add_to_preamble([cgen.Statement(x) for x in host_namespaces])

        host_statements_sort_zip = [
            # Extract information from PyCUDA GPUArray
            # Get length
            # 'tuple shape = extract<tuple>(gpu_array_keys.attr("shape"))',
            "int length = n_vals",
            # 'int length = extract<int>(shape[0])',
            # Get data pointer
            "CUdeviceptr ptr_keys = "
            'extract<CUdeviceptr>(gpu_array_keys.attr("gpudata"))',
            "CUdeviceptr ptr_valsa = "
            'extract<CUdeviceptr>(gpu_array_valsa.attr("gpudata"))',
            "CUdeviceptr ptr_valsb = "
            'extract<CUdeviceptr>(gpu_array_valsb.attr("gpudata"))',
            # Call Thrust routine, compiled into the CudaModule
            "my_sort_zip(ptr_keys, length, ptr_valsa, ptr_valsb)",
            # Return result
            "return gpu_array_keys",
        ]
        host_mod.add_function(
            cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value("object", "sort_by_keys_zip"),
                    [
                        cgen.Value("object", "gpu_array_keys"),
                        cgen.Value("int", "n_vals"),
                        cgen.Value("object", "gpu_array_valsa"),
                        cgen.Value("object", "gpu_array_valsb"),
                    ],
                ),
                cgen.Block(
                    [cgen.Statement(x) for x in host_statements_sort_zip]
                ),
            )
        )

        host_statements_sort_zip_cmplex = [
            # Extract information from PyCUDA GPUArray
            # Get length
            # 'tuple shape = extract<tuple>(gpu_array_keys.attr("shape"))',
            "int length = n_vals",
            # 'int length = extract<int>(shape[0])',
            # Get data pointer
            """CUdeviceptr ptr_keys_a = """
            """extract<CUdeviceptr>(gpu_array_keys_a.attr("gpudata"))""",
            """CUdeviceptr ptr_keys_b = """
            """extract<CUdeviceptr>(gpu_array_keys_b.attr("gpudata"))""",
            """CUdeviceptr ptr_vals = """
            """extract<CUdeviceptr>(gpu_array_vals.attr("gpudata"))""",
            # Call Thrust routine, compiled into the CudaModule
            "my_sort_zip_cmplex(ptr_keys_a, ptr_keys_b, length, ptr_vals)",
            # Return result
            "return gpu_array_keys_a",
        ]
        host_mod.add_function(
            cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value("object", "sort_by_keys_zip_cmplex"),
                    [
                        cgen.Value("object", "gpu_array_keys_a"),
                        cgen.Value("object", "gpu_array_keys_b"),
                        cgen.Value("int", "n_vals"),
                        cgen.Value("object", "gpu_array_vals"),
                    ],
                ),
                cgen.Block(
                    [
                        cgen.Statement(x)
                        for x in host_statements_sort_zip_cmplex
                    ]
                ),
            )
        )

        host_statements_sort_simple = [
            # Extract information from PyCUDA GPUArray
            # Get length
            # 'tuple shape = extract<tuple>(gpu_array_keys.attr("shape"))',
            "int length = n_vals",
            # 'int length = extract<int>(shape[0])',
            # Get data pointer
            """CUdeviceptr ptr_keys = """
            """extract<CUdeviceptr>(gpu_array_keys.attr("gpudata"))""",
            """CUdeviceptr ptr_vals = """
            """extract<CUdeviceptr>(gpu_array_vals.attr("gpudata"))""",
            # Call Thrust routine, compiled into the CudaModule
            "my_sort_simple(ptr_keys, length, ptr_vals)",
            # Return result
            "return gpu_array_keys",
        ]
        host_mod.add_function(
            cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value("object", "sort_by_keys_simple"),
                    [
                        cgen.Value("object", "gpu_array_keys"),
                        cgen.Value("int", "n_vals"),
                        cgen.Value("object", "gpu_array_vals"),
                    ],
                ),
                cgen.Block(
                    [cgen.Statement(x) for x in host_statements_sort_simple]
                ),
            )
        )

        host_statements_prefix_sum = [
            # Extract information from PyCUDA GPUArray
            # Get length
            "int length = n_vals",
            # Get data pointer
            """CUdeviceptr ptr_vals = """
            """extract<CUdeviceptr>(gpu_array_vals.attr("gpudata"))""",
            # Call Thrust routine, compiled into the CudaModule
            "my_prefix_sum(ptr_vals, length)",
            # Return result
            "return gpu_array_vals",
        ]
        host_mod.add_function(
            cgen.FunctionBody(
                cgen.FunctionDeclaration(
                    cgen.Value("object", "prefix_sum"),
                    [
                        cgen.Value("object", "gpu_array_vals"),
                        cgen.Value("int", "n_vals"),
                    ],
                ),
                cgen.Block(
                    [cgen.Statement(x) for x in host_statements_prefix_sum]
                ),
            )
        )

        # Print out generated code, to see what we're actually compiling
        # print("---------------------- Host code ----------------------")
        # print(host_mod.generate())
        # print("--------------------- Device code ---------------------")
        # print(nvcc_mod.generate())
        # print("-------------------------------------------------------")

        # Compile modules

        gcc_toolchain = codepy.toolchain.guess_toolchain()
        nvcc_toolchain = codepy.toolchain.guess_nvcc_toolchain()

        self.thrust_module = nvcc_mod.compile(
            gcc_toolchain, nvcc_toolchain, debug=True
        )

    def create_gpu_struct(self, data):
        if data is None:
            gpu_vect = GPUStruct(
                [
                    (
                        np.int32,
                        "*pos",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*sub_pos",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*id_c",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*start_bp",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*len_bp",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*sub_len",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*circ",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*id",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*prev",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*next",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*l_cont",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*sub_l_cont",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*l_cont_bp",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*ori",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*rep",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*activ",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                    (
                        np.int32,
                        "*id_d",
                        np.zeros((self.n_new_frags,), dtype=np.int32),
                    ),
                ]
            )

        else:
            gpu_vect = GPUStruct(
                [
                    (np.int32, "*pos", np.copy(data["pos"])),
                    (np.int32, "*sub_pos", np.copy(data["sub_pos"])),
                    (np.int32, "*id_c", np.copy(data["id_c"])),
                    (np.int32, "*start_bp", np.copy(data["start_bp"])),
                    (np.int32, "*len_bp", np.copy(data["len_bp"])),
                    (np.int32, "*sub_len", np.copy(data["sub_len"])),
                    (np.int32, "*circ", np.copy(data["circ"])),
                    (np.int32, "*id", np.copy(data["id"])),
                    (np.int32, "*prev", np.copy(data["prev"])),
                    (np.int32, "*next", np.copy(data["next"])),
                    (np.int32, "*l_cont", np.copy(data["l_cont"])),
                    (np.int32, "*sub_l_cont", np.copy(data["sub_l_cont"])),
                    (np.int32, "*l_cont_bp", np.copy(data["l_cont_bp"])),
                    (
                        np.int32,
                        "*ori",
                        np.ones((self.n_new_frags,), dtype=np.int32),
                    ),
                    (np.int32, "*rep", np.copy(data["rep"])),
                    (np.int32, "*activ", np.copy(data["activ"])),
                    (np.int32, "*id_d", np.copy(data["id_d"])),
                ]
            )

        gpu_vect.copy_to_gpu()
        return gpu_vect

    def sparse_data_2_gpu(self,):

        # extract matrix withtout repeats
        id_single = []
        self.sym_sub_sampled_sparse_matrix = (
            self.sub_sampled_sparse_matrix + self.sub_sampled_sparse_matrix.T
        )
        for i in range(0, self.init_n_sub_frags):
            if i not in self.sub_candidates_dup:
                id_single.append(i)

        self.id_single = np.array(id_single, dtype=np.int32)
        self.id_rep = np.array(self.sub_candidates_dup, dtype=np.int32)
        mat_csr = self.sparse_matrix.tocsr()
        tmp_mat_1 = mat_csr[self.id_single, :]
        tmp_mat_2 = tmp_mat_1.tocsc()
        tmp_mat_3 = tmp_mat_2[:, self.id_single]
        self.mat_without_repeats = tmp_mat_3.tocsr()  # single Vs single

        self.hay_repeats = len(self.sub_candidates_dup) > 0
        if self.hay_repeats:
            self.mat_with_repeats = self.sparse_matrix[
                self.sub_candidates_dup, :
            ]  # repeats Vs all
        # HERE WE GO !!!!!
        total_mem_sparse = 0
        self.sub_sparse_sorted_data = []
        self.sub_sparse_sorted_ind = []
        self.nnz = self.sparse_matrix.data.shape[0]
        for i in range(0, self.sub_sampled_sparse_matrix.shape[0]):
            s0 = self.sub_sampled_sparse_matrix.indptr[i]
            s1 = self.sub_sampled_sparse_matrix.indptr[i + 1]
            loc_ind = np.copy(self.sub_sampled_sparse_matrix.indices[s0:s1])
            loc_data = np.copy(self.sub_sampled_sparse_matrix.data[s0:s1])
            id_sort = np.argsort(loc_data)
            data_2_push = list(loc_data[id_sort])
            data_2_push.reverse()
            ind_2_push = list(loc_ind[id_sort])
            ind_2_push.reverse()

            self.sub_sparse_sorted_data.extend(data_2_push)
            self.sub_sparse_sorted_ind.extend(ind_2_push)

        self.sparse_matrix_coo = self.sparse_matrix.tocoo()
        self.mat_without_repeats_coo_tmp = self.mat_without_repeats.tocoo()
        self.mat_without_repeats_coo = scp.sparse.triu(
            self.mat_without_repeats_coo_tmp, k=1, format="coo"
        )

        if self.hay_repeats:
            self.mat_with_repeats_coo_tmp = self.mat_with_repeats.tocoo()
            self.mat_with_repeats_coo = scp.sparse.triu(
                self.mat_with_repeats_coo_tmp, k=1, format="coo"
            )
        # self.gpu_sp_data = ga.to_gpu(ary=self.sparse_matrix.data)
        # self.gpu_sp_indptr = ga.to_gpu(ary=self.sparse_matrix.indptr)
        # self.gpu_sp_rows = ga.to_gpu(ary=self.sparse_matrix_coo.row)
        # self.gpu_sp_cols = ga.to_gpu(ary=self.sparse_matrix_coo.col)

        self.gpu_id_single = ga.to_gpu(ary=self.id_single)
        self.gpu_sp_no_rep_data = ga.to_gpu(
            ary=self.mat_without_repeats_coo.data
        )
        # self.gpu_sp_no_rep_indptr =
        # ga.to_gpu(ary=self.mat_without_repeats.indptr)
        self.gpu_sp_no_rep_rows = ga.to_gpu(
            ary=self.mat_without_repeats_coo.row
        )
        self.gpu_sp_no_rep_cols = ga.to_gpu(
            ary=self.mat_without_repeats_coo.col
        )

        self.gpu_sub_sp_no_rep_data = ga.to_gpu(
            ary=np.zeros_like(self.mat_without_repeats_coo.data)
        )
        self.gpu_sub_sp_no_rep_rows = ga.to_gpu(
            ary=np.zeros_like(self.mat_without_repeats_coo.data)
        )
        self.gpu_sub_sp_no_rep_cols = ga.to_gpu(
            ary=np.zeros_like(self.mat_without_repeats_coo.data)
        )

        self.n_non_zero = self.mat_without_repeats_coo.data.shape[0]
        self.gpu_collect_frags_4_sp = ga.to_gpu(
            ary=np.zeros((self.id_single.shape[0] + 1,), dtype=np.int32)
        )

        if self.hay_repeats:
            self.gpu_id_ok_rep = ga.to_gpu(ary=self.id_rep)
            self.gpu_sp_rep_data = ga.to_gpu(ary=self.mat_with_repeats.data)
            self.gpu_sp_rep_indptr = ga.to_gpu(
                ary=self.mat_with_repeats.indptr
            )
            self.gpu_sp_rep_rows = ga.to_gpu(ary=self.mat_with_repeats_coo.row)
            self.gpu_sp_rep_cols = ga.to_gpu(ary=self.mat_with_repeats_coo.col)

        # self.gpu_sp_n_indices = np.int32(self.sparse_matrix.indices.shape[0])

        total_mem_sparse = (
            self.sparse_matrix.data.nbytes
            + self.sparse_matrix.indptr.nbytes
            + self.sparse_matrix_coo.row.nbytes
            + self.sparse_matrix_coo.col.nbytes
        )

        # self.gpu_sub_sp_data =
        # ga.to_gpu(ary=self.sub_sampled_sparse_matrix.data)
        # self.gpu_sub_sp_indptr =
        # ga.to_gpu(ary=self.sub_sampled_sparse_matrix.indptr)
        # self.gpu_sub_sp_indices =
        # ga.to_gpu(ary=self.sub_sampled_sparse_matrix.indices)
        # self.gpu_sub_sp_n_indices =
        # np.int32(self.sub_sampled_sparse_matrix.indices.shape[0])

        # total_mem_sparse += self.sub_sampled_sparse_matrix.data.nbytes +
        # self.sub_sampled_sparse_matrix.indptr.nbytes + \
        #                     self.sub_sampled_sparse_matrix.indices.nbytes
        logger.info(
            "total mem used by sparse data = {}".format(
                np.float32(total_mem_sparse) / 10 ** 6.
            )
        )

    def sparse_data_4_gl(self, precision):
        # create sparse data 4 opengl purposes. take only value above limit
        # self.sub_mat = self.sub_sampled_sparse_matrix.tocoo()
        self.sub_mat = scp.sparse.triu(
            self.sub_sampled_sparse_matrix, k=1, format="coo"
        )
        rows = self.sub_mat.row
        cols = self.sub_mat.col
        data = self.sub_mat.data
        id = np.nonzero(data > precision)[0]
        self.n_data_4_gl = len(id)
        self.mat_4_gl = scp.sparse.coo_matrix(
            (data[id], (rows[id], cols[id])), shape=self.sub_mat.shape
        )
        self.gpu_rows_4_gl = ga.to_gpu(ary=self.mat_4_gl.row)
        self.gpu_cols_4_gl = ga.to_gpu(ary=self.mat_4_gl.col)
        self.gpu_data_4_gl = ga.to_gpu(ary=self.mat_4_gl.data)
        self.gpu_ptr_4_gl = ga.zeros_like(self.gpu_data_4_gl)
        self.gpu_counter_select_4_gl = ga.zeros((1,), dtype=np.int32)
        self.gpu_vect_gl_pxl_frag = ga.zeros(
            (self.n_new_frags,), dtype=np.int32
        )

    def dist_inter_genome(self, tmp_gpu_vect_frags):
        tmp_gpu_vect_frags.copy_from_gpu()
        g1 = tmp_gpu_vect_frags
        n_frags_blacklisted = len(self.id_frags_blacklisted)
        d = 3.0 * (self.n_new_frags - n_frags_blacklisted)
        norm_distance = 3.0 * (self.n_new_frags - n_frags_blacklisted)
        # d = 3.0 * self.n_new_frags
        for id_f in range(0, self.n_new_frags):
            if id_f not in self.id_frags_blacklisted:
                prev_t0 = self.np_init_prev[id_f]
                prev_t1 = g1.prev[id_f]
                next_t0 = self.np_init_next[id_f]
                next_t1 = g1.next[id_f]
                ori_t0 = self.np_init_ori[id_f]
                ori_t1 = g1.ori[id_f]
                swap = 1
                if ((prev_t1 == prev_t0) and (next_t1 == next_t0)) or (
                    (prev_t1 == next_t0) and (next_t1 == prev_t0)
                ):
                    d -= 1
                    # if not(self.np_init_orientable[id_f]):
                    #     d -= 2
                if self.np_init_orientable[id_f]:
                    if ori_t0 != ori_t1:
                        tmp = prev_t1
                        prev_t1 = next_t1
                        next_t1 = tmp
                        swap = -1
                    if prev_t0 == prev_t1:
                        if prev_t0 == -1:
                            d -= 1
                        elif not (self.np_init_orientable[prev_t1]):
                            d -= 1
                        else:
                            d -= 0.5
                            ori_prev_t0 = self.np_init_ori[prev_t0]
                            ori_prev_t1 = g1.ori[prev_t1]
                            if ori_prev_t0 == swap * ori_prev_t1:
                                d -= 0.5
                    if next_t0 == next_t1:
                        if next_t0 == -1:
                            d -= 1
                        elif not (self.np_init_orientable[next_t1]):
                            d -= 1
                        else:
                            d -= 0.5
                            ori_next_t0 = self.np_init_ori[next_t0]
                            ori_next_t1 = g1.ori[next_t1]
                            if ori_next_t0 == swap * ori_next_t1:
                                d -= 0.5
                else:
                    if (prev_t1 == prev_t0) or (prev_t1 == next_t0):
                        d -= 1
                    if (next_t1 == next_t0) or (next_t1 == prev_t0):
                        d -= 1
        return d / norm_distance

    def approx_single_likelihood_on_zeros(self,):

        start = cuda.Event()
        end = cuda.Event()
        stride = 1

        size_block = 1024
        block_ = (size_block, 1, 1)
        n_blocks = int(int(self.init_n_sub_frags) / size_block + 1)
        grid_ = (max(1, int(n_blocks // stride)), 1)
        # print "block = ", block_
        # print "grid_all = ", grid_
        self.gpu_likelihood_on_zeros.fill(0)
        self.gpu_n_vals_intra.fill(0)
        self.ctx.synchronize()

        start.record()
        self.kern_eval_likelihood_zeros(
            self.gpu_vect_id_c,
            self.gpu_vect_s_tot,
            self.gpu_vect_pos,
            self.gpu_vect_len,
            self.gpu_param_simu,
            np.int32(self.mean_len_bp_frags / 1000.),
            self.gpu_likelihood_on_zeros,
            self.gpu_n_vals_intra,
            np.int32(self.init_n_sub_frags),
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()

        # print "CUDA clock execution timing (compute likelihood on zeros): ",
        # secs
        self.val_on_zero_intra = (
            self.gpu_likelihood_on_zeros.get()[0] * self.log_e
        )
        self.n_vals_intra = self.gpu_n_vals_intra.get()[0]

        self.val_on_zero_inter = (
            self.log_e
            * (self.n_pixl_sub_mat - self.n_vals_intra)
            * -1.0
            * self.param_simu["v_inter"][0]
        )
        self.curr_likelihood_on_z = (
            self.val_on_zero_intra + self.val_on_zero_inter
        )
        # print "GPU execution time ( approx on zeros single) = ", t1 - t0

    def approx_single_likelihood_on_zeros_nuisance(self,):

        start = cuda.Event()
        end = cuda.Event()
        stride = 1

        size_block = 1024
        block_ = (size_block, 1, 1)
        n_blocks = int(int(self.init_n_sub_frags) / size_block + 1)
        grid_ = int(max(1, n_blocks / stride)), 1
        # print "block = ", block_
        # print "grid_all = ", grid_
        self.gpu_likelihood_on_zeros_nuis.fill(0)
        self.gpu_n_vals_intra.fill(0)
        start.record()
        self.kern_eval_likelihood_zeros(
            self.gpu_vect_id_c,
            self.gpu_vect_s_tot,
            self.gpu_vect_pos,
            self.gpu_vect_len,
            self.gpu_param_simu_test,
            np.float32(self.mean_len_bp_frags / 1000.),
            self.gpu_likelihood_on_zeros_nuis,
            self.gpu_n_vals_intra,
            np.int32(self.init_n_sub_frags),
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()
        # print "CUDA clock execution timing (compute likelihood on zeros): ",
        # secs
        self.val_on_zero_intra_nuis = (
            self.gpu_likelihood_on_zeros_nuis.get()[0] * self.log_e
        )
        self.n_vals_intra = self.gpu_n_vals_intra.get()[0]
        self.val_on_zero_inter_nuis = (
            self.log_e
            * (self.n_pixl_sub_mat - self.n_vals_intra)
            * -1.0
            * self.param_simu_test["v_inter"][0]
        )
        self.curr_likelihood_on_z_nuis = (
            self.val_on_zero_intra_nuis + self.val_on_zero_inter_nuis
        )
        # print "GPU execution time ( approx on zeros single) = ", t1 - t0

    def approx_single_likelihood_on_zeros_mut(self,):

        start = cuda.Event()
        end = cuda.Event()
        stride = 1

        size_block = 1024
        block_ = (size_block, 1, 1)
        n_blocks = int(self.init_n_sub_frags) / size_block + 1
        grid_ = (max(1, n_blocks / stride), 1)
        # print "block = ", block_
        # print "grid_all = ", grid_
        self.gpu_likelihood_on_zeros.fill(0)
        self.gpu_n_vals_intra.fill(0)
        self.ctx.synchronize()

        start.record()
        self.kern_eval_likelihood_zeros(
            self.gpu_vect_id_c_mut,
            self.gpu_vect_s_tot_mut,
            self.gpu_vect_pos_mut,
            self.gpu_vect_len_mut,
            self.gpu_param_simu,
            np.float32(self.mean_len_bp_frags / 1000.),
            self.gpu_likelihood_on_zeros,
            self.gpu_n_vals_intra,
            np.int32(self.init_n_sub_frags),
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()

        # print "CUDA clock execution timing (compute likelihood on zeros): ",
        # secs
        self.val_on_zero_intra_mut = (
            self.gpu_likelihood_on_zeros.get()[0] * self.log_e
        )
        self.n_vals_intra = self.gpu_n_vals_intra.get()[0]
        self.val_on_zero_inter_mut = (
            self.log_e
            * (self.n_pixl_sub_mat - self.n_vals_intra)
            * -1.0
            * self.param_simu["v_inter"][0]
        )
        self.curr_likelihood_on_z_mut = (
            self.val_on_zero_intra_mut + self.val_on_zero_inter_mut
        )
        # self.curr_likelihood_on_z_mut = self.val_on_zero_inter_mut
        # print "GPU execution time ( approx on zeros single) = ", t1 - t0

    def approx_all_likelihood_on_zeros(self,):

        start = cuda.Event()
        end = cuda.Event()
        stride = 1

        size_block = 1024
        block_ = (size_block, 1, 1)
        n_blocks = int(int(self.init_n_sub_frags) / size_block + 1)
        grid_ = (int(max(1, n_blocks / stride)), 1)

        self.gpu_vect_likelihood_z.fill(0)
        self.gpu_all_n_vals_intra.fill(0)
        self.ctx.synchronize()

        start.record()
        self.kern_eval_all_likelihood_zeros_1st(
            self.collect_gpu_vect_id_c,
            self.collect_gpu_vect_s_tot,
            self.collect_gpu_vect_pos,
            self.collect_gpu_vect_len,
            self.gpu_param_simu,
            np.float32(self.mean_len_bp_frags / 1000.),
            self.gpu_list_uniq_mutations,
            self.gpu_n_uniq,
            self.gpu_vect_likelihood_z,
            self.gpu_all_n_vals_intra,
            np.int32(self.init_n_sub_frags),
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()

        start.record()
        self.kern_eval_all_likelihood_zeros_2nd(
            self.gpu_list_uniq_mutations,
            self.gpu_n_uniq,
            self.gpu_param_simu,
            self.gpu_vect_likelihood_z,
            self.gpu_all_n_vals_intra,
            self.gpu_n_pixl_sub_mat,
            block=(self.n_threads_mutations, 1, 1),
            grid=(1, 1),
        )
        end.record()
        end.synchronize()

    def fill_dist_all_mut(self,):
        start = cuda.Event()
        end = cuda.Event()
        size_block = 1024
        block_ = (size_block, 1, 1)
        size_all_data = int(self.init_n_sub_frags)
        n_block = size_all_data // (size_block) + 1
        size_grid = max(int(n_block), 1)
        grid_all = (size_grid, 1)

        for id_mut in range(0, self.n_tmp_struct):
            start.record()
            self.kern_fill_vect_dist_all(
                self.gpu_sub_frags_2_frags,
                self.collector_gpu_vect_frags[id_mut].get_ptr(),
                self.collect_gpu_vect_dist,
                self.collect_gpu_vect_id_c,
                self.collect_gpu_vect_s_tot,
                self.collect_gpu_vect_pos,
                self.collect_gpu_vect_len,
                self.gpu_collector_id_repeats,
                self.gpu_frag_dispatcher,
                self.gpu_sub_collector_id_repeats,
                self.gpu_sub_frag_dispatcher,
                np.int32(self.init_n_sub_frags),
                np.int32(id_mut),
                block=block_,
                grid=grid_all,
            )

            end.record()
            end.synchronize()
            # print "CUDA clock execution timing (compute sub vect distances):
            # ", secs print "Total time vect frags to sub frags = ",
            # time.time() - t0

    def fill_dist_single(self,):

        start = cuda.Event()
        end = cuda.Event()
        size_block = 1024
        block_ = (size_block, 1, 1)
        size_all_data = int(self.init_n_sub_frags)
        n_block = size_all_data // (size_block) + 1
        size_grid = max(int(n_block / 1), 1)
        grid_all = (size_grid, 1)

        start.record()
        self.kern_fill_vect_dist_single(
            self.gpu_sub_frags_2_frags,
            self.gpu_vect_frags.get_ptr(),
            self.gpu_vect_dist,
            self.gpu_vect_id_c,
            self.gpu_vect_s_tot,
            self.gpu_vect_pos,
            self.gpu_vect_len,
            self.gpu_collector_id_repeats,
            self.gpu_frag_dispatcher,
            self.gpu_sub_collector_id_repeats,
            self.gpu_sub_frag_dispatcher,
            np.int32(self.init_n_sub_frags),
            block=block_,
            grid=grid_all,
        )
        end.record()
        end.synchronize()
        # print "CUDA clock execution timing (compute sub vect distances): ",
        # secs
        # print "Total time vect frags to sub frags = ", time.time() - t0

    def fill_dist_single_mut(self, id_mut):

        start = cuda.Event()
        end = cuda.Event()
        size_block = 1024
        block_ = (size_block, 1, 1)
        size_all_data = int(self.init_n_sub_frags)
        n_block = size_all_data // (size_block) + 1
        size_grid = max(int(n_block / 1), 1)
        grid_all = (size_grid, 1)

        if id_mut == -1:
            vect_frags = self.gpu_vect_frags
        else:
            vect_frags = self.collector_gpu_vect_frags[id_mut]
        start.record()
        self.kern_fill_vect_dist_single(
            self.gpu_sub_frags_2_frags,
            vect_frags.get_ptr(),
            self.gpu_vect_dist_mut,
            self.gpu_vect_id_c_mut,
            self.gpu_vect_s_tot_mut,
            self.gpu_vect_pos_mut,
            self.gpu_vect_len_mut,
            self.gpu_collector_id_repeats,
            self.gpu_frag_dispatcher,
            self.gpu_sub_collector_id_repeats,
            self.gpu_sub_frag_dispatcher,
            np.int32(self.init_n_sub_frags),
            block=block_,
            grid=grid_all,
        )
        end.record()
        end.synchronize()
        # print "CUDA clock execution timing (compute sub vect distances): ",
        # secs print "Total time vect frags to sub frags = ", time.time() - t0

    def slice_sparse_mat(self, id_ctg1, id_ctg2, id_fragA, id_fragB):

        size_block = self.size_block_4_sub
        block_ = (size_block, 1, 1)
        start = cuda.Event()
        end = cuda.Event()

        grid_ = (int(self.n_non_zero // size_block + 1), 1)
        # print "grid = ", grid_
        self.gpu_counter_select.fill(0)
        self.ctx.synchronize()
        start.record()
        self.kern_slice_sp_mat(
            self.gpu_sp_no_rep_data,
            self.gpu_sp_no_rep_rows,
            self.gpu_sp_no_rep_cols,
            self.gpu_vect_frags.get_ptr(),
            self.gpu_vect_id_c,
            self.gpu_vect_pos,
            self.gpu_sub_sp_no_rep_rows,
            self.gpu_sub_sp_no_rep_cols,
            self.gpu_sub_sp_no_rep_data,
            np.int32(id_ctg1),
            np.int32(id_ctg2),
            np.int32(id_fragA),
            np.int32(id_fragB),
            np.int32(self.max_bounds_insert),
            self.gpu_counter_select,
            np.int32(self.n_non_zero),
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()
        self.n_sub_vals = self.gpu_counter_select.get()[0]

        self.thrust_module.sort_by_keys_zip(
            self.gpu_sub_sp_no_rep_rows,
            int(self.n_sub_vals),
            self.gpu_sub_sp_no_rep_cols,
            self.gpu_sub_sp_no_rep_data,
        )

        n_blocks = int(self.n_sub_vals // self.size_block_4_sub + 1)
        size_grid = max(int(n_blocks), 1)
        grid_all = (size_grid, 1)
        self.gpu_counter_select.fill(0)
        self.ctx.synchronize()

        start.record()
        self.kern_prepare_sparse_call(
            self.gpu_sub_sp_no_rep_rows,
            self.gpu_info_blocks,
            self.gpu_sub_sp_block_indptr,
            self.gpu_counter_select,
            np.int32(self.n_sub_vals),
            block=block_,
            grid=grid_all,
        )
        end.record()
        end.synchronize()

        # print "GPU all splice: elapsed time  = ", t_end - t_start print "GPU
        # Thrust sort: elapsed time  = ", t_thrust_1 - t_thrust_0 print "GPU
        # splice init : elapsed time = ", elapsed_seconds_splice print "GPU
        # splice prepare call : elapsed time = ", elapsed_seconds_prepare

    def show_sub_slice(self, id_ctg1, id_ctg2, id_frag_a, id_frag_b):
        self.gpu_sub_sp_no_rep_data.fill(0)
        self.gpu_sub_sp_no_rep_rows.fill(0)
        self.gpu_sub_sp_no_rep_cols.fill(0)
        self.slice_sparse_mat(id_ctg1, id_ctg2, id_frag_a, id_frag_b)
        dat = self.gpu_sub_sp_no_rep_data.get()[: self.n_sub_vals]
        rows = self.gpu_sub_sp_no_rep_rows.get()[: self.n_sub_vals]
        cols = self.gpu_sub_sp_no_rep_cols.get()[: self.n_sub_vals]

        self.np_sub_mat = scp.sparse.coo_matrix(
            (dat, (rows, cols)),
            shape=(self.init_n_sub_frags, self.init_n_sub_frags),
        )
        plt.spy(self.np_sub_mat, markersize=0.1)
        plt.show()

    def eval_all_sub_likelihood(self,):

        self.fill_dist_all_mut()

        self.approx_all_likelihood_on_zeros()

        block_ = (self.size_block_4_sub, 1, 1)

        start = cuda.Event()
        end = cuda.Event()

        n_blocks = int(self.n_sub_vals // self.size_block_4_sub + 1)
        size_grid = max(int(n_blocks), 1)
        grid_all = (size_grid, 1)
        self.gpu_sub_vect_likelihood_nz.fill(0.0)
        self.gpu_all_scores.fill(0.0)
        self.ctx.synchronize()

        start.record()
        self.kern_sub_likelihood(
            self.gpu_sub_sp_no_rep_data,
            self.gpu_info_blocks,
            self.gpu_sub_sp_block_indptr,
            self.gpu_sub_sp_no_rep_rows,
            self.gpu_sub_sp_no_rep_cols,
            self.gpu_param_simu,
            np.float32(self.mean_len_bp_frags / 1000.0),
            self.collect_gpu_vect_dist,
            self.collect_gpu_vect_id_c,
            self.collect_gpu_vect_s_tot,
            self.collect_gpu_vect_pos,
            self.collect_gpu_vect_len,
            self.gpu_list_uniq_mutations,
            self.gpu_n_uniq,
            self.gpu_sub_vect_likelihood_nz,
            np.int32(self.n_sub_vals),
            np.int32(self.init_n_sub_frags),
            block=block_,
            grid=grid_all,
        )
        end.record()
        end.synchronize()
        # print "CUDA clock execution timing(sub likelihood computing): ", secs
        start.record()
        self.kern_eval_all_scores(
            self.gpu_list_uniq_mutations,
            self.gpu_n_uniq,
            self.gpu_vect_likelihood_z,
            self.gpu_sub_vect_likelihood_nz,
            self.gpu_curr_likelihood_nz_extract,
            self.gpu_curr_likelihood_nz,
            self.gpu_all_scores,
            block=(32, 1, 1),
            grid=(1, 1),
        )
        end.record()
        end.synchronize()

        score = self.gpu_all_scores.get()

        return score

    def extract_current_sub_likelihood(self):

        block_ = (self.size_block_4_sub, 1, 1)

        start = cuda.Event()
        end = cuda.Event()
        n_blocks = int(self.n_sub_vals // self.size_block_4_sub + 1)
        size_grid = max(int(n_blocks), 1)
        grid_all = (size_grid, 1)
        self.gpu_curr_likelihood_nz_extract.fill(0.0)
        self.ctx.synchronize()
        start.record()
        self.kern_extract_sub_likelihood(
            self.gpu_sub_sp_no_rep_data,
            self.gpu_info_blocks,
            self.gpu_sub_sp_block_indptr,
            self.gpu_sub_sp_no_rep_rows,
            self.gpu_sub_sp_no_rep_cols,
            self.gpu_param_simu,
            np.float32(self.mean_len_bp_frags / 1000.0),
            self.gpu_vect_dist,
            self.gpu_vect_id_c,
            self.gpu_vect_s_tot,
            self.gpu_vect_pos,
            self.gpu_vect_len,
            self.gpu_curr_likelihood_nz_extract,
            np.int32(self.n_sub_vals),
            np.int32(self.init_n_sub_frags),
            block=block_,
            grid=grid_all,
        )
        end.record()
        end.synchronize()
        # print "CUDA clock execution timing(extract sub likelihood computing):
        # ", secs
        self.likelihood_extracted_nz = (
            self.gpu_curr_likelihood_nz_extract.get()
        )

    def eval_likelihood_init(self,):

        start = cuda.Event()
        end = cuda.Event()

        self.fill_dist_single()

        pct_size = 1
        stride = 1
        size_block = 1024
        block_ = (size_block, 1, 1)
        # pct_size = 1
        size_all_data = int(self.gpu_sp_no_rep_data.shape[0] * pct_size)
        # print "size all data = ", size_all_data
        n_block = size_all_data // (size_block) + 1
        size_grid = max(int(n_block / stride), 1)
        grid_all = (size_grid, 1)
        # print "block = ", block_
        # print "grid_all = ", grid_all

        self.approx_single_likelihood_on_zeros()
        self.gpu_curr_likelihood_nz.fill(0.0)
        self.ctx.synchronize()

        start.record()
        self.kern_evaluate_likelihood_single(
            self.gpu_sp_no_rep_data,
            self.gpu_sp_no_rep_rows,
            self.gpu_sp_no_rep_cols,
            self.gpu_id_single,
            self.gpu_param_simu,
            np.float32(self.mean_len_bp_frags / 1000.0),
            self.gpu_vect_dist,
            self.gpu_vect_id_c,
            self.gpu_vect_s_tot,
            self.gpu_vect_pos,
            self.gpu_vect_len,
            self.gpu_curr_likelihood_nz,
            np.int32(size_all_data),
            self.n_single_frags,
            block=block_,
            grid=grid_all,
        )
        end.record()
        end.synchronize()

        likelihood_on_nz = self.gpu_curr_likelihood_nz.get()
        self.curr_likelihood_on_nz = likelihood_on_nz
        self.likelihood_t = likelihood_on_nz + self.curr_likelihood_on_z

    def eval_likelihood(self,):

        start = cuda.Event()
        end = cuda.Event()

        self.fill_dist_single()

        pct_size = 1
        stride = 1
        size_block = 1024
        block_ = (size_block, 1, 1)
        # pct_size = 1
        size_all_data = int(self.gpu_sp_no_rep_data.shape[0] * pct_size)
        # print "size all data = ", size_all_data
        n_block = size_all_data // (size_block) + 1
        size_grid = max(int(n_block / stride), 1)
        grid_all = (size_grid, 1)
        # print "block = ", block_
        # print "grid_all = ", grid_all

        self.approx_single_likelihood_on_zeros()

        self.gpu_curr_likelihood_nz.fill(0.0)
        self.ctx.synchronize()

        start.record()
        self.kern_evaluate_likelihood_single(
            self.gpu_sp_no_rep_data,
            self.gpu_sp_no_rep_rows,
            self.gpu_sp_no_rep_cols,
            self.gpu_id_single,
            self.gpu_param_simu,
            np.float32(self.mean_len_bp_frags / 1000.0),
            self.gpu_vect_dist,
            self.gpu_vect_id_c,
            self.gpu_vect_s_tot,
            self.gpu_vect_pos,
            self.gpu_vect_len,
            self.gpu_curr_likelihood_nz,
            np.int32(size_all_data),
            self.n_single_frags,
            block=block_,
            grid=grid_all,
        )
        end.record()
        end.synchronize()
        # likelihood_on_nz = self.gpu_curr_likelihood_nz.get()
        # self.curr_likelihood_on_nz = likelihood_on_nz
        # self.curr_likelihood = likelihood_on_nz + self.curr_likelihood_on_z
        # print "GPU time likelihood ALL = ", time.time() - t1
        # print "likelihood on nz= ", likelihood_on_nz
        # return self.curr_likelihood_on_nz, self.curr_likelihood_on_z

    def eval_likelihood_4_nuisance(self,):
        start = cuda.Event()
        end = cuda.Event()

        pct_size = 1
        stride = 1
        size_block = 1024
        block_ = (size_block, 1, 1)
        # pct_size = 1
        size_all_data = int(self.gpu_sp_no_rep_data.shape[0] * pct_size)
        # print "size all data = ", size_all_data
        n_block = size_all_data // (size_block) + 1
        size_grid = max(int(n_block / stride), 1)
        grid_all = (size_grid, 1)
        # print "block = ", block_
        # print "grid_all = ", grid_all

        self.approx_single_likelihood_on_zeros_nuisance()

        self.gpu_curr_likelihood_nz_nuis.fill(0.0)
        start.record()
        self.kern_evaluate_likelihood_single(
            self.gpu_sp_no_rep_data,
            self.gpu_sp_no_rep_rows,
            self.gpu_sp_no_rep_cols,
            self.gpu_id_single,
            self.gpu_param_simu_test,
            np.float32(self.mean_len_bp_frags / 1000.0),
            self.gpu_vect_dist,
            self.gpu_vect_id_c,
            self.gpu_vect_s_tot,
            self.gpu_vect_pos,
            self.gpu_vect_len,
            self.gpu_curr_likelihood_nz_nuis,
            np.int32(size_all_data),
            self.n_single_frags,
            block=block_,
            grid=grid_all,
        )
        end.record()
        end.synchronize()
        likelihood_on_nz_nuis = self.gpu_curr_likelihood_nz_nuis.get()
        # self.curr_likelihood_on_nz = likelihood_on_nz
        self.curr_likelihood_nuis = (
            likelihood_on_nz_nuis + self.curr_likelihood_on_z_nuis
        )
        # print "GPU time likelihood ALL = ", time.time() - t1
        # print "likelihood on nz= ", likelihood_on_nz
        return self.curr_likelihood_nuis

    def eval_likelihood_on_mut(self, id_mut):

        start = cuda.Event()
        end = cuda.Event()
        self.fill_dist_single_mut(id_mut)
        self.approx_single_likelihood_on_zeros_mut()

        pct_size = 1
        stride = 1
        size_block = 1024
        block_ = (size_block, 1, 1)
        # pct_size = 1
        size_all_data = int(self.gpu_sp_no_rep_data.shape[0] * pct_size)
        # print "size all data = ", size_all_data
        n_block = size_all_data // (size_block) + 1
        size_grid = max(int(n_block / stride), 1)
        grid_all = (size_grid, 1)
        # print "block = ", block_
        # print "grid_all = ", grid_all

        self.gpu_val_likelihood_4_mut.fill(0.0)
        self.ctx.synchronize()

        start.record()
        self.kern_evaluate_likelihood_single(
            self.gpu_sp_no_rep_data,
            self.gpu_sp_no_rep_rows,
            self.gpu_sp_no_rep_cols,
            self.gpu_id_single,
            self.gpu_param_simu,
            np.float32(self.mean_len_bp_frags / 1000.0),
            self.gpu_vect_dist_mut,
            self.gpu_vect_id_c_mut,
            self.gpu_vect_s_tot_mut,
            self.gpu_vect_pos_mut,
            self.gpu_vect_len_mut,
            self.gpu_val_likelihood_4_mut,
            np.int32(size_all_data),
            self.n_single_frags,
            block=block_,
            grid=grid_all,
        )
        end.record()
        end.synchronize()

        # likelihood_on_nz = ga.sum(self.gpu_val_likelihood_4_mut,
        # dtype=np.float64).get()
        likelihood_on_nz = self.gpu_val_likelihood_4_mut.get()
        self.curr_likelihood_on_nz_mut = likelihood_on_nz
        # self.curr_likelihood_on_nz_mut = likelihood_on_nz

        # print "GPU time likelihood ALL = ", time.time() - t1
        # print "likelihood on nz= ", likelihood_on_nz
        return self.curr_likelihood_on_nz_mut + self.curr_likelihood_on_z_mut

    def step_sampler(self, id_frag, n_neighbours, dt):

        self.candidates = self.return_neighbours(id_frag, n_neighbours)
        self.candidates.sort()
        n = len(self.candidates)

        self.fill_dist_single()

        self.eval_likelihood()
        self.gpu_vect_frags.copy_from_gpu()
        id_c_prev_cand = -1
        id_ctg_a = self.gpu_vect_frags.id_c[id_frag]
        # print "id_contig a = ", id_ctg_a
        self.all_scores = np.zeros((self.n_tmp_struct * n), dtype=np.float64)
        max_id = self.modify_gl_cuda_buffer(id_frag, dt)
        flip_eject = 1
        for (id_cand, i) in zip(self.candidates, list(range(0, n))):
            self.gl_window.remote_update()
            self.extract_uniq_mutations(id_frag, id_cand, flip_eject)
            self.perform_mutations(id_frag, id_cand, max_id, 1 == 0)
            id_ctg_b = self.gpu_vect_frags.id_c[id_cand]
            if id_ctg_b != id_c_prev_cand:
                # print "id_contig b = ", id_ctg_b
                # print "slicing sparse matrix!"
                self.slice_sparse_mat(id_ctg_a, id_ctg_b, id_frag, id_cand)
                self.extract_current_sub_likelihood()
            else:
                self.slice_sparse_mat(id_ctg_a, id_ctg_b, id_frag, id_cand)
                self.extract_current_sub_likelihood()
            id_c_prev_cand = id_ctg_b
            self.all_scores[
                i * self.n_tmp_struct : (i + 1) * self.n_tmp_struct
            ] = self.eval_all_sub_likelihood()
            flip_eject = 0
        # print "done!"

        scores_ok = np.copy(self.all_scores)
        scores_ok[scores_ok == 0] = -np.inf
        max_score = scores_ok.max()
        thresh_overflow = 30
        # filtered_score[filtered_score < max_score - thresh] = 0
        filtered_score = scores_ok - (max_score - thresh_overflow)
        filtered_score[filtered_score < 0] = 0

        global_id = np.argmax(filtered_score)
        # print "global id = ", global_id
        id_f_sampled = self.candidates[int(global_id / self.n_tmp_struct)]
        op_sampled = global_id % self.n_tmp_struct

        # print "id frag sampled = ", id_f_sampled
        # print "operation sampled = ", self.modification_str[op_sampled]
        # print 'id operation =', op_sampled

        self.test_copy_struct(id_frag, id_f_sampled, op_sampled, max_id)
        self.modify_gl_cuda_buffer(id_frag, self.gl_window.dt)
        o = self.all_scores[global_id]
        self.o = o
        dist = self.dist_inter_genome(self.gpu_vect_frags)
        self.likelihood_t = o
        return (
            o,
            dist,
            op_sampled,
            id_f_sampled,
            self.mean_length_contigs,
            self.n_contigs,
        )

    def step_sampler_debug(self, id_frag, n_neighbours):

        # n_neighbours = 5000

        self.candidates = list(range(0, n_neighbours + 1))
        self.candidates.pop(id_frag)
        n = len(self.candidates)

        self.fill_dist_single()
        # curr_likelihood_on_nz, curr_likelihood_on_z = self.eval_likelihood()
        self.eval_likelihood()
        self.gpu_vect_frags.copy_from_gpu()
        id_c_prev_cand = -1
        id_ctg_a = self.gpu_vect_frags.id_c[id_frag]
        # print "id_contig a = ", id_ctg_a
        self.all_scores = np.zeros((self.n_tmp_struct * n), dtype=np.float64)
        max_id = self.modify_gl_cuda_buffer(id_frag, self.gl_window.dt)
        flip_eject = 1
        for (id_cand, i) in zip(self.candidates, list(range(0, n))):
            self.gl_window.remote_update()
            self.extract_uniq_mutations(id_frag, id_cand, flip_eject)
            self.perform_mutations(id_frag, id_cand, max_id, 1 == 0)
            id_ctg_b = self.gpu_vect_frags.id_c[id_cand]
            if id_ctg_b != id_c_prev_cand:
                # print "id_contig b = ", id_ctg_b
                # print "slicing sparse matrix!"
                self.slice_sparse_mat(id_ctg_a, id_ctg_b, id_frag, id_cand)
                self.extract_current_sub_likelihood()
            id_c_prev_cand = id_ctg_b
            self.all_scores[
                i * self.n_tmp_struct : (i + 1) * self.n_tmp_struct
            ] = self.eval_all_sub_likelihood()
            flip_eject = 0

    def extract_uniq_mutations(self, id_fi, id_fj, flip_eject):
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        self.gpu_list_uniq_mutations.fill(0)
        self.ctx.synchronize()

        self.kern_uniq_mutations(
            self.gpu_vect_frags.get_ptr(),
            np.int32(id_fi),
            np.int32(id_fj),
            self.gpu_list_uniq_mutations,
            self.gpu_list_valid_insert,
            self.gpu_n_uniq,
            np.int32(flip_eject),
            block=(self.n_threads_mutations, 1, 1),
            grid=(1, 1),
        )
        end.record()
        end.synchronize()
        # print "CUDA clock execution timing(define uniq mutations): ", secs

    def loadProgram(self, filename):
        # read in the Cuda source file as a string
        f = open(filename, "r")
        raw_fstr = "".join(f.readlines())
        f.close()
        if self.active_insert_blocks:
            logger.info(
                "size array in shared memory = {}".format(
                    str(self.n_tmp_struct * self.size_block_4_sub)
                )
            )
            fstr = (
                raw_fstr.replace(
                    "N_STRUCT_BY_BLOCK_SIZE",
                    str(self.n_tmp_struct * self.size_block_4_sub),
                )
                .replace("SIZE_BLOCK_4_SUB_3", str(self.size_block_4_sub * 3))
                .replace("N_TMP_STRUCT", str(self.n_tmp_struct))
                .replace("SIZE_BLOCK_4_SUB", str(self.size_block_4_sub))
                .replace("N_TO_CUT", str(self.n_insert_blocks))
            )

        else:
            fstr = raw_fstr

        # create the program
        self.module = pycuda.compiler.SourceModule(
            fstr,
            no_extern_c=True,
            options=[
                # options=["--cubin", "-dc=true", "-lcudadevrt", "-m64",
                #          "--keep",
                # options=["--cubin","-lcudadevrt", "-m64",
            ],
            # options=["-lcudadevrt", "-m64"],
        )

        # self.sub_evaluate_likelihood_sparse =
        # self.module.get_function('sub_evaluate_likelihood_sparse')

        self.kern_fill_vect_dist_all = self.module.get_function(
            "fill_vect_dist"
        )
        self.kern_fill_vect_dist_single = self.module.get_function(
            "uni_fill_vect_dist"
        )
        self.init_rng = self.module.get_function("init_rng")
        self.kern_evaluate_likelihood_single = self.module.get_function(
            "evaluate_likelihood_sparse"
        )
        self.kern_sub_likelihood = self.module.get_function(
            "eval_sub_likelihood"
        )
        # self.kern_select_them = self.module.get_function('select_them')
        self.kern_slice_sp_mat = self.module.get_function("slice_sp_mat")
        self.kern_prepare_sparse_call = self.module.get_function(
            "prepare_sparse_call"
        )
        self.kern_eval_likelihood_zeros = self.module.get_function(
            "eval_likelihood_on_zero"
        )
        self.kern_eval_all_likelihood_zeros_1st = self.module.get_function(
            "eval_all_likelihood_on_zero_1st"
        )
        self.kern_eval_all_likelihood_zeros_2nd = self.module.get_function(
            "eval_all_likelihood_on_zero_2nd"
        )
        self.kern_extract_sub_likelihood = self.module.get_function(
            "extract_sub_likelihood"
        )
        self.kern_uniq_mutations = self.module.get_function(
            "extract_uniq_mutations"
        )
        self.kern_eval_all_scores = self.module.get_function("eval_all_scores")
        self.kern_select_uniq_id_c = self.module.get_function(
            "select_uniq_id_c"
        )
        self.kern_make_old_2_new_id_c = self.module.get_function(
            "make_old_2_new_id_c"
        )
        self.kern_count_vals = self.module.get_function("count_num")
        self.kern_update_gpu_vect = self.module.get_function(
            "update_gpu_vect_frags"
        )
        self.kern_explode_genome = self.module.get_function("explode_genome")
        if self.active_insert_blocks:
            self.kern_get_bounds = self.module.get_function("get_bounds")
            self.kern_extract_block = self.module.get_function("extract_block")
            self.kern_insert_block = self.module.get_function("insert_block")
        self.kern_frags_2_gl_pxl = self.module.get_function("gpu_struct_2_pxl")
        self.kern_update_matrix = self.module.get_function("update_matrix")
        self.kern_update_gl_buffer = self.module.get_function(
            "update_gl_buffer"
        )
        self.kern_prepare_sparse_call_4_gl = self.module.get_function(
            "prepare_sparse_call_4_gl"
        )

        self.set_null = self.module.get_function("set_null")
        self.copy_gpu_array = self.module.get_function("copy_gpu_array")
        self.gl_update_pos = self.module.get_function("gl_update_pos")
        self.gpu_transloc = []
        self.pop_out = self.module.get_function("pop_out_frag")
        self.flip_frag = self.module.get_function("flip_frag")
        self.pop_in_1 = self.module.get_function(
            "pop_in_frag_1"
        )  # split insert @ left
        self.pop_in_2 = self.module.get_function(
            "pop_in_frag_2"
        )  # split insert @ right
        self.pop_in_3 = self.module.get_function(
            "pop_in_frag_3"
        )  # insert @ left
        # self.pop_in_4 = self.module.get_function('pop_in_frag_4') # insert @
        # right
        self.split = self.module.get_function("split_contig")
        self.paste = self.module.get_function("paste_contigs")
        self.simple_copy = self.module.get_function("simple_copy")
        self.copy_vect = self.module.get_function("copy_struct")
        self.swap_activity = self.module.get_function("swap_activity_frag")
        self.modification_str = [
            "eject frag",
            "flip frag",
            "pop out split insert @ left or 1",
            "pop out split insert @ left or -1",
            "pop out split insert @ right or 1",
            "pop out split insert @ right or -1",
            "pop out insert @ right or 1",
            "pop out insert @ right or -1",
            # 'swap activity',
            # 'pop out insert @ left or 1', 'pop out insert @ left or -1',
            "transloc_1",
            "transloc_2",
            "transloc_3",
            "transloc_4",
            "local_scramble d1",
            "local_scramble d2",
            "local_scramble d3",
            "local_scramble d4",
        ]

    def update_neighbourhood(self,):
        tmp_sorted = self.hic_matrix_sub_sampled.argsort(axis=1)
        sorted_neighbours = []
        for i in self.list_frag_to_sample:
            all_idx = tmp_sorted[i, :]
            pos = np.nonzero(all_idx == i)[0][0]
            line = list(all_idx)
            line.pop(pos)
            logger.info("filtering neighbourhood of : {}".format(i))
            for j in self.list_to_pop_out:
                line = np.array(line)
                pos = np.nonzero(line == j)[0][0]
                line = list(line)
                line.pop(pos)

            sorted_neighbours.append(line)
        self.sorted_neighbours = np.array(sorted_neighbours)

    def pop_out_pop_in(self, id_f_pop, id_f_ins, mode, max_id):
        size_block = 1024
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags // size_block + 1), 1)
        # max_id = np.int32(ga.max(self.gpu_id_contigs).get())
        # print 'max_id contig before pop out= ', max_id
        start = cuda.Event()
        end = cuda.Event()
        start.record()

        self.pop_out(
            self.pop_gpu_vect_frags.get_ptr(),
            self.gpu_vect_frags.get_ptr(),
            self.pop_gpu_id_contigs,
            np.int32(id_f_pop),
            np.int32(max_id),
            self.n_new_frags,
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()

        max_id2 = np.int32(ga.max(self.pop_gpu_id_contigs).get())
        # print 'max_id contig after pop out= ', max_id2
        start.record()
        # modif_vector = self.collector_gpu_vect_frags[mode]
        or_watson = np.int32(1)
        or_crick = np.int32(-1)
        # print "max_id from pop = ", max_id
        if mode == 0:
            self.simple_copy(
                self.collector_gpu_vect_frags[mode].get_ptr(),
                self.pop_gpu_vect_frags.get_ptr(),
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
        elif mode == 1:
            self.flip_frag(
                self.collector_gpu_vect_frags[mode].get_ptr(),
                self.gpu_vect_frags.get_ptr(),
                np.int32(id_f_pop),
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
        elif mode == 2:
            self.pop_in_1(
                self.collector_gpu_vect_frags[mode].get_ptr(),
                self.pop_gpu_vect_frags.get_ptr(),
                np.int32(id_f_pop),
                np.int32(id_f_ins),
                max_id2,
                or_watson,
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
        elif mode == 3:
            self.pop_in_1(
                self.collector_gpu_vect_frags[mode].get_ptr(),
                self.pop_gpu_vect_frags.get_ptr(),
                np.int32(id_f_pop),
                np.int32(id_f_ins),
                max_id2,
                or_crick,
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
        elif mode == 4:
            self.pop_in_2(
                self.collector_gpu_vect_frags[mode].get_ptr(),
                self.pop_gpu_vect_frags.get_ptr(),
                np.int32(id_f_pop),
                np.int32(id_f_ins),
                max_id2,
                or_watson,
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
        elif mode == 5:
            self.pop_in_2(
                self.collector_gpu_vect_frags[mode].get_ptr(),
                self.pop_gpu_vect_frags.get_ptr(),
                np.int32(id_f_pop),
                np.int32(id_f_ins),
                max_id2,
                or_crick,
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
        elif mode == 6:
            self.pop_in_3(
                self.collector_gpu_vect_frags[mode].get_ptr(),
                self.pop_gpu_vect_frags.get_ptr(),
                np.int32(id_f_pop),
                np.int32(id_f_ins),
                max_id2,
                or_watson,
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
        elif mode == 7:
            self.pop_in_3(
                self.collector_gpu_vect_frags[mode].get_ptr(),
                self.pop_gpu_vect_frags.get_ptr(),
                np.int32(id_f_pop),
                np.int32(id_f_ins),
                max_id2,
                or_crick,
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
        # elif mode == 8:
        # self.pop_in_4(self.collector_gpu_vect_frags[mode].get_ptr(),
        # self.pop_gpu_vect_frags.get_ptr(), np.int32(id_f_pop),
        # np.int32(id_f_ins), max_id2, or_watson, self.n_frags, block=block_,
        # grid=grid_) elif mode == 9:
        # self.pop_in_4(self.collector_gpu_vect_frags[mode].get_ptr(),
        # self.pop_gpu_vect_frags.get_ptr(), np.int32(id_f_pop),
        # np.int32(id_f_ins), max_id2, or_crick, self.n_frags, block=block_,
        # grid=grid_)

        # elif mode == 8:
        # self.swap_activity(self.collector_gpu_vect_frags[mode].get_ptr(),
        # self.pop_gpu_vect_frags.get_ptr(), np.int32(id_f_pop), max_id2,
        # self.n_new_frags, block=block_, grid=grid_)

        end.record()
        end.synchronize()
        # print "CUDA clock execution timing( generate mutations): ", secs

    def transloc(self, id_fA, id_fB, max_id):
        size_block = 128
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags // size_block + 1), 1)
        # max_id = np.int32(ga.max(self.gpu_id_contigs).get())
        start = cuda.Event()
        end = cuda.Event()
        mode = 0
        # id_start_transloc = 8
        id_start_transloc = 8
        for upstreamfA in range(0, 2):
            start.record()
            self.split(
                self.trans1_gpu_vect_frags.get_ptr(),
                self.gpu_vect_frags.get_ptr(),
                self.trans1_gpu_id_contigs,
                np.int32(id_fA),
                np.int32(upstreamfA),
                np.int32(max_id),
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
            end.record()
            end.synchronize()
            for upstreamfB in range(0, 2):
                max_id1 = np.int32(ga.max(self.trans1_gpu_id_contigs).get())
                # print 'max_id1 = ', max_id1
                start.record()
                self.split(
                    self.trans2_gpu_vect_frags.get_ptr(),
                    self.trans1_gpu_vect_frags.get_ptr(),
                    self.trans2_gpu_id_contigs,
                    np.int32(id_fB),
                    np.int32(upstreamfB),
                    max_id1,
                    self.n_new_frags,
                    block=block_,
                    grid=grid_,
                )
                end.record()
                end.synchronize()
                max_id2 = np.int32(ga.max(self.trans2_gpu_id_contigs).get())
                # print 'max_id2 = ', max_id2
                curr_vect_trans = self.collector_gpu_vect_frags[
                    id_start_transloc + mode
                ]
                start.record()
                # self.simple_copy(curr_vect_trans.get_ptr(),
                # self.trans2_gpu_vect_frags.get_ptr(), self.n_frags,
                # block=block_, grid=grid_)
                self.paste(
                    curr_vect_trans.get_ptr(),
                    self.trans2_gpu_vect_frags.get_ptr(),
                    np.int32(id_fA),
                    np.int32(id_fB),
                    max_id2,
                    self.n_new_frags,
                    block=block_,
                    grid=grid_,
                )
                end.record()
                end.synchronize()
                mode += 1

    def insert_blocks(self, id_fA, id_fB, max_id):
        size_block = self.size_block_4_sub
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags // size_block + 1), 1)
        # max_id = np.int32(ga.max(self.gpu_id_contigs).get())
        start = cuda.Event()
        end = cuda.Event()

        id_start_insert = 12

        start.record()
        self.gpu_list_valid_insert.fill(-1)
        self.gpu_list_f_upstream.fill(-1)
        self.gpu_list_f_downstream.fill(-1)
        self.ctx.synchronize()
        self.kern_get_bounds(
            self.gpu_vect_frags.get_ptr(),
            np.int32(id_fA),
            np.int32(id_fB),
            self.gpu_list_valid_insert,
            self.gpu_list_bounds,
            self.gpu_list_f_upstream,
            self.gpu_list_f_downstream,
            np.int32(self.n_insert_blocks),
            self.n_new_frags,
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()
        id = 0
        for i in range(0, self.n_insert_blocks):
            for j in [1, 0]:
                if j == 1:
                    list_bounds = self.gpu_list_f_upstream
                else:
                    list_bounds = self.gpu_list_f_downstream

                start.record()
                self.kern_extract_block(
                    self.trans1_gpu_vect_frags.get_ptr(),
                    self.gpu_vect_frags.get_ptr(),
                    self.trans1_gpu_id_contigs,
                    np.int32(id_fA),
                    list_bounds,
                    np.int32(i),  # id f in list_frag upstream
                    np.int32(j),  # upstream
                    np.int32(max_id),
                    self.n_new_frags,
                    block=block_,
                    grid=grid_,
                )
                end.record()
                end.synchronize()

                start.record()
                self.kern_insert_block(
                    self.collector_gpu_vect_frags[
                        id_start_insert + id
                    ].get_ptr(),
                    self.trans1_gpu_vect_frags.get_ptr(),
                    self.gpu_vect_frags.get_ptr(),
                    np.int32(id_fA),
                    np.int32(id_fB),
                    list_bounds,
                    self.gpu_list_valid_insert,
                    np.int32(id),
                    np.int32(i),
                    np.int32(j),
                    self.n_new_frags,
                    block=block_,
                    grid=grid_,
                )
                end.record()
                end.synchronize()
                id += 1

    def perform_mutations(self, id_fA, id_fB, max_id, is_first):
        for mode in range(0, 8):
            self.pop_out_pop_in(id_fA, id_fB, mode, max_id)
        self.transloc(id_fA, id_fB, max_id)
        if self.active_insert_blocks:
            self.insert_blocks(id_fA, id_fB, max_id)

        # for mode in xrange(14, self.n_tmp_struct):
        #     self.local_flip(id_fA, mode, max_id)
        # self.all_pop_out_pop_in(id_fA, id_fB, max_id, is_first)
        # tic_fillB = time.time()
        # self.all_transloc(id_fA, id_fB, max_id, is_first)
        # print "all_pop out time execution  = ", time.time() - tic_fillB

    def bomb_the_genome(self,):

        size_block = 256
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags // size_block + 1), 1)
        a = np.arange(0, self.n_new_frags, dtype=np.int32)
        np.random.shuffle(a)
        gpu_shuffle_order = ga.to_gpu(ary=a)
        self.ctx.synchronize()
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        self.kern_explode_genome(
            self.gpu_vect_frags.get_ptr(),
            gpu_shuffle_order,
            self.n_new_frags,
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()
        self.modify_gl_cuda_buffer(0, self.gl_window.dt)

    def local_flip(self, id_fA, mode, max_id):
        # mode = 12
        size_block = 256
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags // size_block + 1), 1)
        start = cuda.Event()
        end = cuda.Event()
        local_delta = mode - 11
        vect_frags = self.gpu_vect_frags
        vect_frags.copy_from_gpu()

        pos_fA = vect_frags.pos[id_fA]
        id_contig_A = vect_frags.id_c[id_fA]
        len_contig_A = vect_frags.l_cont[id_fA]

        id_f_in_contig_A = np.nonzero(vect_frags.id_c == id_contig_A)[0]
        neighbours = id_f_in_contig_A
        pos_neighbours = vect_frags.pos[neighbours]

        arg_sort_id = np.argsort(pos_neighbours)

        ordered_neighbours = neighbours[arg_sort_id]
        orientations_neighbours = vect_frags.ori[ordered_neighbours]
        id_up = max(pos_fA - local_delta, 0)
        id_down = min(pos_fA + local_delta, len_contig_A - 1)
        # print "id_fA = ", id_fA
        # print "pos_fA = ", pos_fA
        # print "pos up = ", id_up
        # print "pos down = ", id_down

        start.record()
        self.simple_copy(
            self.scrambled_gpu_vect_frags.get_ptr(),
            self.gpu_vect_frags.get_ptr(),
            self.n_new_frags,
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()
        # print "ordered neighbours = ", ordered_neighbours
        for i in range(id_up, id_down + 1):
            id_fB = ordered_neighbours[i]
            if id_fB != id_fA:
                start.record()
                self.pop_out(
                    self.pop_gpu_vect_frags.get_ptr(),
                    self.scrambled_gpu_vect_frags.get_ptr(),
                    self.pop_gpu_id_contigs,
                    np.int32(id_fB),
                    max_id,
                    self.n_new_frags,
                    block=block_,
                    grid=grid_,
                )
                end.record()
                end.synchronize()
                start.record()
                self.simple_copy(
                    self.scrambled_gpu_vect_frags.get_ptr(),
                    self.pop_gpu_vect_frags.get_ptr(),
                    self.n_new_frags,
                    block=block_,
                    grid=grid_,
                )
                end.record()
                end.synchronize()
                self.scrambled_gpu_vect_frags.copy_from_gpu()
                max_id = self.scrambled_gpu_vect_frags.id_c.max()

        for j in range(id_down, pos_fA, -1):
            id_fB = ordered_neighbours[j]
            ori_fB = orientations_neighbours[j] * -1
            # print "id_fB = ", id_fB
            start.record()
            self.pop_in_4(
                self.collector_gpu_vect_frags[mode].get_ptr(),
                self.scrambled_gpu_vect_frags.get_ptr(),
                np.int32(id_fB),
                np.int32(id_fA),
                max_id,
                np.int32(ori_fB),
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
            end.record()
            end.synchronize()
            start.record()
            self.simple_copy(
                self.scrambled_gpu_vect_frags.get_ptr(),
                self.collector_gpu_vect_frags[mode].get_ptr(),
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
            end.record()
            end.synchronize()
            self.scrambled_gpu_vect_frags.copy_from_gpu()
            max_id = self.scrambled_gpu_vect_frags.id_c.max()
        # print "insert left ok"
        for j in range(id_up, pos_fA):
            id_fB = ordered_neighbours[j]
            ori_fB = orientations_neighbours[j] * -1
            # print "id_fB = ", id_fB
            start.record()
            self.pop_in_3(
                self.collector_gpu_vect_frags[mode].get_ptr(),
                self.scrambled_gpu_vect_frags.get_ptr(),
                np.int32(id_fB),
                np.int32(id_fA),
                max_id,
                np.int32(ori_fB),
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
            end.record()
            end.synchronize()
            start.record()
            self.simple_copy(
                self.scrambled_gpu_vect_frags.get_ptr(),
                self.collector_gpu_vect_frags[mode].get_ptr(),
                self.n_new_frags,
                block=block_,
                grid=grid_,
            )
            end.record()
            end.synchronize()
            self.scrambled_gpu_vect_frags.copy_from_gpu()
            max_id = self.scrambled_gpu_vect_frags.id_c.max()

        start.record()
        self.flip_frag(
            self.collector_gpu_vect_frags[mode].get_ptr(),
            self.scrambled_gpu_vect_frags.get_ptr(),
            np.int32(id_fA),
            self.n_new_frags,
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()

    def test_copy_struct(self, id_fA, id_f_sampled, mode, max_id):
        self.gpu_vect_frags.copy_from_gpu()
        # c = self.gpu_vect_frags
        # print 'id fA = ', id_fA
        # print 'id fB = ', id_f_sampled
        # print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
        # print 'id_cont id_fA = ', c.id_c[id_fA]
        # print 'id_cont id_fB = ', c.id_c[id_f_sampled]
        # print 'id_d id_fA = ', c.id_d[id_fA]
        # print 'id_d id_fB = ', c.id_d[id_f_sampled]
        # print 'rep id_fA = ', c.rep[id_fA]
        # print 'rep id_fB = ', c.rep[id_f_sampled]
        # print 'pos id_fA = ', c.pos[id_fA]
        # print 'pos id_fB = ', c.pos[id_f_sampled]
        # print 'l_cont id_fA = ', c.l_cont[id_fA]
        # print 'l_cont id_fB = ', c.l_cont[id_f_sampled]
        # print 'start_bp id_fA = ', c.start_bp[id_fA]
        # print 'start_bp id_fB = ', c.start_bp[id_f_sampled]
        # print 'l_cont_bp id_fA = ', c.l_cont_bp[id_fA]
        # print 'l_cont_bp id_fB = ', c.l_cont_bp[id_f_sampled]
        # print 'is circle cont id_fA =', c.circ[id_fA]
        # print 'is circle cont id_fB =', c.circ[id_f_sampled]
        # print 'prev id_fA = ', c.prev[id_fA]
        # print 'next id_fA = ', c.next[id_fA]
        # print 'prev id_fB = ', c.prev[id_f_sampled]
        # print 'next id_fB = ', c.next[id_f_sampled]
        # print '########################'
        if mode < 8:
            self.pop_out_pop_in(id_fA, id_f_sampled, mode, max_id)
        elif mode < 12:
            self.transloc(id_fA, id_f_sampled, max_id)
        elif mode >= 12 and self.active_insert_blocks:
            self.insert_blocks(id_fA, id_f_sampled, max_id)
        # else:
        #     self.local_scramble(id_fA, max_id)
        size_block = 1024
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags // size_block + 1), 1)
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        # print 'mode = ', mode
        sampled_vect_frags = self.collector_gpu_vect_frags[mode]
        # sampled_vect_frags.copy_from_gpu()
        # plt.plot(sampled_vect_frags.id_c)
        # plt.show()

        self.copy_vect(
            self.gpu_vect_frags.get_ptr(),
            sampled_vect_frags.get_ptr(),
            self.gpu_id_contigs,
            self.n_new_frags,
            block=block_,
            grid=grid_,
            shared=0,
        )
        end.record()
        end.synchronize()
        # secs = start.time_till(end) * 1e-3
        # self.gpu_vect_frags.copy_from_gpu()
        # c = self.gpu_vect_frags
        # print '########################'
        # print 'id_cont id_fA = ', c.id_c[id_fA]
        # print 'id_cont id_fB = ', c.id_c[id_f_sampled]
        # print 'id_d id_fA = ', c.id_d[id_fA]
        # print 'id_d id_fB = ', c.id_d[id_f_sampled]
        # print 'rep id_fA = ', c.rep[id_fA]
        # print 'rep id_fB = ', c.rep[id_f_sampled]
        # print 'pos id_fA = ', c.pos[id_fA]
        # print 'pos id_fB = ', c.pos[id_f_sampled]
        # print 'l_cont id_fA = ', c.l_cont[id_fA]
        # print 'l_cont id_fB = ', c.l_cont[id_f_sampled]
        # print 'start_bp id_fA = ', c.start_bp[id_fA]
        # print 'start_bp id_fB = ', c.start_bp[id_f_sampled]
        # print 'l_cont_bp id_fA = ', c.l_cont_bp[id_fA]
        # print 'l_cont_bp id_fB = ', c.l_cont_bp[id_f_sampled]
        # print 'is circle cont id_fA =', c.circ[id_fA]
        # print 'is circle cont id_fB =', c.circ[id_f_sampled]
        # print 'prev id_fA = ', c.prev[id_fA]
        # print 'next id_fA = ', c.next[id_fA]
        # print 'prev id_fB = ', c.prev[id_f_sampled]
        # print 'next id_fB = ', c.next[id_f_sampled]
        # print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'

    def setup_rippe_parameters_4_simu(
        self, kuhn, lm, slope, d, val_inter, d_max
    ):

        kuhn = np.float32(kuhn)
        lm = np.float32(lm)
        c1 = np.float32(
            (0.53 * np.power(lm / kuhn, slope)) * np.power(kuhn, -3)
        )
        slope = np.float32(slope)
        d = np.float32(d)
        d_max = np.float32(d_max)
        val_inter = np.float32(val_inter)
        parameters = [kuhn, lm, slope, d]

        def rippe(param, dist):

            return (
                0.53
                * (param[0] ** -3.)
                * np.power((param[1] * dist / param[0]), (param[2]))
                * np.exp(
                    (param[3] - 2)
                    / ((np.power((param[1] * dist / param[0]), 2) + param[3]))
                )
            )

        rippe_inter = rippe(parameters, d_max)
        fact = val_inter / rippe_inter
        p = np.array(
            [(kuhn, lm, c1, slope, d, d_max, fact, val_inter)],
            dtype=self.param_simu_T,
        )
        return p

    def setup_rippe_parameters(self, param, d_max):

        kuhn, lm, slope, d, fact = param
        fact = np.float32(np.abs(fact))
        kuhn = np.float32(np.abs(kuhn))
        lm = np.float32(np.abs(lm))
        c1 = np.float32(
            (0.53 * np.power(lm / kuhn, slope)) * np.power(kuhn, -3)
        )
        slope = np.float32(slope)
        d = np.float32(d)
        fact = np.float32(fact)
        d_max = np.float32(d_max)
        p = np.array(
            [(kuhn, lm, c1, slope, d, d_max, fact, self.mean_value_trans)],
            dtype=self.param_simu_rippe,
        )
        return p

    def setup_model_parameters(self, param, d_max):

        d0, alpha_0, alpha_1, fact = param

        d0 = np.float32(d0)
        d_max = np.float32(d_max)
        alpha_0 = np.float32(alpha_0)
        alpha_1 = np.float32(alpha_1)
        fact = np.float32(fact)
        p = np.array(
            [(d0, d_max, alpha_0, alpha_1, fact, self.mean_value_trans)],
            dtype=self.param_simu_exp,
        )

        return p

    def estimate_parameters_rippe(
        self, max_dist_kb, size_bin_kb, display_graph
    ):
        """
        estimation by least square optimization of Rippe parameters on the
        experimental data
        :param max_dist_kb:
        :param size_bin_kb:
        """
        logger.info("estimation of the parameters of the model")
        self.bins = np.arange(
            size_bin_kb, max_dist_kb + size_bin_kb, size_bin_kb
        )

        self.mean_contacts = np.zeros_like(self.bins, dtype=np.float32)
        self.dict_collect = dict()
        self.gpu_vect_frags.copy_from_gpu()
        epsi = self.mean_value_trans
        for k in self.bins:
            self.dict_collect[k] = []
        for i in range(0, self.n_frags // 10):
            # print "frag i = ", i
            start = self.sparse_matrix.indptr[i]
            end = self.sparse_matrix.indptr[i + 1]
            id_j = self.sparse_matrix.indices[start:end]
            data = self.sparse_matrix.data[start:end]

            info_i = self.np_sub_frags_2_frags[i]
            init_id_fi = int(info_i[0])
            # pos_i = self.S_o_A_frags["pos"][init_id_fi]
            id_c_i = self.S_o_A_frags["id_c"][init_id_fi]
            s_i = (
                self.S_o_A_frags["start_bp"][init_id_fi] / 1000.0
                + self.np_sub_frags_2_frags[i][1]
            )
            len_kb_c_i = self.S_o_A_frags["l_cont_bp"][init_id_fi] / 1000
            # local_bins = np.arange(size_bin_kb, min(len_kb_c_i, max_dist_kb)
            # + size_bin_kb, size_bin_kb) local_storage =
            # np.zeros_like(local_bins, dtype=np.int32)
            #
            # for fj, dj in zip(id_j, data):
            #     info_j = self.np_sub_frags_2_frags[fj]
            #     init_id_fj = info_j[0]
            #     id_c_j = self.S_o_A_frags['id_c'][init_id_fj]
            #     if id_c_i == id_c_j:
            #         pos_j = self.S_o_A_frags['pos'][init_id_fj]
            #         s_j = self.S_o_A_frags['start_bp'][init_id_fj]/1000.0 +
            #         self.np_sub_frags_2_frags[fj][1]
            #         d = np.abs(s_i - s_j)
            #         if d < max_dist_kb:
            #             id_bin = d / size_bin_kb
            #             local_storage[id_bin] += dj
            #             # self.dict_collect[self.bins[id_bin]].append(dj)
            # # we have to add also the zeros
            # for bin, val in zip(local_bins, local_storage):
            #     # print "bin = ", bin
            #     self.dict_collect[bin].append(val)

            if size_bin_kb < len_kb_c_i:
                # local_bins = np.arange(size_bin_kb, min(len_kb_c_i,
                # max_dist_kb) + size_bin_kb, size_bin_kb)
                local_bins = np.arange(
                    size_bin_kb, max_dist_kb + size_bin_kb, size_bin_kb
                )
                local_storage = np.zeros_like(local_bins, dtype=np.int32)

                for fj, dj in zip(id_j, data):
                    info_j = self.np_sub_frags_2_frags[fj]
                    init_id_fj = int(info_j[0])
                    id_c_j = self.S_o_A_frags["id_c"][init_id_fj]
                    if id_c_i == id_c_j:
                        # pos_j = self.S_o_A_frags["pos"][init_id_fj]
                        s_j = (
                            self.S_o_A_frags["start_bp"][init_id_fj] / 1000.0
                            + self.np_sub_frags_2_frags[fj][1]
                        )
                        d = np.abs(s_i - s_j)
                        if d < max_dist_kb:
                            id_bin = int(d / size_bin_kb)
                            local_storage[id_bin] += dj
                            # self.dict_collect[self.bins[id_bin]].append(dj)
                # we have to add also the zeros
                for bin, val in zip(local_bins, local_storage):
                    # print "bin = ", bin
                    self.dict_collect[bin].append(val)

        for id_bin in range(0, len(self.bins)):
            k = self.bins[id_bin]
            self.mean_contacts[id_bin] = np.mean(self.dict_collect[k])

        for id_bin in range(0, len(self.bins)):
            k = self.bins[id_bin]
            tmp = np.mean(self.dict_collect[k])
            if np.isnan(tmp) or tmp == 0:
                # if np.isnan(tmp):
                # if np.isnan(tmp):
                #     print "removing nan"
                self.mean_contacts[id_bin] = np.nan
            else:
                self.mean_contacts[id_bin] = tmp + epsi

        self.mean_contacts_upd = []
        self.bins_upd = []

        for count, ele in zip(self.mean_contacts, self.bins):
            if not np.isnan(count):
                self.bins_upd.append(ele)
                self.mean_contacts_upd.append(count)

        self.bins_upd = np.array(self.bins_upd)
        # self.mean_contacts_upd.sort()
        # self.mean_contacts_upd.reverse()
        self.mean_contacts_upd = np.array(self.mean_contacts_upd)

        # self.mean_contacts_upd =
        # ndi.filters.gaussian_filter1d(self.mean_contacts_upd,
        # sigma=len(self.bins_upd) / 5.)

        # print "size mean contacts vector = ", self.mean_contacts_upd.shape
        # print "mean contacts vector = ", self.mean_contacts_upd

        p, self.y_estim = opti.estimate_param_rippe(
            self.mean_contacts_upd, self.bins_upd
        )
        ##########################################
        logger.info("p from estimate parameters  = {}".format(p))
        # p = list(p[0])
        # p[3] = 2
        # p = tuple(p)
        ##########################################
        fit_param = p
        logger.info("mean value trans = {}".format(self.mean_value_trans))
        ##########################################
        logger.info("BEWARE!!! : I will lower mean value trans  !!!")
        self.mean_value_trans = self.mean_value_trans / 10.0
        ##########################################

        estim_max_dist = opti.estimate_max_dist_intra(
            fit_param, self.mean_value_trans
        )
        logger.info("estimate max dist cis trans = {}".format(estim_max_dist))
        self.param_simu = self.setup_rippe_parameters(
            fit_param, estim_max_dist
        )
        self.param_simu_test = self.param_simu
        self.gpu_param_simu = cuda.mem_alloc(self.param_simu.nbytes)
        self.gpu_param_simu_test = cuda.mem_alloc(self.param_simu.nbytes)

        cuda.memcpy_htod(self.gpu_param_simu, self.param_simu)
        cuda.memcpy_htod(self.gpu_param_simu_test, self.param_simu_test)
        display_graph = False
        if display_graph:
            plt.figure()
            plt.loglog(self.bins_upd, self.mean_contacts_upd, "-*b")
            plt.loglog(self.bins_upd, self.y_estim, "-*r")
            plt.xlabel("genomic distance (kb)")
            plt.ylabel("frequency of contact")
            plt.title(
                r"$\mathrm{Frequency\ of\ contact\ versus\ genomic\ distance"
                " \(data\ tricho test):}\ slope=%.3f,\ max\ cis\ distance(kb)"
                "=%.3f\ d=%.3f\ scale\ factor=%.3f\ $"
                % (
                    self.param_simu["slope"],
                    estim_max_dist,
                    self.param_simu["d"],
                    self.param_simu["fact"],
                )
            )
            plt.legend(["obs", "fit"])
            plt.savefig()
            plt.close()

        self.eval_likelihood_init()

    def estimate_parameters(self, max_dist_kb, size_bin_kb, display_graph):
        """
        estimation by least square optimization of Rippe parameters on the
        experimental data
        :param max_dist_kb:
        :param size_bin_kb:
        """
        logger.info("estimation of the parameters of the model")
        self.bins = np.arange(
            size_bin_kb, max_dist_kb + size_bin_kb, size_bin_kb
        )

        self.mean_contacts = np.zeros_like(self.bins, dtype=np.float32)
        self.dict_collect = dict()
        self.gpu_vect_frags.copy_from_gpu()
        epsi = self.mean_value_trans
        for k in self.bins:
            self.dict_collect[k] = []
        for i in range(0, 2000):
            # print "frag i = ", i
            start = self.sparse_matrix.indptr[i]
            end = self.sparse_matrix.indptr[i + 1]
            id_j = self.sparse_matrix.indices[start:end]
            data = self.sparse_matrix.data[start:end]

            info_i = self.np_sub_frags_2_frags[i]
            init_id_fi = info_i[0]
            # pos_i = self.S_o_A_frags["pos"][init_id_fi]
            id_c_i = self.S_o_A_frags["id_c"][init_id_fi]
            s_i = (
                self.S_o_A_frags["start_bp"][init_id_fi] / 1000.0
                + self.np_sub_frags_2_frags[i][1]
            )
            len_kb_c_i = self.S_o_A_frags["l_cont_bp"][init_id_fi] / 1000
            local_bins = np.arange(
                size_bin_kb,
                min(len_kb_c_i, max_dist_kb) + size_bin_kb,
                size_bin_kb,
            )
            local_storage = np.zeros_like(local_bins, dtype=np.int32)

            for fj, dj in zip(id_j, data):
                info_j = self.np_sub_frags_2_frags[fj]
                init_id_fj = info_j[0]
                id_c_j = self.S_o_A_frags["id_c"][init_id_fj]
                if id_c_i == id_c_j:
                    # pos_j = self.S_o_A_frags["pos"][init_id_fj]
                    s_j = (
                        self.S_o_A_frags["start_bp"][init_id_fj] / 1000.0
                        + self.np_sub_frags_2_frags[fj][1]
                    )
                    d = np.abs(s_i - s_j)
                    if d < max_dist_kb:
                        id_bin = d / size_bin_kb
                        local_storage[id_bin] += dj
                        # self.dict_collect[self.bins[id_bin]].append(dj)
            # we have to add also the zeros
            for my_bin, val in zip(local_bins, local_storage):
                # print "bin = ", bin
                self.dict_collect[my_bin].append(val)

        for id_bin in range(0, len(self.bins)):
            k = self.bins[id_bin]
            self.mean_contacts[id_bin] = np.mean(self.dict_collect[k])

        for id_bin in range(0, len(self.bins)):
            k = self.bins[id_bin]
            tmp = np.mean(self.dict_collect[k])
            if np.isnan(tmp) or tmp == 0:
                # if np.isnan(tmp):
                # if np.isnan(tmp):
                #     print "removing nan"
                self.mean_contacts[id_bin] = np.nan
            else:
                self.mean_contacts[id_bin] = tmp + epsi

        self.mean_contacts_upd = []
        self.bins_upd = []

        for count, ele in zip(self.mean_contacts, self.bins):
            if not np.isnan(count):
                self.bins_upd.append(ele)
                self.mean_contacts_upd.append(count)

        self.bins_upd = np.array(self.bins_upd)
        self.mean_contacts_upd = np.array(self.mean_contacts_upd)

        # self.mean_contacts_upd =
        # ndi.filters.gaussian_filter1d(self.mean_contacts_upd,
        # sigma=len(self.bins_upd) / 5.)

        p, self.y_estim = nuis.estimate_param_hic(
            self.mean_contacts_upd, self.bins_upd
        )
        ##########################################
        fit_param = p.x
        ##########################################
        logger.info("mean value trans = {}".format(self.mean_value_trans))
        ##########################################
        estim_max_dist = nuis.estimate_max_dist_intra(
            fit_param, self.mean_value_trans
        )
        logger.info("max distance cis/trans = {}".format(estim_max_dist))
        ##########################################
        self.param_simu = self.setup_model_parameters(
            fit_param, estim_max_dist
        )
        self.gpu_param_simu = cuda.mem_alloc(self.param_simu.nbytes)
        self.gpu_param_simu_test = cuda.mem_alloc(self.param_simu.nbytes)

        cuda.memcpy_htod(self.gpu_param_simu, self.param_simu)

        if display_graph:
            plt.loglog(self.bins_upd, self.mean_contacts_upd, "-*b")
            plt.loglog(self.bins_upd, self.y_estim, "-*r")
            plt.xlabel("genomic distance (kb)")
            plt.ylabel("frequency of contact")
            plt.legend(["obs", "fit"])
            plt.show()

    def insert_repeats(self, id_f_ins):
        for id in range(0, self.n_new_frags):
            self.gpu_vect_frags.copy_from_gpu()
            max_id = self.gpu_vect_frags.id_c.max()
            if self.gpu_vect_frags.rep[id] == 1:
                logger.info("id repeats = {}".format(id))
                mode = 7
                self.test_copy_struct(id, id_f_ins, mode, max_id)

    def modify_genome(self, n):
        list_breaks = np.random.choice(self.n_new_frags, n * 2, replace=False)
        list_modes = np.random.choice(self.n_tmp_struct, n, replace=True)
        for i in range(0, n):
            self.gpu_vect_frags.copy_from_gpu()
            max_id = self.gpu_vect_frags.id_c.max()
            self.test_copy_struct(
                list_breaks[2 * i],
                list_breaks[2 * i + 1],
                list_modes[i],
                max_id,
            )
            self.gpu_vect_frags.copy_from_gpu()
            c = self.gpu_vect_frags
            if (
                np.any(c.pos < 0)
                or np.any(c.l_cont < 0)
                or np.any(c.l_cont_bp < 0)
                or np.any(c.start_bp < 0)
                or np.any(c.l_cont_bp - c.start_bp <= 0)
                or np.any((c.start_bp != 0) * (c.pos == 0))
                or np.any((c.start_bp == 0) * (c.pos != 0))
                or np.any(c.__next__ == c.id)
                or np.any(c.prev == c.id)
            ):
                logger.info("problem!!!!")
                input("what shoud I do????")
            if np.any(c.l_cont == 0) or np.any(c.l_cont_bp == 0):
                logger.info("problem null contig !!!!")
                input("what shoud I do????")

    def explode_genome(self, dt):
        for i in range(0, self.n_new_frags):
            # print "frag id = ", i
            self.modify_gl_cuda_buffer(i, dt)
            self.gpu_vect_frags.copy_from_gpu()
            max_id = self.gpu_vect_frags.id_c.max()
            self.test_copy_struct(i, 0, 0, max_id)
            self.gpu_vect_frags.copy_from_gpu()
            c = self.gpu_vect_frags
            self.gl_window.remote_update()
            if (
                np.any(c.pos < 0)
                or np.any(c.l_cont < 0)
                or np.any(c.l_cont_bp < 0)
                or np.any(c.start_bp < 0)
                or np.any(c.l_cont_bp - c.start_bp <= 0)
                or np.any((c.start_bp != 0) * (c.pos == 0))
                or np.any((c.start_bp == 0) * (c.pos != 0))
                or np.any(c.__next__ == c.id)
                or np.any(c.prev == c.id)
            ):
                logger.info("problem!!!!")
                input("what shoud I do????")
            if np.any(c.l_cont == 0) or np.any(c.l_cont_bp == 0):
                logger.info("problem null contig !!!!")
                input("what shoud I do????")
        logger.info("genome exploded")
        logger.info("max id = {}".format(max_id))

    def apply_replay_simu(self, id_fA, id_fB, op_sampled, dt):

        # n_modif = len(op_sampled)
        # for i in xrange(0, n_modif):
        #     self.modify_gl_cuda_buffer(i, dt)
        #     self.gpu_vect_frags.copy_from_gpu()
        #     max_id = self.gpu_vect_frags.id_c.max()
        #     self.test_copy_struct(id_fA[i], id_fB[i], op_sampled[i], max_id)
        #     self.gpu_vect_frags.copy_from_gpu()
        #     c = self.gpu_vect_frags
        #     self.gl_window.remote_update()

        self.modify_gl_cuda_buffer(id_fA, dt)
        self.gpu_vect_frags.copy_from_gpu()
        max_id = self.gpu_vect_frags.id_c.max()
        self.test_copy_struct(id_fA, id_fB, op_sampled, max_id)
        self.gpu_vect_frags.copy_from_gpu()
        # c = self.gpu_vect_frags
        self.gl_window.remote_update()

    def display_current_matrix(self, filename):
        self.gpu_vect_frags.copy_from_gpu()
        c = self.gpu_vect_frags
        self.gpu_id_contigs.get(ary=self.cpu_id_contigs)
        pos_frag = np.copy(self.gpu_vect_frags.pos)
        list_id_frags = np.copy(self.gpu_vect_frags.id_d)
        list_id = np.copy(self.cpu_id_contigs)
        list_activ = np.copy(self.gpu_vect_frags.activ)
        unique_contig_id = np.unique(list_id)
        dict_contig = dict()
        full_order = []
        full_order_high = []
        for k in unique_contig_id:
            dict_contig[k] = []
            id_pos = np.ix_(list_id == k)
            is_activ = list_activ[id_pos]
            if np.all(is_activ == 1):
                tmp_ord = np.argsort(pos_frag[id_pos])
                ordered_frag = list_id_frags[id_pos[0][tmp_ord]]
                dict_contig[k].extend(ordered_frag)
                full_order.extend(ordered_frag)
                for i in ordered_frag:
                    ori = c.ori[i]
                    v_high_id_all = list(self.np_sub_frags_id[i])
                    # print "v_high_id_all = ", v_high_id_all
                    # print "sub id =", range(0,v_high_id_all[3])
                    v_high_id = v_high_id_all[: v_high_id_all[3]]
                    id_2_push = list(v_high_id)
                    if ori == -1:
                        id_2_push.reverse()
                    full_order_high.extend(id_2_push)
        # fig = plt.figure(figsize=(10,10))
        # val_max = self.hic_matrix_sub_sampled.max() * 0.01
        # plt.imshow(self.hic_matrix_sub_sampled[np.ix_(full_order,full_order)],
        # vmin=0, vmax=50, interpolation='nearest')
        # fig.savefig(file)
        # plt.show()
        # plt.close()
        # plt.figure()
        # plt.imshow(self.hic_matrix[np.ix_(full_order_high, full_order_high)],
        #            vmin=0, vmax=20,
        #            interpolation='nearest')
        # plt.show()
        matrix = self.hic_matrix[np.ix_(full_order_high, full_order_high)]

        plt.imshow(matrix, vmax=np.percentile(matrix, 99))
        plt.savefig(filename)
        return full_order, dict_contig, full_order_high

    def genome_content(self):

        self.gpu_vect_frags.copy_from_gpu()
        self.gpu_id_contigs.get(ary=self.cpu_id_contigs)
        pos_frag = np.copy(self.gpu_vect_frags.pos)
        next_frag = np.copy(self.gpu_vect_frags.__next__)
        prev_frag = np.copy(self.gpu_vect_frags.prev)
        start_bp = np.copy(self.gpu_vect_frags.start_bp)
        list_id_frags = np.copy(self.gpu_vect_frags.id_d)
        list_activ = np.copy(self.gpu_vect_frags.activ)
        list_id = np.copy(self.gpu_vect_frags.id_c)
        unique_contig_id = np.unique(list_id)
        dict_contig = dict()

        full_order = []
        for k in unique_contig_id:
            dict_contig[k] = dict()
            dict_contig[k]["id"] = []
            dict_contig[k]["pos"] = []
            dict_contig[k]["next"] = []
            dict_contig[k]["prev"] = []
            dict_contig[k]["start_bp"] = []
            dict_contig[k]["id_c"] = []

            id_pos = np.ix_(list_id == k)[0]
            if np.all(list_activ[id_pos] == 1):
                # print "len id_pos = ", id_pos[0]
                tmp_ord = np.argsort(pos_frag[id_pos])
                # print "tmp_ord = ", tmp_ord
                l_start = start_bp[id_pos[tmp_ord]]
                l_pos = pos_frag[id_pos[tmp_ord]]
                l_id_c = list_id[id_pos[tmp_ord]]
                l_next = next_frag[id_pos[tmp_ord]]
                l_prev = prev_frag[id_pos[tmp_ord]]
                ordered_frag = list_id_frags[id_pos[tmp_ord]]
                dict_contig[k]["id"].extend(ordered_frag)
                dict_contig[k]["pos"].extend(l_pos)
                dict_contig[k]["start_bp"].extend(l_start)
                dict_contig[k]["id_c"].extend(l_id_c)
                dict_contig[k]["prev"].extend(l_prev)
                dict_contig[k]["next"].extend(l_next)
                full_order.extend(ordered_frag)
        return full_order, dict_contig

    def load_gl_cuda_vbo(self,):
        # CUDA Ressorces
        self.pos_vbo.bind()

        # Depends on whether pyopengl_accelerate is disabled
        try:
            self.gpu_pos = cudagl.RegisteredBuffer(
                int(self.pos_vbo.buffers[0]), cudagl.graphics_map_flags.NONE
            )
            self.gpu_col = cudagl.RegisteredBuffer(
                int(self.col_vbo.buffers[0]), cudagl.graphics_map_flags.NONE
            )
        except AttributeError:
            self.gpu_pos = cudagl.RegisteredBuffer(
                int(self.pos_vbo.buffer), cudagl.graphics_map_flags.NONE
            )
            self.gpu_col = cudagl.RegisteredBuffer(
                int(self.col_vbo.buffer), cudagl.graphics_map_flags.NONE
            )
        self.col_vbo.bind()
        self.gpu_vel = ga.to_gpu(ary=self.vel)

        self.pos_gen_cuda = cuda.mem_alloc(self.pos.nbytes)
        cuda.memcpy_htod(self.pos_gen_cuda, self.pos)
        self.vel_gen_cuda = cuda.mem_alloc(self.vel.nbytes)
        cuda.memcpy_htod(self.vel_gen_cuda, self.vel)

        self.ctx.synchronize()

    def load_gl_cuda_tex_buffer(self, im_init):
        self.cuda_pbo_resource = cudagl.BufferObject(
            int(self.pbo_im_buffer)
        )  # Mapping GLBuffer to cuda_resource
        self.gpu_im_gl = ga.zeros(
            (self.gl_size_im, self.gl_size_im), dtype=np.int32
        )

        # self.array = cuda.matrix_to_array(im_init, "C") # C-style instead of
        # Fortran-style: row-major self.array = ga.to_gpu(ary=im_init) #
        # C-style instead of Fortran-style: row-major

        # self.texref.set_array(self.array)
        # self.texref.set_flags(cuda.TRSA_OVERRIDE_FORMAT)

    def modify_image_thresh(self, val):
        self.im_thresh += val
        self.im_thresh = min(self.im_thresh, 255)
        self.im_thresh = max(self.im_thresh, 1)
        self.im_thresh = np.uint8(self.im_thresh)
        # print "threshold image = ", self.im_thresh
        # size_block = (16, 16, 1)
        # size_grid = (int(self.n_frags / 16) + 1, int(self.n_frags / 16) + 1)
        # mapping_obj = self.cuda_pbo_resource.map()
        # im_2_update = mapping_obj.device_ptr()
        # self.gl_update_im_thresh(np.intp(im_2_update), self.im_thresh,
        # block=size_block, grid=size_grid, texrefs=[self.texref])
        # self.ctx.synchronize()
        # mapping_obj.unmap() # Unmap the GlBuffer

    def modify_gl_cuda_buffer(self, id_fi, dt):

        start = cuda.Event()
        end = cuda.Event()
        size_block = 512
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags / size_block + 1), 1)
        self.gpu_counter_select_gl.fill(0)
        self.ctx.synchronize()
        self.kern_select_uniq_id_c(
            self.gpu_vect_frags.get_ptr(),
            self.gpu_uniq_id_c,
            self.gpu_uniq_len,
            self.gpu_counter_select_gl,  # retrieve genome contig count&
            np.int32(self.n_new_frags),
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()

        self.n_contigs = self.gpu_counter_select_gl.get()[0]
        self.thrust_module.sort_by_keys_simple(
            self.gpu_uniq_len, int(self.n_contigs), self.gpu_uniq_id_c
        )  # sort the contigs by length

        self.cpu_length_contigs = np.float32(self.gpu_uniq_len.get())
        self.mean_length_contigs = self.cpu_length_contigs[
            : self.n_contigs
        ].mean()

        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_contigs / size_block + 1), 1)
        start.record()
        self.kern_make_old_2_new_id_c(
            self.gpu_uniq_id_c,
            self.gpu_old_2_new_id_c,  # structure to update contig ids
            np.int32(self.n_contigs),
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()

        self.gpu_counter_select_gl.fill(0)
        self.ctx.synchronize()
        start.record()
        self.kern_count_vals(
            self.gpu_uniq_len,
            np.int32(1),
            self.gpu_counter_select_gl,  # collect contigs with length 1
            np.int32(self.n_contigs),
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()
        size_block = 1024
        block_ = (size_block, 1, 1)
        grid_ = (int(self.n_new_frags // size_block) + 1, 1)
        map_pos = self.gpu_pos.map()
        (pos_ptr, pos_siz) = map_pos.device_ptr_and_size()
        map_col = self.gpu_col.map()
        (col_ptr, col_siz) = map_col.device_ptr_and_size()

        max_id = np.float32(self.n_contigs - 1)

        # self.gpu_vect_frags.copy_from_gpu()
        # c = self.gpu_vect_frags
        # n_un = np.nonzero(c.l_cont == 1)[0].shape[0]
        # self.n_contigs_un = self.gpu_counter_select_gl.get()[0]
        # if n_un != self.n_contigs_un:
        #     raw_input(" WTF?....")

        # update pos particles #####
        self.gl_update_pos(
            self.gpu_uniq_len,
            np.intp(pos_ptr),
            np.intp(col_ptr),
            self.gpu_vel,
            self.pos_gen_cuda,
            self.vel_gen_cuda,
            self.gpu_vect_frags.get_ptr(),
            self.gpu_old_2_new_id_c,
            self.gpu_id_contigs,
            max_id,
            self.n_new_frags,
            np.int32(id_fi),
            self.gpu_counter_select_gl,
            self.rng_states,
            np.int32(self.n_generators),
            dt,
            block=block_,
            grid=grid_,
        )
        self.ctx.synchronize()
        map_pos.unmap()
        map_col.unmap()

        # update image #####
        # compute cumulated length thrust...
        self.thrust_module.prefix_sum(self.gpu_uniq_len, int(self.n_contigs))
        # self.d = self.gpu_uniq_len.get()
        # print "prefix sum done!"
        max_id = np.int32(max_id)
        self.kern_frags_2_gl_pxl(
            self.gpu_vect_frags.get_ptr(),
            self.gpu_vect_gl_pxl_frag,
            self.gpu_uniq_len,
            np.int32(max_id),
            np.float32(self.gl_size_im),
            self.n_new_frags,
            block=block_,
            grid=grid_,
        )
        self.ctx.synchronize()
        # print "frags to pixels!"
        # self.e = self.gpu_vect_gl_pxl_frag.get()
        mapping_obj = self.cuda_pbo_resource.map()
        im_2_update = mapping_obj.device_ptr()

        size_block_4_gl = 1024
        block_ = (int(size_block_4_gl), 1, 1)
        grid_ = (int(self.n_data_4_gl / size_block_4_gl + 1), 1)
        # print "grid = ", grid_
        self.gpu_counter_select_4_gl.fill(0)
        self.ctx.synchronize()
        self.kern_prepare_sparse_call_4_gl(
            self.gpu_rows_4_gl,
            self.gpu_cols_4_gl,
            self.gpu_ptr_4_gl,
            self.gpu_vect_gl_pxl_frag,
            self.gpu_info_blocks,
            self.gpu_counter_select_4_gl,
            np.int32(self.gl_size_im),
            np.int32(self.n_data_4_gl),
            block=block_,
            grid=grid_,
        )
        self.ctx.synchronize()
        self.gpu_im_gl.fill(0)
        self.ctx.synchronize()
        self.kern_update_matrix(
            self.gpu_rows_4_gl,
            self.gpu_cols_4_gl,
            self.gpu_data_4_gl,
            self.gpu_ptr_4_gl,
            self.gpu_info_blocks,
            self.gpu_vect_gl_pxl_frag,
            self.gpu_im_gl,
            np.int32(self.gl_size_im),
            np.int32(self.n_data_4_gl),
            block=block_,
            grid=grid_,
        )
        self.ctx.synchronize()

        self.im_thresh = max(1, self.im_thresh)
        all_pix_gl = self.gl_size_im ** 2
        grid_all = (int(all_pix_gl / 1024 + 1), 1)
        self.kern_update_gl_buffer(
            np.intp(im_2_update),
            self.gpu_im_gl,
            np.int32(self.im_thresh),
            np.int32(all_pix_gl),
            block=(1024, 1, 1),
            grid=grid_all,
        )
        self.ctx.synchronize()
        mapping_obj.unmap()  # Unmap the GlBuffer
        # print "time to update gl image = ", t1 - t0
        return max_id

    def test_thrust(self):
        t_start = time.time()
        self.thrust_module.sort_by_keys_zip(
            self.gpu_sub_sp_no_rep_rows,
            int(self.n_sub_vals),
            self.gpu_sub_sp_no_rep_cols,
        )
        t_end = time.time()
        logger.info(
            "GPU thrust sort 1: elapsed time  = {}".format(t_end - t_start)
        )
        self.thrust_module.sort_by_keys_zip(
            self.gpu_sub_sp_no_rep_rows,
            int(self.n_sub_vals),
            self.gpu_sub_sp_no_rep_data,
        )
        t_end_2 = time.time()
        logger.info(
            "GPU thrust sort 2: elapsed time  = {}".format(t_end_2 - t_end)
        )

    def prepare_sparse_call(self):
        size_block = 512
        block_ = (size_block, 1, 1)
        n_blocks = int(self.n_sub_vals // size_block + 1)
        grid_ = (n_blocks, 1)
        start = cuda.Event()
        end = cuda.Event()

        self.gpu_counter_select.fill(0)
        self.ctx.synchronize()

        self.gpu_sub_sp_block_indptr = ga.to_gpu(
            np.zeros((self.n_sub_vals,), dtype=np.int32)
        )
        self.gpu_info_blocks = ga.to_gpu(
            np.zeros((n_blocks,), dtype=self.int3)
        )
        # print "grid = ", grid_
        self.gpu_counter_select.fill(0)
        self.ctx.synchronize()

        start.record()
        self.kern_prepare_sparse_call(
            self.gpu_sub_sp_no_rep_rows,
            self.gpu_info_blocks,
            self.gpu_sub_sp_block_indptr,
            self.gpu_counter_select,
            np.int32(self.n_sub_vals),
            block=block_,
            grid=grid_,
        )
        end.record()
        end.synchronize()
        elapsed_seconds = end.time_since(start) * 1e-3
        logger.debug(
            "GPU prepare sparse call: elapsed time  = {}".format(
                elapsed_seconds
            )
        )

    def return_rippe_vals(self, p0):
        y_eval = opti.peval(self.bins, p0)
        return y_eval

    def f_rippe(self, x, param):

        kuhn, lm, c1, slope, d, d_max, fact, d_nuc = param[0]

        if x < d_max:
            rippe = fact * (
                0.53
                * (kuhn ** -3.)
                * np.power((lm * x / kuhn), slope)
                * np.exp((d - 2) / ((np.power((lm * x / kuhn), 2) + d)))
            )
        else:
            rippe = d_nuc

        return rippe

    def f_hic(self, x, param):
        d_init, d_max, alpha_0, alpha_1, A, v_inter = param[0]
        hic_c = np.zeros(x.shape)
        val_lim_0 = A * np.power(d_init, alpha_0 - alpha_1)
        for i in range(0, len(hic_c)):
            if x[i] < d_init:
                hic_c[i] = A * np.power(x[i], alpha_0)
            else:
                hic_c[i] = val_lim_0 * np.power(x[i], alpha_1)

        return hic_c

    def step_nuisance_parameters(self, dt, t, n_step):

        self.gpu_vect_frags.copy_from_gpu()
        # max_id = self.modify_gl_cuda_buffer(0, dt)
        self.gl_window.remote_update()

        curr_param = np.copy(self.param_simu)
        kuhn, lm, c1, slope, d, d_max, fact, d_nuc = curr_param[0]
        # print " curr param = ", curr_param[0]
        self.sigma_fact = 10 ** (np.log10(fact) - 2)  # for G1
        self.sigma_slope = 0.005  # test malaysian
        self.sigma_d_max = 100  # test simu s1
        self.sigma_d_nuc = 10 ** (np.log10(d_nuc) - 2)  # test s1
        self.sigma_d = 10  # ok for s1
        # randomly select a modifier
        id_modif = np.random.choice(4)
        # id_modif = np.random.choice(4)
        # print "id_modif", id_modif

        if id_modif == 0:  # scale factor
            new_fact = fact + np.random.normal(loc=0.0, scale=self.sigma_fact)
            test_param = [kuhn, lm, slope, d, new_fact]
            # new_d_max = opti.estimate_max_dist_intra(test_param, d_nuc)
            new_d_max = opti.estimate_max_dist_intra_nuis(
                test_param, d_nuc, d_max
            )
            c1 = np.float32(
                (0.53 * np.power(lm / kuhn, slope)) * np.power(kuhn, -3)
            )
            out_test_param = [
                (kuhn, lm, c1, slope, d, new_d_max, new_fact, d_nuc)
            ]
        elif id_modif == 1:  # slope
            new_slope = slope + np.random.normal(
                loc=0.0, scale=self.sigma_slope
            )
            test_param = [kuhn, lm, new_slope, d, fact]
            # new_d_max = opti.estimate_max_dist_intra(test_param, d_nuc)
            new_d_max = opti.estimate_max_dist_intra_nuis(
                test_param, d_nuc, d_max
            )
            c1 = np.float32(
                (0.53 * np.power(lm / kuhn, new_slope)) * np.power(kuhn, -3)
            )
            out_test_param = [
                (kuhn, lm, c1, new_slope, d, new_d_max, fact, d_nuc)
            ]
        elif id_modif == 2:  # max distance intra
            new_d_max = d_max + np.random.normal(
                loc=0.0, scale=self.sigma_d_max
            )
            test_param = [kuhn, lm, slope, d, fact]
            new_d_nuc = opti.peval(new_d_max, test_param)
            c1 = np.float32(
                (0.53 * np.power(lm / kuhn, slope)) * np.power(kuhn, -3)
            )
            out_test_param = [
                (kuhn, lm, c1, slope, d, new_d_max, fact, new_d_nuc)
            ]
        elif id_modif == 3:  # val trans
            if self.sigma_d_nuc <= 0:
                new_d_nuc = d_nuc
            else:
                new_d_nuc = d_nuc + np.random.normal(
                    loc=0.0, scale=self.sigma_d_nuc
                )
            test_param = [kuhn, lm, slope, d, fact]
            # new_d_max = opti.estimate_max_dist_intra(test_param, new_d_nuc)
            new_d_max = opti.estimate_max_dist_intra_nuis(
                test_param, new_d_nuc, d_max
            )
            c1 = np.float32(
                (0.53 * np.power(lm / kuhn, slope)) * np.power(kuhn, -3)
            )
            out_test_param = [
                (kuhn, lm, c1, slope, d, new_d_max, fact, new_d_nuc)
            ]
        else:  # d
            new_d = d + np.random.normal(loc=0.0, scale=self.sigma_d)

            test_param = [kuhn, lm, slope, new_d, fact]
            # new_d_max = opti.estimate_max_dist_intra(test_param, d_nuc)
            new_d_max = opti.estimate_max_dist_intra_nuis(
                test_param, d_nuc, d_max
            )
            c1 = np.float32(
                (0.53 * np.power(lm / kuhn, slope)) * np.power(kuhn, -3)
            )
            out_test_param = [
                (kuhn, lm, c1, slope, new_d, new_d_max, fact, d_nuc)
            ]

        out_test_param = np.array(out_test_param, dtype=self.param_simu_T)
        # print "test param = ", test_param
        # print "out test param = ",out_test_param
        self.param_simu_test = out_test_param
        cuda.memcpy_htod(self.gpu_param_simu_test, out_test_param)

        self.likelihood_nuis = self.eval_likelihood_4_nuisance()
        F_t = self.temperature(t, n_step)
        ratio = np.exp((self.likelihood_nuis - self.likelihood_t) / F_t)
        u = np.random.rand()
        success = 0
        if ratio >= u:
            # print "success"
            success = 1
            cuda.memcpy_htod(self.gpu_param_simu, out_test_param)
            self.param_simu = out_test_param
            self.likelihood_t = self.likelihood_nuis
        # else:
        #     print "reject"

        kuhn, lm, c1, slope, d, d_max, fact, d_nuc = self.param_simu[0]
        p0 = [kuhn, lm, slope, d, fact]
        y_rippe = self.return_rippe_vals(p0)
        return (
            fact,
            d,
            d_max,
            d_nuc,
            slope,
            self.likelihood_t,
            success,
            y_rippe,
        )

    def setup_distri_frags(self,):
        # generates random variables for every frags
        # TO DO : prendre en compte la composition initiale pour les probas!!!

        logger.info("setup jumping distribution: start")
        self.distri_frags = dict()
        fact = 3.0
        # print "N frags = ", self.n_frags
        logger.info(
            "Shape sub mat = {}".format(
                self.sym_sub_sampled_sparse_matrix.shape
            )
        )
        # print "shape indices = ",
        # self.sub_sampled_sparse_matrix.indices.shape
        # print "shape indptr = ", self.sub_sampled_sparse_matrix.indptr.shape
        for i in range(0, self.n_frags):
            start = self.sym_sub_sampled_sparse_matrix.indptr[i]
            end = self.sym_sub_sampled_sparse_matrix.indptr[i + 1]
            # print "id_sp max= ", np.max(id_sp)
            vk = self.sym_sub_sampled_sparse_matrix.data[start:end]
            yk = self.sym_sub_sampled_sparse_matrix.indices[start:end]

            # remove auto contact
            vtmp_1 = np.copy(vk)
            xk_tmp = np.copy(yk)
            id_hetero_contact = np.nonzero(yk != i)[0]

            xk = xk_tmp[id_hetero_contact]
            vtmp = vtmp_1[id_hetero_contact]
            dat = np.float32(vtmp) * fact
            # remove auto contact
            if dat.sum() > 0:
                # tmp_pk = (dat / dat.sum())**fact
                # pk = tmp_pk / tmp_pk.sum()
                pk = dat / np.linalg.norm(dat, 1)
            else:
                tmp = np.ones_like(dat, dtype=np.float32)
                pk = tmp / tmp.sum()

            if len(xk) > 0:
                self.distri_frags[i] = dict()
                # self.distri_frags[i]['distri'] =
                # stats.rv_discrete(name='frag_'+str(i), values=(xk, pk))
                self.distri_frags[i]["distri"] = "ok"
                self.distri_frags[i]["xk"] = xk
                self.distri_frags[i]["pk"] = pk
            else:
                self.distri_frags[i] = dict()
                self.distri_frags[i]["distri"] = None
        logger.info("setup jumping distribution: done")

    def return_neighbours(self, id_fA, delta0):
        # print "id_frag = ", id_fA

        # delta0 = self.n_neighbours

        # self.n_neighbours = 10

        ori_id = self.gpu_vect_frags.id_d[id_fA]
        # delta = min(self.n_neighbours, delta0)
        delta = delta0

        # DEBUG
        # if ori_id in self.id_frag_duplicated:
        #     delta = max(self.n_neighbors, delta)

        # fact = 3
        # pk = self.distri_frags[ori_id]['pk']**fact
        # distri = pk / pk.sum()

        if self.distri_frags[ori_id]["distri"] is not None:
            distri = self.distri_frags[ori_id]["pk"]
            n_max_candidates = min(delta, np.nonzero(distri != 0)[0].shape[0])

            init_id = np.random.choice(
                self.distri_frags[ori_id]["xk"],
                n_max_candidates,
                p=distri,
                replace=False,
            )
        else:
            init_id = np.random.choice(self.n_frags, delta, replace=False)
        out = []

        if ori_id in self.id_frag_duplicated:
            d = self.frag_dispatcher[ori_id]
            dup = np.lib.arraysetops.setdiff1d(
                self.collector_id_repeats[d["x"] : d["y"]], id_fA
            )
            out.extend(dup)

        for id_fB in init_id:
            d = self.frag_dispatcher[id_fB]
            out.extend(self.collector_id_repeats[d["x"] : d["y"]])

        real_out = []
        for ele in out:
            if ele not in self.id_frags_blacklisted:
                real_out.append(ele)

        return real_out

    def set_jumping_distributions_parameters(self, delta):
        self.define_neighbourhood()
        self.jump_dictionnary = dict()
        for i in range(0, self.n_frags):
            id_neighbours = self.sorted_neighbours[i, -delta:]
            # id_no_neighbours = self.sorted_neighbours[i, 0:delta]
            scores = np.array(
                self.matrix_normalized[i, id_neighbours], dtype=np.float32
            )
            # val_mean = self.matrix_normalized[i, id_no_neighbours].mean()
            norm_scores = scores / scores.sum()
            self.jump_dictionnary[i] = dict()
            self.jump_dictionnary[i]["proba"] = norm_scores
            self.jump_dictionnary[i]["frags"] = np.array(
                id_neighbours, dtype=np.int32
            )
            self.jump_dictionnary[i]["distri"] = np.zeros(
                (self.n_frags), dtype=np.float32
            )
            self.jump_dictionnary[i]["set_frags"] = set()
            for k in range(0, delta):
                id_frag = self.jump_dictionnary[i]["frags"][k]
                self.jump_dictionnary[i]["set_frags"].add(id_frag)
                proba = self.jump_dictionnary[i]["proba"][k]
                self.jump_dictionnary[i]["distri"][id_frag] = proba
        # test = True
        # while test:
        #     id = raw_input(" give a frag id ?")
        #     id = int(id)
        #     print self.jump_dictionnary[id]
        #     test = raw_input(" keep on ? ")
        #     test = int(test) != 0

    def temperature(self, t, n_step):
        # T0 = np.float32(6 * 10 ** 3)
        # Tf = np.float32(6*10 ** 2)
        #
        # n_step = n_step
        # limit_rejection = 0.5
        # if t <= n_step * limit_rejection:
        #     val = T0 * (Tf / T0)**(t / (n_step* limit_rejection))
        # else:
        #     val = T0 * (Tf / T0)**(limit_rejection)
        #     # val = Tf
        # # print "temperature = ", val
        val = 1.0
        return val

    def free_gpu(self,):

        self.gpu_vect_frags.__del__()
        self.rng_states.free()
        (free, total) = cuda.mem_get_info()
        logger.debug(
            (
                "Global memory occupancy after cleaning processes: %f%% free"
                % (free * 100 / total)
            )
        )
        logger.debug(("Global free memory  :%i Mo free" % (free / 10 ** 6)))
        self.ctx.detach()
        del self.module

    def modify_param_simu(self, param_simu, id_val, val):
        new_param_simu = np.copy(param_simu)
        if id_val == 0:
            new_param_simu["d"] = np.float32(val)
        elif id_val == 1:
            new_param_simu["slope"] = np.float32(val)

        return new_param_simu
