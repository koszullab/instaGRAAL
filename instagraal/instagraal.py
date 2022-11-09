#!/usr/bin/env python3

"""Large genome reassembly based on Hi-C data.

Usage:
    instagraal <hic_folder> <reference.fa> [<output_folder>]
               [--level=4] [--cycles=100] [--coverage-std=1]
               [--neighborhood=5] [--device=0] [--circular] [--bomb]
               [--save-matrix] [--pyramid-only] [--save-pickle] [--simple]
               [--quiet] [--debug]

Options:
    -h, --help              Display this help message.
    --version               Display the program's current version.
    -l 4, --level 4         Level (resolution) of the contact map.
                            Increasing level by one means a threefold smaller
                            resolution but also a threefold faster computation
                            time. [default: 4]
    -n 100, --cycles 100    Number of iterations to perform for each bin.
                            (row/column of the contact map). A high number of
                            cycles has diminishing returns but there is a
                            necessary minimum for assembly convergence.
                            [default: 100]
    -c 1, --coverage-std 1  Number of standard deviations below the mean.
                            coverage, below which fragments should be filtered
                            out prior to binning. [default: 1]
    -N 5, --neighborhood 5  Number of neighbors to sample for potential
                            mutations for each bin. [default: 5]
    --device 0              If multiple graphic cards are available, select
                            a specific device (numbered from 0). [default: 0]
    -C, --circular          Indicates genome is circular. [default: False]
    -b, --bomb              Explode the genome prior to scaffolding.
                            [default: False]
    --pyramid-only          Only build multi-resolution contact maps (pyramids)
                            and don't do any scaffolding. [default: False]
    --save-pickle           Dump all info from the instaGRAAL run into a
                            pickle. Primarily for development purposes, but
                            also for advanced post hoc introspection.
                            [default: False]
    --save-matrix           Saves a preview of the contact map after each
                            cycle. [default: False]
    --simple                Only perform operations at the edge of the contigs.
                            [default: False]
    --quiet                 Only display warnings and errors as outputs.
                            [default: False]
    --debug                 Display debug information. For development purposes
                            only. Mutually exclusive with --quiet, and will
                            override it. [default: False]

"""

import sys
import os
import docopt

import pycuda.autoinit
import pycuda.driver as cuda

# helper modules
from instagraal import glutil
from instagraal.vector import Vec
import numpy as np
import matplotlib.pyplot as plt
from instagraal.simu_single import simulation

import pickle
import logging
from instagraal import log
from instagraal.log import logger

VERSION_NUMBER = "0.1.2"

DEFAULT_CYCLES = 100
DEFAULT_LEVEL = 4
DEFAULT_NEIGHBOURS = 5
DEFAULT_BOMB = True
DEFAULT_ITERATIONS_MCMC = 100
DEFAULT_ITERATIONS_EM = 30
DEFAULT_BURN_IN_CYCLES = 2
DEFAULT_COVERAGE_STDS = 1
DEFAULT_CIRCULAR = False


class instagraal_class:
    """class to manage the calculations performed by
    the scaffolder.
    
    [description]
    
    Parameters
    ----------
    name : str
        The name of the project. Will determine the window title.
    folder_path : str or pathlib.Path
        The directory containing the Hi-C conact map.
    fasta : str or pathlib.Path
        The path to the reference genome in FASTA format.
    device : int
        The identifier of the graphic card to be used, numbered from 0. If only
        one is available, it should be 0.
    level : int
        The level (resolution) at which to perform scaffolding.
    n_iterations_em : int
        The number of EM (expectation maximization) iterations.
    n_iterations_mcmc : int
        The number of MCMC (Markov chain Monte-Carlo) iterations.
    is_simu : bool
        Whether the parameters should be simulated. Mutually exclusive with
        use_rippe and will override it.
    scrambled : bool
        Whether to scramble the genome.
    perform_em : bool
        Whether to perform EM (expectation maximization).
    use_rippe : bool
        Whether to explicitly use the model from Rippe et al., 2001.
    sample_param : bool
        Whether to sample the parameters.
    thresh_factor : float
        The sparsity (coverage) threshold below which fragments are discarded,
        as a number of standard deviations below the mean.
    output_folder : str or pathlib.Path
        The path to the output folder where the scaffolded genome and other
        relevant information will be saved.
    """

    def __init__(
        self,
        name,
        folder_path,
        fasta,
        device,
        level,
        n_iterations_em,
        n_iterations_mcmc,
        is_simu,
        scrambled,
        perform_em,
        use_rippe,
        sample_param,
        thresh_factor,
        output_folder,
    ):
        """Initialize parameters
        """

        self.device = device
        self.scrambled = scrambled
        self.n_iterations_em = n_iterations_em
        self.n_iterations_mcmc = n_iterations_mcmc
        self.sample_param = sample_param
        self.simulation = simulation(
            name,
            folder_path,
            fasta,
            level,
            n_iterations_em,
            is_simu,
            use_rippe,
            thresh_factor,
            output_folder=output_folder,
        )
        self.dt = np.float32(0.01)
        self.collect_likelihood = []
        self.collect_n_contigs = []
        self.collect_mean_len = []
        self.collect_op_sampled = []
        # self.collect_id_f_sampled = []
        self.collect_id_fA_sampled = []
        self.collect_id_fB_sampled = []
        self.collect_full_likelihood = []
        self.collect_dist_from_init_genome = []
        self.collect_fact = []
        self.collect_slope = []
        self.collect_d = []
        self.collect_d_nuc = []
        self.collect_d_max = []
        self.collect_likelihood_nuisance = []
        self.collect_success = []
        self.collect_all = []
        self.file_mean_len = os.path.join(
            self.simulation.output_folder, "behaviour_mean_len.pdf"
        )
        self.file_n_contigs = os.path.join(
            self.simulation.output_folder, "behaviour_n_contigs.pdf"
        )

        self.file_fact = os.path.join(
            self.simulation.output_folder, "behaviour_fact.pdf"
        )
        self.file_slope = os.path.join(
            self.simulation.output_folder, "behaviour_slope.pdf"
        )
        self.file_d_nuc = os.path.join(
            self.simulation.output_folder, "behaviour_d_nuc.pdf"
        )
        self.file_d = os.path.join(self.simulation.output_folder, "behaviour_d.pdf")
        self.file_d_max = os.path.join(
            self.simulation.output_folder, "behaviour_d_max.pdf"
        )

        self.file_dist_init_genome = os.path.join(
            self.simulation.output_folder, "behaviour_dist_init_genome.pdf"
        )

        self.txt_file_mean_len = os.path.join(
            self.simulation.output_folder, "list_mean_len.txt"
        )
        self.txt_file_n_contigs = os.path.join(
            self.simulation.output_folder, "list_n_contigs.txt"
        )
        self.txt_file_dist_init_genome = os.path.join(
            self.simulation.output_folder, "list_dist_init_genome.txt"
        )
        self.txt_file_likelihood = os.path.join(
            self.simulation.output_folder, "list_likelihood.txt"
        )

        self.txt_file_fact = os.path.join(
            self.simulation.output_folder, "list_fact.txt"
        )
        self.txt_file_slope = os.path.join(
            self.simulation.output_folder, "list_slope.txt"
        )
        self.txt_file_d_nuc = os.path.join(
            self.simulation.output_folder, "list_d_nuc.txt"
        )
        self.txt_file_d = os.path.join(self.simulation.output_folder, "list_d.txt")
        self.txt_file_d_max = os.path.join(
            self.simulation.output_folder, "list_d_max.txt"
        )
        self.txt_file_success = os.path.join(
            self.simulation.output_folder, "list_success.txt"
        )
        self.txt_file_list_mutations = os.path.join(
            self.simulation.output_folder, "list_mutations.txt"
        )
        self.file_all_data = os.path.join(
            self.simulation.output_folder, "behaviour_all.txt"
        )

    def full_em(
        self, n_cycles, n_neighbours, bomb, id_start_sample_param, save_matrix=False
    ):
        sampler = self.simulation.sampler
        if bomb:
            sampler.bomb_the_genome()
        list_frags = np.arange(0, sampler.n_new_frags)
        t = 0
        n_iter = n_cycles * sampler.n_new_frags
        for j in range(0, n_cycles):
            sampler.gpu_vect_frags.copy_from_gpu()

            np.random.shuffle(list_frags)
            logger.info("cycle = {}".format(j))
            # np.random.shuffle(list_frags)
            count = 0
            nb_frags = list_frags.size
            for id_frag in list_frags:
                count += 1
                if count % 100 == 0:
                    print("{}% proceeded".format(count / nb_frags))
                (
                    o,
                    dist,
                    op_sampled,
                    id_f_sampled,
                    mean_len,
                    n_contigs,
                ) = sampler.step_sampler(id_frag, n_neighbours, self.dt)
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_op_sampled.append(op_sampled)
                self.collect_id_fB_sampled.append(id_f_sampled)
                self.collect_id_fA_sampled.append(id_frag)
                self.collect_dist_from_init_genome.append(dist)

                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(sampler.n_contigs)
                self.str_curr_id_frag = "current frag = " + str(id_frag)
                self.str_curr_dist = "current dist = " + str(dist)
                self.str_curr_cycle = "current cycle = " + str(j)
                if self.sample_param and j > id_start_sample_param:
                    (
                        fact,
                        d,
                        d_max,
                        d_nuc,
                        slope,
                        self.likelihood_t_nuis,
                        success,
                        y_rippe,
                    ) = sampler.step_nuisance_parameters(self.dt, t, n_iter)

                    self.collect_fact.append(fact)
                    self.collect_d.append(d)
                    self.collect_d_max.append(d_max)
                    self.collect_d_nuc.append(d_nuc)
                    self.collect_slope.append(slope)
                    self.collect_likelihood_nuisance.append(self.likelihood_t_nuis)
                    self.collect_success.append(success)
                    self.y_eval = y_rippe
                t += 1
            c = sampler.gpu_vect_frags
            c.copy_from_gpu()
            file_out = os.path.join(
                self.simulation.output_folder, "save_simu_step_" + str(j) + ".txt"
            )
            h = open(file_out, "w")
            for pos, start_bp, id_c, ori in zip(c.pos, c.start_bp, c.id_c, c.ori):
                h.write(
                    str(pos)
                    + "\t"
                    + str(start_bp)
                    + "\t"
                    + str(id_c)
                    + "\t"
                    + str(ori)
                    + "\n"
                )
            h.close()
            try:
                self.simulation.export_new_fasta()
                self.save_behaviour_to_txt()
            except OSError as e:
                logger.warning(
                    "Warning, could not write output files at {}: {}".format(j, e)
                )
            try:
                if save_matrix:
                    my_file_path = os.path.join(
                        self.simulation.output_folder, "matrix_cycle_" + str(j) + ".png"
                    )
                    matrix = self.simulation.sampler.gpu_im_gl.get()
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(
                        top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
                    )
                    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.figure()
                    plt.imshow(matrix, vmax=np.percentile(matrix, 99), cmap="Reds")
                    plt.axis("off")
                    plt.savefig(
                        my_file_path, bbox_inches="tight", pad_inches=0.0, dpi=300
                    )
                    plt.close()
            except OSError as e:
                logger.warning(
                    "Could not write matrix at cycle {} "
                    "due to error: {}".format(j, e)
                )

        self.save_behaviour_to_txt()

    def start_EM(self,):
        logger.info("start expectation maximization ... ")
        delta = 15
        logger.info(self.simulation.n_iterations)
        delta = np.int32(
            np.floor(np.linspace(3, 5, np.floor(self.n_iterations_em / 3.0)))
        )  # param ok simu
        # delta = np.int32(np.floor(np.linspace(3, 4,
        # np.floor(self.n_iterations_em / 2.))))
        delta = list(delta)
        d_ext = list(
            np.floor(np.linspace(10, 15, np.floor(self.n_iterations_em / 3.0) + 1))
        )  # param ok simu
        delta.extend(d_ext)
        logger.info(delta)
        logger.info(("len delta = ", len(delta)))
        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.input_matrix
        )
        self.simulation.sampler.init_likelihood()
        self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
        # ready = 0
        # while ready != '1':
        #     ready = raw_input("ready to start?")
        #     self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
        #     self.remote_update()
        if self.scrambled:
            #     self.simulation.sampler.modify_genome(500)
            self.simulation.sampler.explode_genome(self.dt)
        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.scrambled_input_matrix
        )
        list_frags = np.arange(0, self.simulation.sampler.n_new_frags, dtype=np.int32)
        iteration = 0
        n_iter = np.float32(self.n_iterations_em)
        self.bins_rippe = self.simulation.sampler.bins
        for j in range(0, self.n_iterations_em):
            logger.info("cycle = {}".format(j))
            self.str_curr_cycle = "current cycle = " + str(j)
            np.random.shuffle(list_frags)
            # d = self.simulation.sampler.step_nuisance_parameters(0, 0, 0)
            for i in list_frags:
                # print "id_frag =", i
                # if j == 0 and iter == 0:
                #     raw_input("ready ?")
                (
                    o,
                    n_contigs,
                    min_len,
                    mean_len,
                    max_len,
                    op_sampled,
                    id_f_sampled,
                    dist,
                    temp,
                ) = self.simulation.sampler.step_max_likelihood(
                    i, delta[j], 512, self.dt, np.float32(j), n_iter
                )

                # o, n_contigs, min_len, mean_len, max_len =
                # self.simulation.sampler.new_sample_fi(i, delta[j], 512, 200)
                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(n_contigs)
                self.str_curr_id_frag = "current frag = " + str(i)
                self.str_curr_dist = "current dist = " + str(dist)
                self.str_curr_temp = "current temperature = " + str(temp)
                # self.str_curr_d = "current d = "+ str(d)
                self.collect_full_likelihood.append(
                    self.simulation.sampler.likelihood_t
                )
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_op_sampled.append(op_sampled)
                self.collect_id_fB_sampled.append(id_f_sampled)
                self.collect_id_fA_sampled.append(i)
                self.collect_dist_from_init_genome.append(dist)
                iteration += 1
                # sampling nuisance parameters
                (
                    fact,
                    d,
                    d_max,
                    d_nuc,
                    slope,
                    likeli,
                    success,
                    y_eval,
                ) = self.simulation.sampler.step_nuisance_parameters(
                    self.dt, np.float32(j), n_iter
                )
                self.collect_fact.append(fact)
                self.collect_d.append(d)
                self.collect_d_max.append(d_max)
                self.collect_d_nuc.append(d_nuc)
                self.collect_slope.append(slope)
                self.collect_likelihood_nuisance.append(likeli)
                self.collect_success.append(success)
                self.y_eval = y_eval

        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.output_matrix_em
        )
        self.simulation.export_new_fasta()
        self.simulation.plot_info_simu(
            self.collect_likelihood,
            self.collect_n_contigs,
            self.file_n_contigs,
            "n_contigs",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood,
            self.collect_mean_len,
            self.file_mean_len,
            "mean length contigs",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood,
            self.collect_dist_from_init_genome,
            self.file_dist_init_genome,
            "distance from init genome",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood_nuisance,
            self.collect_slope,
            self.file_slope,
            "slope",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood_nuisance,
            self.collect_fact,
            self.file_fact,
            "scale factor",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood_nuisance,
            self.collect_d_nuc,
            self.file_d_nuc,
            "val trans",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood_nuisance, self.collect_d, self.file_d, "d"
        )
        self.save_behaviour_to_txt()

    def start_EM_all(self,):
        logger.info("start expectation maximization ... ")
        delta = 15
        logger.info((self.simulation.n_iterations))
        delta = np.int32(
            np.floor(np.linspace(3, 4, np.floor(self.n_iterations_em / 2.0)))
        )  # param ok simu
        # delta = np.int32(np.floor(np.linspace(3, 4,
        # np.floor(self.n_iterations_em / 2.))))
        delta = list(delta)
        d_ext = list(
            np.floor(np.linspace(5, 10, np.floor(self.n_iterations_em / 2.0) + 1))
        )  # param ok simu
        # d_ext = list(np.floor(np.linspace(10, 15,
        # np.floor(self.n_iterations_em / 2.) + 1)))
        delta.extend(d_ext)
        logger.info(delta)
        logger.info(("len delta = ", len(delta)))
        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.input_matrix
        )
        self.simulation.sampler.init_likelihood()
        self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
        if self.scrambled:
            self.simulation.sampler.explode_genome(self.dt)
        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.scrambled_input_matrix
        )
        list_frags = np.arange(0, self.simulation.sampler.n_new_frags, dtype=np.int32)
        iteration = 0
        n_iter = np.float32(self.n_iterations_em)
        self.bins_rippe = self.simulation.sampler.bins
        self.headers = [
            "likelihood",
            "n_contigs",
            "min_len_contigs",
            "mean_len_contigs",
            "max_len_contigs",
            "operation_sampled",
            "id_f_sampled",
            "distance_from_init",
            "scale factor",
            "d",
            "max_distance_intra",
            "n_contacts_inter",
            "slope",
            "success",
        ]

        (
            fact,
            d,
            d_max,
            d_nuc,
            slope,
            likeli,
            success,
            y_eval,
        ) = self.simulation.sampler.step_nuisance_parameters(
            self.dt, np.float32(0), n_iter
        )
        for j in range(0, self.n_iterations_em):
            logger.info("cycle = {}".format(j))
            self.str_curr_cycle = "current cycle = " + str(j)
            np.random.shuffle(list_frags)
            # d = self.simulation.sampler.step_nuisance_parameters(0, 0, 0)
            for i in list_frags:
                # print "id_frag =", i
                # if j == 0 and iter == 0:
                #     raw_input("ready ?")
                (
                    o,
                    n_contigs,
                    min_len,
                    mean_len,
                    max_len,
                    op_sampled,
                    id_f_sampled,
                    dist,
                    temp,
                ) = self.simulation.sampler.step_max_likelihood(
                    i, delta[j], 512, self.dt, np.float32(j), n_iter
                )

                success = 1
                vect_collect = (
                    o,
                    n_contigs,
                    min_len,
                    mean_len,
                    max_len,
                    op_sampled,
                    id_f_sampled,
                    dist,
                    fact,
                    d,
                    d_max,
                    d_nuc,
                    slope,
                    success,
                )

                self.collect_all.append(vect_collect)

                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(n_contigs)
                self.str_curr_id_frag = "current frag = " + str(i)
                self.str_curr_dist = "current dist = " + str(dist)
                self.str_curr_temp = "current temperature = " + str(temp)
                # self.str_curr_d = "current d = "+ str(d)
                self.collect_full_likelihood.append(
                    self.simulation.sampler.likelihood_t
                )
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_op_sampled.append(op_sampled)
                self.collect_id_fB_sampled.append(id_f_sampled)
                self.collect_id_fA_sampled.append(i)
                self.collect_dist_from_init_genome.append(dist)
                iteration += 1
                # sampling nuisance parameters
                (
                    fact,
                    d,
                    d_max,
                    d_nuc,
                    slope,
                    likeli,
                    success,
                    y_eval,
                ) = self.simulation.sampler.step_nuisance_parameters(
                    self.dt, np.float32(j), n_iter
                )
                vect_collect = (
                    likeli,
                    n_contigs,
                    min_len,
                    mean_len,
                    max_len,
                    op_sampled,
                    id_f_sampled,
                    dist,
                    fact,
                    d,
                    d_max,
                    d_nuc,
                    slope,
                    success,
                )

                self.collect_all.append(vect_collect)
                self.collect_fact.append(fact)
                self.collect_d.append(d)
                self.collect_d_max.append(d_max)
                self.collect_d_nuc.append(d_nuc)
                self.collect_slope.append(slope)
                self.collect_likelihood_nuisance.append(likeli)
                self.collect_success.append(success)
                self.y_eval = y_eval

        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.output_matrix_em
        )
        self.simulation.export_new_fasta()
        self.simulation.plot_all_info_simu(self.collect_all, self.header)

    def start_EM_nuisance(self,):
        logger.info("start expectation maximization ... ")
        delta = 15
        logger.info((self.simulation.n_iterations))
        delta = np.int32(
            np.floor(np.linspace(3, 4, np.floor(self.n_iterations_em / 2.0)))
        )  # param ok simu
        # delta = np.int32(np.floor(np.linspace(3, 4,
        # np.floor(self.n_iterations_em / 2.))))
        delta = list(delta)
        d_ext = list(
            np.floor(np.linspace(5, 10, np.floor(self.n_iterations_em / 2.0) + 1))
        )  # param ok simu
        # d_ext = list(np.floor(np.linspace(10, 15,
        # np.floor(self.n_iterations_em / 2.) + 1)))
        delta.extend(d_ext)
        logger.info(delta)
        logger.info(("len delta = ", len(delta)))
        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.input_matrix
        )
        self.simulation.sampler.init_likelihood()
        self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
        # ready = 0
        # while ready != '1':
        #     ready = raw_input("ready to start?")
        #     self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
        #     self.remote_update()
        if self.scrambled:
            #     self.simulation.sampler.modify_genome(500)
            self.simulation.sampler.explode_genome(self.dt)
        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.scrambled_input_matrix
        )
        list_frags = np.arange(0, self.simulation.sampler.n_new_frags, dtype=np.int32)
        # iter = 0
        n_iter = np.float32(self.n_iterations_em)
        self.bins_rippe = self.simulation.sampler.bins
        for j in range(0, self.n_iterations_em):
            logger.info("cycle = {}".format(j))
            self.str_curr_cycle = "current cycle = " + str(j)
            np.random.shuffle(list_frags)
            # d = self.simulation.sampler.step_nuisance_parameters(0, 0, 0)
            for i in list_frags:
                # # print "id_frag =", i
                # if bool(glutMainLoopEvent):
                #     glutMainLoopEvent()
                # else:
                #     glutCheckLoop()
                # # if j == 0 and iter == 0:
                # #     raw_input("ready ?")
                # o, n_contigs, min_len, mean_len, max_len, op_sampled,
                # id_f_sampled, dist, temp =
                # self.simulation.sampler.step_max_likelihood(i, delta[j], 512,
                # self.dt, np.float32(j), n_iter)
                #
                # o, n_contigs, min_len, mean_len, max_len =
                # self.simulation.sampler.new_sample_fi(i, delta[j], 512, 200)
                # self.str_likelihood = "likelihood = " + str(o)
                # self.str_n_contigs = "n contigs = " + str(n_contigs)
                # self.str_curr_id_frag = "current frag = "+ str(i)
                # self.str_curr_dist = "current dist = "+ str(dist)
                # self.str_curr_temp = "current temperature = "+str(temp)
                # # self.str_curr_d = "current d = "+ str(d)
                # self.collect_full_likelihood.append(self.simulation.sampler.
                # likelihood_t)
                # self.collect_likelihood.append(o)
                # self.collect_n_contigs.append(n_contigs)
                # self.collect_mean_len.append(mean_len)
                # self.collect_op_sampled.append(op_sampled)
                # self.collect_id_fB_sampled.append(id_f_sampled)
                # self.collect_id_fA_sampled.append(i)
                # self.collect_dist_from_init_genome.append(dist)
                # iter += 1
                # # sampling nuisance parameters
                (
                    fact,
                    d,
                    d_max,
                    d_nuc,
                    slope,
                    likeli,
                    success,
                    y_eval,
                ) = self.simulation.sampler.step_nuisance_parameters(
                    self.dt, np.float32(j), n_iter
                )
                self.collect_fact.append(fact)
                self.collect_d.append(d)
                self.collect_d_max.append(d_max)
                self.collect_d_nuc.append(d_nuc)
                self.collect_slope.append(slope)
                self.collect_likelihood_nuisance.append(likeli)
                self.collect_success.append(success)
                self.y_eval = y_eval

        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.output_matrix_em
        )
        self.simulation.export_new_fasta()
        # self.simulation.plot_info_simu(self.collect_likelihood,
        # self.collect_n_contigs, self.file_n_contigs, "n_contigs")
        # self.simulation.plot_info_simu(self.collect_likelihood,
        # self.collect_mean_len, self.file_mean_len, "mean length contigs")
        # self.simulation.plot_info_simu(self.collect_likelihood,
        # self.collect_dist_from_init_genome, self.file_dist_init_genome,
        # "distance from init genome")
        # self.simulation.plot_info_simu(self.collect_likelihood_nuisance,
        # self.collect_slope, self.file_slope, "slope")
        # self.simulation.plot_info_simu(self.collect_likelihood_nuisance,
        # self.collect_fact, self.file_fact, "scale factor")
        # self.simulation.plot_info_simu(self.collect_likelihood_nuisance,
        # self.collect_d_nuc, self.file_d_nuc, "val trans")
        # self.simulation.plot_info_simu(self.collect_likelihood_nuisance,
        # self.collect_d, self.file_d, "d")
        # self.save_behaviour_to_txt()

    def start_EM_no_scrambled(self,):
        logger.info("start expectation maximization ... ")
        delta = 15
        logger.info((self.simulation.n_iterations))
        # delta = np.int32(np.floor(np.linspace(3, 4,
        # np.floor(self.n_iterations_em / 2.)))) # param ok simu
        delta = np.int32(
            np.floor(np.linspace(3, 4, np.floor(self.n_iterations_em / 2.0)))
        )
        delta = list(delta)
        # d_ext = list(np.floor(np.linspace(5, 10,
        # np.floor(self.n_iterations_em / 2.) + 1))) # param ok simu
        d_ext = list(
            np.floor(np.linspace(10, 15, np.floor(self.n_iterations_em / 2.0) + 1))
        )
        delta.extend(d_ext)
        logger.info(delta)
        logger.info(("len delta = ", len(delta)))
        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.input_matrix
        )
        # self.simulation.sampler.init_likelihood()

        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.scrambled_input_matrix
        )
        list_frags = np.arange(0, self.simulation.sampler.n_new_frags, dtype=np.int32)
        iter = 0
        n_iter = np.float32(self.n_iterations_em)
        for j in range(0, self.n_iterations_em):
            logger.info("cycle = {}".format(j))
            self.str_curr_cycle = "current cycle = " + str(j)
            np.random.shuffle(list_frags)
            # d = self.simulation.sampler.step_nuisance_parameters(0, 0, 0)
            for i in list_frags:
                # print "id_frag =", i
                # if j == 0 and iter == 0:
                #     raw_input("ready ?")
                (
                    o,
                    n_contigs,
                    min_len,
                    mean_len,
                    max_len,
                    op_sampled,
                    id_f_sampled,
                    dist,
                    temp,
                ) = self.simulation.sampler.step_max_likelihood(
                    i, delta[j], 512, self.dt, np.float32(j), n_iter
                )
                # o, n_contigs, min_len, mean_len, max_len =
                # self.simulation.sampler.new_sample_fi(i, delta[j], 512, 200)
                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(n_contigs)
                self.str_curr_id_frag = "current frag = " + str(i)
                self.str_curr_dist = "current dist = " + str(dist)
                self.str_curr_temp = "current temperature = " + str(temp)
                # self.str_curr_d = "current d = "+ str(d)
                self.collect_full_likelihood.append(
                    self.simulation.sampler.likelihood_t
                )
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_op_sampled.append(op_sampled)
                self.collect_id_fB_sampled.append(id_f_sampled)
                self.collect_id_fA_sampled.append(i)

                self.collect_dist_from_init_genome.append(dist)
                iter += 1
        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.output_matrix_em
        )
        self.simulation.export_new_fasta()
        self.simulation.plot_info_simu(
            self.collect_likelihood,
            self.collect_n_contigs,
            self.file_n_contigs,
            "n_contigs",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood,
            self.collect_mean_len,
            self.file_mean_len,
            "mean length contigs",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood,
            self.collect_dist_from_init_genome,
            self.file_dist_init_genome,
            "distance from init genome",
        )
        self.save_behaviour_to_txt()

    def save_behaviour_to_txt(self):
        list_file = [
            self.txt_file_mean_len,
            self.txt_file_n_contigs,
            self.txt_file_dist_init_genome,
            self.txt_file_likelihood,
            self.txt_file_fact,
            self.txt_file_slope,
            self.txt_file_d_max,
            self.txt_file_d_nuc,
            self.txt_file_d,
            self.txt_file_success,
        ]
        list_data = [
            self.collect_mean_len,
            self.collect_n_contigs,
            self.collect_dist_from_init_genome,
            self.collect_likelihood,
            self.collect_fact,
            self.collect_slope,
            self.collect_d_max,
            self.collect_d_nuc,
            self.collect_d,
            self.collect_success,
        ]
        for d in range(0, len(list_file)):
            thefile = list_file[d]
            h = open(thefile, "w")
            data = list_data[d]
            for item in data:
                h.write("%s\n" % item)
            h.close()
        f_mutations = open(self.txt_file_list_mutations, "w")
        f_mutations.write("%s\t%s\t%s\n" % ("id_fA", "id_fB", "id_mutation"))
        for i in range(0, len(self.collect_id_fA_sampled)):
            id_fA = self.collect_id_fA_sampled[i]
            id_fB = self.collect_id_fB_sampled[i]
            id_mut = self.collect_op_sampled[i]
            f_mutations.write("%s\t%s\t%s\n" % (id_fA, id_fB, id_mut))
        f_mutations.close()

    def start_MCMC(self,):
        logger.info("set jumping distribution...")
        delta = 5
        self.simulation.sampler.set_jumping_distributions_parameters(delta)
        self.simulation.sampler.init_likelihood()
        logger.info("start sampling launched ... ")
        logger.info((self.simulation.n_iterations))
        delta = list(range(5, 5 + self.simulation.n_iterations * 2, 2))
        logger.info(delta)
        # if self.scrambled:
        #     self.simulation.sampler.modify_genome(500)
        # o, d, d_high =
        # self.simulation.sampler.display_current_matrix(
        # self.simulation.input_matrix
        # )
        n_iter = np.float32(self.simulation.n_iterations)
        list_frags = np.arange(0, self.n_frags, dtype=np.int32)
        for j in range(0, self.n_iterations_mcmc):
            logger.info("cycle = {}".format(j))
            self.str_curr_cycle = "current cycle = " + str(j)
            np.random.shuffle(list_frags)
            for i in list_frags:
                # print "id_frag =", i
                (
                    o,
                    n_contigs,
                    min_len,
                    mean_len,
                    max_len,
                    temp,
                    dist,
                ) = self.simulation.sampler.step_metropolis_hastings_s_a(
                    i, np.float32(j), n_iter, self.dt
                )
                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(n_contigs)
                self.str_curr_id_frag = "current frag = " + str(i)
                self.str_curr_temp = "current temperature = " + str(temp)
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_dist_from_init_genome.append(dist)

        self.simulation.export_new_fasta()
        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.output_matrix_mcmc
        )

        self.simulation.plot_info_simu(
            self.collect_likelihood,
            self.collect_n_contigs,
            self.file_n_contigs,
            "n_contigs",
        )

        self.simulation.plot_info_simu(
            self.collect_likelihood,
            self.collect_mean_len,
            self.file_mean_len,
            "mean length contigs",
        )

        self.simulation.plot_info_simu(
            self.collect_likelihood,
            self.collect_dist_from_init_genome,
            self.file_dist_init_genome,
            "distance from init genome",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood_nuisance,
            self.collect_slope,
            self.file_slope,
            "slope",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood_nuisance,
            self.collect_fact,
            self.file_fact,
            "scale factor",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood_nuisance,
            self.collect_d_nuc,
            self.file_d_nuc,
            "val trans",
        )

        self.save_behaviour_to_txt()

    def start_MTM(self,):
        logger.info("set jumping distribution...")
        delta = 5
        self.simulation.sampler.set_jumping_distributions_parameters(delta)
        self.simulation.sampler.init_likelihood()
        logger.info("start sampling launched ... ")
        logger.info((self.simulation.n_iterations))
        delta = list(range(5, 5 + self.simulation.n_iterations * 2, 2))
        logger.info(delta)
        # if self.scrambled:
        #     self.simulation.sampler.modify_genome(500)
        # o, d, d_high =
        # self.simulation.sampler.display_current_matrix(
        # self.simulation.input_matrix
        # )
        if self.scrambled:
            #     self.simulation.sampler.modify_genome(500)
            self.simulation.sampler.explode_genome(self.dt)
        n_iter = np.float32(self.simulation.n_iterations)
        list_frags = np.arange(0, self.n_frags, dtype=np.int32)
        for j in range(0, self.n_iterations_mcmc):
            logger.info("cycle = {}".format(j))
            self.str_curr_cycle = "current cycle = " + str(j)
            np.random.shuffle(list_frags)
            for i in list_frags:
                # print "id_frag =", i
                (
                    o,
                    n_contigs,
                    min_len,
                    mean_len,
                    max_len,
                    temp,
                    dist,
                ) = self.simulation.sampler.step_mtm(i, np.float32(j), n_iter, self.dt)
                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(n_contigs)
                self.str_curr_id_frag = "current frag = " + str(i)
                self.str_curr_temp = "current temperature = " + str(temp)
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_dist_from_init_genome.append(dist)

        self.simulation.export_new_fasta()
        o, d, d_high = self.simulation.sampler.display_current_matrix(
            self.simulation.output_matrix_mcmc
        )

        self.simulation.plot_info_simu(
            self.collect_likelihood,
            self.collect_n_contigs,
            self.file_n_contigs,
            "n_contigs",
        )

        self.simulation.plot_info_simu(
            self.collect_likelihood,
            self.collect_mean_len,
            self.file_mean_len,
            "mean length contigs",
        )

        self.simulation.plot_info_simu(
            self.collect_likelihood,
            self.collect_dist_from_init_genome,
            self.file_dist_init_genome,
            "distance from init genome",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood_nuisance,
            self.collect_slope,
            self.file_slope,
            "slope",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood_nuisance,
            self.collect_fact,
            self.file_fact,
            "scale factor",
        )
        self.simulation.plot_info_simu(
            self.collect_likelihood_nuisance,
            self.collect_d_nuc,
            self.file_d_nuc,
            "val trans",
        )

        self.save_behaviour_to_txt()

    def setup_simu(self, id_f_ins):
        self.simulation.sampler.insert_repeats(id_f_ins)
        self.simulation.sampler.simulate_rippe_contacts()
        plt.imshow(
            self.simulation.sampler.hic_matrix,
            interpolation="nearest",
            vmin=0,
            vmax=100,
        )
        plt.show()


def main():

    arguments = docopt.docopt(__doc__, version=VERSION_NUMBER)
    # print(arguments)

    project_folder = arguments["<hic_folder>"]
    output_folder = arguments["<output_folder>"]
    reference_fasta = arguments["<reference.fa>"]

    number_cycles = int(arguments["--cycles"])
    level = int(arguments["--level"])
    thresh_factor = float(arguments["--coverage-std"])
    neighborhood = int(arguments["--neighborhood"])
    device = int(arguments["--device"])
    circ = arguments["--circular"]
    bomb = arguments["--bomb"]
    save_matrix = arguments["--save-matrix"]
    simple = arguments["--simple"]
    quiet = arguments["--quiet"]
    debug = arguments["--debug"]
    pyramid_only = arguments["--pyramid-only"]
    pickle_name = arguments["--save-pickle"]

    log_level = logging.INFO

    if quiet:
        log_level = logging.WARNING

    if debug:
        log_level = logging.DEBUG

    logger.setLevel(log_level)

    log.CURRENT_LOG_LEVEL = log_level

    name = os.path.basename(os.path.normpath(project_folder))

    is_simu = False
    scrambled = False

    n_iterations_em = 100
    n_iterations_mcmc = 30
    perform_em = False
    use_rippe = True
    gl_size_im = 1000
    sample_param = True

    if not output_folder:
        output_folder = None

    p2 = instagraal_class(
        name=name,
        folder_path=project_folder,
        fasta=reference_fasta,
        device=device,
        level=level,
        n_iterations_em=DEFAULT_ITERATIONS_EM,
        n_iterations_mcmc=DEFAULT_ITERATIONS_MCMC,
        is_simu=False,
        scrambled=False,
        perform_em=False,
        use_rippe=True,
        sample_param=True,
        thresh_factor=thresh_factor,
        output_folder=output_folder,
    )
    if circ:
        p2.simulation.level.S_o_A_frags["circ"] += 1

    if not pyramid_only:
        if not simple:
            p2.full_em(
                n_cycles=number_cycles,
                n_neighbours=neighborhood,
                bomb=bomb,
                id_start_sample_param=4,
                save_matrix=save_matrix,
            )
        else:
            p2.simple_start(
                n_cycles=number_cycles, n_neighbours=neighborhood, bomb=bomb
            )

    if pickle_name:
        with open("graal.pkl", "wb") as pickle_handle:
            pickle.dump(p2, pickle_handle)

    # p2.ctx_gl.pop()
    # sampler.step_sampler(50)
    # sampler.gpu_vect_frags.copy_from_gpu()
    # max_id = sampler.gpu_vect_frags.id_c.max()
    # frag_a = 0
    # frag_b = 1
    # id_c_a = sampler.gpu_vect_frags.id_c[frag_a]
    # id_c_b = sampler.gpu_vect_frags.id_c[frag_b]
    # print "id_frag a =", frag_a, "id contig = ", id_c_a
    # print "id_frag b =", frag_b, "id contig = ", id_c_b
    # #############################################################
    # sampler.perform_mutations(frag_a, frag_b, max_id, 1 == 0)
    # #############################################################
    # flip_eject = 1
    # sampler.extract_uniq_mutations(frag_a, frag_b, flip_eject)
    # t0 = time.time()
    # l = sampler.eval_likelihood()
    # t1 = time.time()
    # sampler.slice_sparse_mat(id_c_a, id_c_b)
    # sampler.extract_current_sub_likelihood()
    # v_l = sampler.eval_all_sub_likelihood()
    # t2 = time.time()
    # print "single likelihood = ", l
    # print "vect likelihood = ", v_l
    # print "Time single = ", t1 - t0
    # print "Time all_mut = ", t2 - t1
    # print "###################################################"
    # # t0 = time.time()
    # # l = sampler.eval_likelihood()
    # t1 = time.time()
    # # sampler.slice_sparse_mat(id_c_a, id_c_b)
    # v_l = sampler.eval_all_sub_likelihood()
    # t2 = time.time()
    # # print "single likelihood = ", l
    # print "vect likelihood = ", v_l
    # # print "Time single = ", t1 - t0
    # print "Time all_mut = ", t2 - t1

    # n_neighbours = 5
    # # sampler.explode_genome(p2.dt)

    #
    # sampler.bomb_the_genome()


if __name__ == "__main__":
    main()
