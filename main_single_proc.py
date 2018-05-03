__author__ = 'hervemn'

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import pycuda.driver as cuda
#import pycuda.gl as cudagl
#helper modules
import glutil
from vector import Vec
import numpy as np
import matplotlib.pyplot as plt
#from OpenGL.arrays import vbo
from simu_single import simulation
import argparse
# import time
# from pycuda import gpuarray as ga
# import sys, socket
# import pyramid as pyr

class window(object):

    def __init__(self, name, level, n_iterations_em, n_iterations_mcmc, is_simu, scrambled, perform_em, use_rippe,
                 gl_size_im, sample_param, size_pyramid=7):
        #mouse handling for transforming scene
        self.mouse_down = False
        self.size_points = 6
        self.mouse_old = Vec([0., 0.])
        self.rotate = Vec([0., 0., 0.])
        self.translate = Vec([0., 0., 0.])
        self.initrans = Vec([0., 0., -2.])
        self.scrambled = scrambled
        self.width = 400
        self.height = 400
        self.n_iterations_em = n_iterations_em
        self.n_iterations_mcmc = n_iterations_mcmc
        self.sample_param = sample_param
        self.dt = np.float32(0.01)
        self.white = 1
        self.gl_size_im = gl_size_im
        #glutInit(sys.argv)
        #glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        #glutInitWindowSize(self.width, self.height)
        #glutInitWindowPosition(0, 0)
        #if perform_em:
        #    name_window = "Expectation Maximization : " + name
        #else:
        #    name_window = "MCMC Metropolis-Hasting : " + name
        #self.win = glutCreateWindow(name_window)
        # manava = raw_input("do we have a deal?")

        #gets called by GLUT every frame
        #glutDisplayFunc(self.draw)

        #handle user input
        #glutKeyboardFunc(self.on_key)
        #glutMouseFunc(self.on_click)
        #glutMotionFunc(self.on_mouse_motion)

        #this will call draw every 30 ms
        #glutTimerFunc(30, self.timer, 30)

        #setup OpenGL scene
        #self.glinit()
        #setup CUDA
        self.cuda_gl_init()
        #set up initial conditions
        self.use_rippe = use_rippe


        self.simulation = simulation(name,
                                     level,
                                     n_iterations_em,
                                     is_simu,
                                     self,
                                     use_rippe,
                                     gl_size_im,
                                     size_pyramid=size_pyramid)
        self.gl_size_im = self.simulation.gl_size_im
        self.texid = self.simulation.texid
        self.init_n_frags = self.simulation.init_n_frags
        self.pbo_im_buffer = self.simulation.pbo_im_buffer
        self.pos_vbo = self.simulation.pos_vbo
        self.col_vbo = self.simulation.col_vbo
        self.n_frags = self.simulation.sampler.n_new_frags

        self.str_likelihood = "likelihood = " + str(0)
        self.str_n_contigs = "n contigs = " + str(0)
        self.str_curr_id_frag = "current id frag = " + str(0)
        self.str_curr_cycle = "current cycle = " + str(0)
        self.str_curr_temp = "current temperature = " + str(0)
        self.str_curr_dist = "current dist = " + str(0)
        # if perform_em:
        #     self.start_EM()
        # else:
        #     self.start_MCMC()
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
        # self.simulation.sampler.eval_likelihood()
        # h = self.simulation.sampler.gpu_full_expected_lin.get()
        # print "non zero h = ", np.nonzero(h>0)[0]
        # self.simulation.sampler.step_nuisance_parameters(0, 0, 0)
        # self.test_model(665, 600)

        self.file_mean_len = os.path.join(self.simulation.output_folder, 'behaviour_mean_len.pdf')
        self.file_n_contigs = os.path.join(self.simulation.output_folder, 'behaviour_n_contigs.pdf')

        self.file_fact = os.path.join(self.simulation.output_folder, 'behaviour_fact.pdf')
        self.file_slope = os.path.join(self.simulation.output_folder, 'behaviour_slope.pdf')
        self.file_d_nuc = os.path.join(self.simulation.output_folder, 'behaviour_d_nuc.pdf')
        self.file_d = os.path.join(self.simulation.output_folder, 'behaviour_d.pdf')
        self.file_d_max = os.path.join(self.simulation.output_folder, 'behaviour_d_max.pdf')

        self.file_dist_init_genome = os.path.join(self.simulation.output_folder, 'behaviour_dist_init_genome.pdf')

        self.txt_file_mean_len = os.path.join(self.simulation.output_folder, 'list_mean_len.txt')
        self.txt_file_n_contigs = os.path.join(self.simulation.output_folder, 'list_n_contigs.txt')
        self.txt_file_dist_init_genome = os.path.join(self.simulation.output_folder, 'list_dist_init_genome.txt')
        self.txt_file_likelihood = os.path.join(self.simulation.output_folder, 'list_likelihood.txt')

        self.txt_file_fact = os.path.join(self.simulation.output_folder, 'list_fact.txt')
        self.txt_file_slope = os.path.join(self.simulation.output_folder, 'list_slope.txt')
        self.txt_file_d_nuc = os.path.join(self.simulation.output_folder, 'list_d_nuc.txt')
        self.txt_file_d = os.path.join(self.simulation.output_folder, 'list_d.txt')
        self.txt_file_d_max = os.path.join(self.simulation.output_folder, 'list_d_max.txt')
        self.txt_file_success = os.path.join(self.simulation.output_folder, 'list_success.txt')
        self.txt_file_list_mutations = os.path.join(self.simulation.output_folder, 'list_mutations.txt')
        self.file_all_data = os.path.join(self.simulation.output_folder, 'behaviour_all.txt')

        # self.remote_update()
        # id_f_ins = 250
        # self.simulation.sampler.insert_repeats(id_f_ins)
        # self.setup_simu(id_f_ins)
        # self.start_EM()
        # self.start_MCMC()


    def replay_simu(self, file_simu, file_likelihood, file_n_contigs, file_distances):

        h = open(file_simu, 'r')
        all_lines = h.readlines()
        list_id_fA = []
        list_id_fB = []
        list_op_sampled = []
        for i in xrange(1, len(all_lines)):
            line = all_lines[i]
            data = line.split('\t')
            id_fA = int(data[0])
            id_fB = int(data[1])
            id_op = int(data[2])
            list_id_fA.append(id_fA)
            list_id_fB.append(id_fB)
            list_op_sampled.append(id_op)
        h.close()

        h_likeli = open(file_likelihood, 'r')
        all_lines_likeli = h_likeli.readlines()
        list_likelihood = []
        for i in xrange(0, len(all_lines_likeli)):
            line = all_lines_likeli[i]
            likeli = np.float32(line)
            list_likelihood.append(likeli)
        h_likeli.close()

        h_n_contigs = open(file_n_contigs, 'r')
        list_n_contigs = []
        all_lines_contigs = h_n_contigs.readlines()
        for i in xrange(0, len(all_lines_contigs)):
            line = all_lines_contigs[i]
            n_contigs = np.float32(line)
            list_n_contigs.append(n_contigs)
        h_n_contigs.close()

        h_distances = open(file_distances, 'r')
        list_distances = []
        all_lines_distances = h_distances.readlines()
        for i in xrange(0, len(all_lines_distances)):
            line = all_lines_distances[i]
            distance = np.float32(line)
            list_distances.append(distance)
        h_distances.close()

        for i in xrange(0, 10):
            self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
            self.remote_update()
            self.str_n_contigs = "n contigs = " + str(self.simulation.sampler.gpu_vect_frags.id_c.max())

        # self.simulation.sampler.explode_genome(self.dt)
        self.simulation.sampler.bomb_the_genome()
        # print list_likelihood
        for i in xrange(0, len(list_id_fA)):
            self.simulation.sampler.apply_replay_simu(list_id_fA[i], list_id_fB[i], list_op_sampled[i], self.dt)
            self.str_curr_cycle = "cycle = " + str(int(i))
            self.str_likelihood = "likelihood = " + str(list_likelihood[i])
            self.str_n_contigs = "n contigs = " + str(list_n_contigs[i])
            self.str_curr_id_frag = "current frag = "+ str(list_id_fA[i])
            self.str_curr_dist = "current dist = "+ str(list_distances[i])
            self.str_curr_temp = "current temperature = "+str(0)

    def start_EM(self,):
        print "start expectation maximization ... "
        delta = 15
        print self.simulation.n_iterations
        delta = np.int32(np.floor(np.linspace(3, 5, np.floor(self.n_iterations_em / 3.)))) # param ok simu
        # delta = np.int32(np.floor(np.linspace(3, 4, np.floor(self.n_iterations_em / 2.))))
        delta = list(delta)
        d_ext = list(np.floor(np.linspace(10, 15, np.floor(self.n_iterations_em / 3.) + 1))) # param ok simu
        delta.extend(d_ext)
        print delta
        print "len delta = ", len(delta)
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.input_matrix)
        self.simulation.sampler.init_likelihood()
        self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
        ready = 0
        # while ready != '1':
        #     ready = raw_input("ready to start?")
        #     self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
        #     self.remote_update()
        if self.scrambled:
        #     self.simulation.sampler.modify_genome(500)
            self.simulation.sampler.explode_genome(self.dt)
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.scrambled_input_matrix)
        list_frags = np.arange(0, self.simulation.sampler.n_new_frags, dtype=np.int32)
        iter = 0
        n_iter = np.float32(self.n_iterations_em)
        self.bins_rippe = self.simulation.sampler.bins
        for j in xrange(0, self.n_iterations_em):
            print "cycle = ", j
            self.str_curr_cycle = "current cycle = "+ str(j)
            np.random.shuffle(list_frags)
            # d = self.simulation.sampler.step_nuisance_parameters(0, 0, 0)
            for i in list_frags:
                # print "id_frag =", i
                #if bool(glutMainLoopEvent):
                #    glutMainLoopEvent()
                #else:
                #    glutCheckLoop()
                # if j == 0 and iter == 0:
                #     raw_input("ready ?")
                o, n_contigs, min_len, mean_len, max_len, op_sampled, id_f_sampled, dist, temp = self.simulation.sampler.step_max_likelihood(i,
                                                                                                 delta[j],
                                                                                                 512, self.dt,
                                                                                                 np.float32(j), n_iter)

                # o, n_contigs, min_len, mean_len, max_len = self.simulation.sampler.new_sample_fi(i,
                #                                                                                  delta[j],
                #                                                                                  512, 200)
                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(n_contigs)
                self.str_curr_id_frag = "current frag = "+ str(i)
                self.str_curr_dist = "current dist = "+ str(dist)
                self.str_curr_temp = "current temperature = "+str(temp)
                # self.str_curr_d = "current d = "+ str(d)
                self.collect_full_likelihood.append(self.simulation.sampler.likelihood_t)
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_op_sampled.append(op_sampled)
                self.collect_id_fB_sampled.append(id_f_sampled)
                self.collect_id_fA_sampled.append(i)
                self.collect_dist_from_init_genome.append(dist)
                iter += 1
                # sampling nuisance parameters
                fact, d, d_max, d_nuc, slope, likeli, success, y_eval  = self.simulation.sampler.step_nuisance_parameters(self.dt, np.float32(j),
                                                                                                n_iter)
                self.collect_fact.append(fact)
                self.collect_d.append(d)
                self.collect_d_max.append(d_max)
                self.collect_d_nuc.append(d_nuc)
                self.collect_slope.append(slope)
                self.collect_likelihood_nuisance.append(likeli)
                self.collect_success.append(success)
                self.y_eval = y_eval

        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.output_matrix_em)
        self.simulation.export_new_fasta()
        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_n_contigs, self.file_n_contigs,
                                       "n_contigs")
        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_mean_len, self.file_mean_len,
                                       "mean length contigs")
        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_dist_from_init_genome, self.file_dist_init_genome,
                                       "distance from init genome")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_slope, self.file_slope,
                                   "slope")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_fact, self.file_fact,
                                       "scale factor")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d_nuc, self.file_d_nuc,
                           "val trans")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d, self.file_d,
                           "d")
        self.save_behaviour_to_txt()




    def start_EM_all(self,):
        print "start expectation maximization ... "
        delta = 15
        print self.simulation.n_iterations
        delta = np.int32(np.floor(np.linspace(3, 4, np.floor(self.n_iterations_em / 2.)))) # param ok simu
        # delta = np.int32(np.floor(np.linspace(3, 4, np.floor(self.n_iterations_em / 2.))))
        delta = list(delta)
        d_ext = list(np.floor(np.linspace(5, 10, np.floor(self.n_iterations_em / 2.) + 1))) # param ok simu
        # d_ext = list(np.floor(np.linspace(10, 15, np.floor(self.n_iterations_em / 2.) + 1)))
        delta.extend(d_ext)
        print delta
        print "len delta = ", len(delta)
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.input_matrix)
        self.simulation.sampler.init_likelihood()
        self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
        if self.scrambled:
            self.simulation.sampler.explode_genome(self.dt)
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.scrambled_input_matrix)
        list_frags = np.arange(0, self.simulation.sampler.n_new_frags, dtype=np.int32)
        iter = 0
        n_iter = np.float32(self.n_iterations_em)
        self.bins_rippe = self.simulation.sampler.bins
        self.headers = ['likelihood', 'n_contigs', 'min_len_contigs', 'mean_len_contigs', 'max_len_contigs',
                        'operation_sampled', 'id_f_sampled', 'distance_from_init',
                        'scale factor', 'd', 'max_distance_intra', 'n_contacts_inter', 'slope', 'success']

        fact, d, d_max, d_nuc, slope, likeli, success, y_eval  = self.simulation.sampler.step_nuisance_parameters(self.dt,
                                                                                                                  np.float32(0),
                                                                                                                  n_iter)
        for j in xrange(0, self.n_iterations_em):
            print "cycle = ", j
            self.str_curr_cycle = "current cycle = "+ str(j)
            np.random.shuffle(list_frags)
            # d = self.simulation.sampler.step_nuisance_parameters(0, 0, 0)
            for i in list_frags:
                # print "id_frag =", i
                #if bool(glutMainLoopEvent):
                #    glutMainLoopEvent()
                #else:
                #    glutCheckLoop()
                # if j == 0 and iter == 0:
                #     raw_input("ready ?")
                o, n_contigs, min_len, mean_len, max_len, op_sampled, id_f_sampled, dist, temp = self.simulation.sampler.step_max_likelihood(i,
                                                                                                 delta[j],
                                                                                                 512, self.dt,
                                                                                                 np.float32(j), n_iter)

                success = 1
                vect_collect = (o, n_contigs, min_len, mean_len, max_len, op_sampled, id_f_sampled, dist,
                                fact, d, d_max, d_nuc, slope, success)


                self.collect_all.append(vect_collect)

                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(n_contigs)
                self.str_curr_id_frag = "current frag = "+ str(i)
                self.str_curr_dist = "current dist = "+ str(dist)
                self.str_curr_temp = "current temperature = "+str(temp)
                # self.str_curr_d = "current d = "+ str(d)
                self.collect_full_likelihood.append(self.simulation.sampler. likelihood_t)
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_op_sampled.append(op_sampled)
                self.collect_id_fB_sampled.append(id_f_sampled)
                self.collect_id_fA_sampled.append(i)
                self.collect_dist_from_init_genome.append(dist)
                iter += 1
                # sampling nuisance parameters
                fact, d, d_max, d_nuc, slope, likeli, success, y_eval  = self.simulation.sampler.step_nuisance_parameters(self.dt, np.float32(j),
                                                                                                n_iter)
                vect_collect = (likeli, n_contigs, min_len, mean_len, max_len, op_sampled, id_f_sampled, dist,
                                fact, d, d_max, d_nuc, slope, success)

                self.collect_all.append(vect_collect)
                self.collect_fact.append(fact)
                self.collect_d.append(d)
                self.collect_d_max.append(d_max)
                self.collect_d_nuc.append(d_nuc)
                self.collect_slope.append(slope)
                self.collect_likelihood_nuisance.append(likeli)
                self.collect_success.append(success)
                self.y_eval = y_eval

        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.output_matrix_em)
        self.simulation.export_new_fasta()
        self.simulation.plot_all_info_simu(self.collect_all, self.header)


    def start_EM_nuisance(self,):
        print "start expectation maximization ... "
        delta = 15
        print self.simulation.n_iterations
        delta = np.int32(np.floor(np.linspace(3, 4, np.floor(self.n_iterations_em / 2.)))) # param ok simu
        # delta = np.int32(np.floor(np.linspace(3, 4, np.floor(self.n_iterations_em / 2.))))
        delta = list(delta)
        d_ext = list(np.floor(np.linspace(5, 10, np.floor(self.n_iterations_em / 2.) + 1))) # param ok simu
        # d_ext = list(np.floor(np.linspace(10, 15, np.floor(self.n_iterations_em / 2.) + 1)))
        delta.extend(d_ext)
        print delta
        print "len delta = ", len(delta)
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.input_matrix)
        self.simulation.sampler.init_likelihood()
        self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
        ready = 0
        # while ready != '1':
        #     ready = raw_input("ready to start?")
        #     self.simulation.sampler.modify_gl_cuda_buffer(0, self.dt)
        #     self.remote_update()
        if self.scrambled:
        #     self.simulation.sampler.modify_genome(500)
            self.simulation.sampler.explode_genome(self.dt)
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.scrambled_input_matrix)
        list_frags = np.arange(0, self.simulation.sampler.n_new_frags, dtype=np.int32)
        iter = 0
        n_iter = np.float32(self.n_iterations_em)
        self.bins_rippe = self.simulation.sampler.bins
        for j in xrange(0, self.n_iterations_em):
            print "cycle = ", j
            self.str_curr_cycle = "current cycle = "+ str(j)
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
                # o, n_contigs, min_len, mean_len, max_len, op_sampled, id_f_sampled, dist, temp = self.simulation.sampler.step_max_likelihood(i,
                #                                                                                  delta[j],
                #                                                                                  512, self.dt,
                #                                                                                  np.float32(j), n_iter)
                #
                # # o, n_contigs, min_len, mean_len, max_len = self.simulation.sampler.new_sample_fi(i,
                # #                                                                                  delta[j],
                # #                                                                                  512, 200)
                # self.str_likelihood = "likelihood = " + str(o)
                # self.str_n_contigs = "n contigs = " + str(n_contigs)
                # self.str_curr_id_frag = "current frag = "+ str(i)
                # self.str_curr_dist = "current dist = "+ str(dist)
                # self.str_curr_temp = "current temperature = "+str(temp)
                # # self.str_curr_d = "current d = "+ str(d)
                # self.collect_full_likelihood.append(self.simulation.sampler. likelihood_t)
                # self.collect_likelihood.append(o)
                # self.collect_n_contigs.append(n_contigs)
                # self.collect_mean_len.append(mean_len)
                # self.collect_op_sampled.append(op_sampled)
                # self.collect_id_fB_sampled.append(id_f_sampled)
                # self.collect_id_fA_sampled.append(i)
                # self.collect_dist_from_init_genome.append(dist)
                # iter += 1
                # # sampling nuisance parameters
                fact, d, d_max, d_nuc, slope, likeli, success, y_eval  = self.simulation.sampler.step_nuisance_parameters(self.dt, np.float32(j),
                                                                                                n_iter)
                self.collect_fact.append(fact)
                self.collect_d.append(d)
                self.collect_d_max.append(d_max)
                self.collect_d_nuc.append(d_nuc)
                self.collect_slope.append(slope)
                self.collect_likelihood_nuisance.append(likeli)
                self.collect_success.append(success)
                self.y_eval = y_eval

        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.output_matrix_em)
        self.simulation.export_new_fasta()
        # self.simulation.plot_info_simu(self.collect_likelihood, self.collect_n_contigs, self.file_n_contigs,
        #                                "n_contigs")
        # self.simulation.plot_info_simu(self.collect_likelihood, self.collect_mean_len, self.file_mean_len,
        #                                "mean length contigs")
        # self.simulation.plot_info_simu(self.collect_likelihood, self.collect_dist_from_init_genome, self.file_dist_init_genome,
        #                                "distance from init genome")
        # self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_slope, self.file_slope,
        #                            "slope")
        # self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_fact, self.file_fact,
        #                                "scale factor")
        # self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d_nuc, self.file_d_nuc,
        #                    "val trans")
        # self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d, self.file_d,
        #                    "d")
        # self.save_behaviour_to_txt()



    def start_EM_no_scrambled(self,):
        print "start expectation maximization ... "
        delta = 15
        print self.simulation.n_iterations
        # delta = np.int32(np.floor(np.linspace(3, 4, np.floor(self.n_iterations_em / 2.)))) # param ok simu
        delta = np.int32(np.floor(np.linspace(3, 4, np.floor(self.n_iterations_em / 2.))))
        delta = list(delta)
        # d_ext = list(np.floor(np.linspace(5, 10, np.floor(self.n_iterations_em / 2.) + 1))) # param ok simu
        d_ext = list(np.floor(np.linspace(10, 15, np.floor(self.n_iterations_em / 2.) + 1)))
        delta.extend(d_ext)
        print delta
        print "len delta = ", len(delta)
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.input_matrix)
        # self.simulation.sampler.init_likelihood()

        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.scrambled_input_matrix)
        list_frags = np.arange(0, self.simulation.sampler.n_new_frags, dtype=np.int32)
        iter = 0
        n_iter = np.float32(self.n_iterations_em)
        for j in xrange(0, self.n_iterations_em):
            print "cycle = ", j
            self.str_curr_cycle = "current cycle = "+ str(j)
            np.random.shuffle(list_frags)
            # d = self.simulation.sampler.step_nuisance_parameters(0, 0, 0)
            for i in list_frags:
                # print "id_frag =", i
                #if bool(glutMainLoopEvent):
                #    glutMainLoopEvent()
                #else:
                #    glutCheckLoop()
                # if j == 0 and iter == 0:
                #     raw_input("ready ?")
                o, n_contigs, min_len, mean_len, max_len, op_sampled, id_f_sampled, dist, temp = self.simulation.sampler.step_max_likelihood(i,
                                                                                                 delta[j],
                                                                                                 512, self.dt,
                                                                                                 np.float32(j), n_iter)
                # o, n_contigs, min_len, mean_len, max_len = self.simulation.sampler.new_sample_fi(i,
                #                                                                                  delta[j],
                #                                                                                  512, 200)
                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(n_contigs)
                self.str_curr_id_frag = "current frag = "+ str(i)
                self.str_curr_dist = "current dist = "+ str(dist)
                self.str_curr_temp = "current temperature = "+str(temp)
                # self.str_curr_d = "current d = "+ str(d)
                self.collect_full_likelihood.append(self.simulation.sampler. likelihood_t)
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_op_sampled.append(op_sampled)
                self.collect_id_fB_sampled.append(id_f_sampled)
                self.collect_id_fA_sampled.append(i)

                self.collect_dist_from_init_genome.append(dist)
                iter += 1
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.output_matrix_em)
        self.simulation.export_new_fasta()
        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_n_contigs, self.file_n_contigs,
                                       "n_contigs")
        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_mean_len, self.file_mean_len,
                                       "mean length contigs")
        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_dist_from_init_genome, self.file_dist_init_genome,
                                       "distance from init genome")
        self.save_behaviour_to_txt()

    def save_behaviour_to_txt(self):
        list_file = [self.txt_file_mean_len, self.txt_file_n_contigs, self.txt_file_dist_init_genome, self.txt_file_likelihood,
                     self.txt_file_fact, self.txt_file_slope, self.txt_file_d_max, self.txt_file_d_nuc, self.txt_file_d,
                     self.txt_file_success]
        list_data = [self.collect_mean_len, self.collect_n_contigs, self.collect_dist_from_init_genome, self.collect_likelihood,
                     self.collect_fact, self.collect_slope, self.collect_d_max, self.collect_d_nuc, self.collect_d,
                     self.collect_success]
        for d in range(0, len(list_file)):
            thefile = list_file[d]
            h = open(thefile, 'w')
            data = list_data[d]
            for item in data:
                h.write("%s\n" % item)
            h.close()
        f_mutations = open(self.txt_file_list_mutations, 'w')
        f_mutations.write("%s\t%s\t%s\n"%("id_fA", "id_fB", "id_mutation"))
        for i in xrange(0, len(self.collect_id_fA_sampled)):
            id_fA = self.collect_id_fA_sampled[i]
            id_fB = self.collect_id_fB_sampled[i]
            id_mut = self.collect_op_sampled[i]
            f_mutations.write("%s\t%s\t%s\n"%(id_fA, id_fB, id_mut))
        f_mutations.close()

    def start_MCMC(self,):
        print "set jumping distribution..."
        delta = 5
        self.simulation.sampler.set_jumping_distributions_parameters(delta)
        self.simulation.sampler.init_likelihood()
        print "start sampling launched ... "
        print self.simulation.n_iterations
        delta = range(5, 5 + self.simulation.n_iterations * 2, 2)
        print delta
        # if self.scrambled:
        #     self.simulation.sampler.modify_genome(500)
        # o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.input_matrix)
        n_iter = np.float32(self.simulation.n_iterations)
        list_frags = np.arange(0, self.n_frags, dtype=np.int32)
        for j in xrange(0, self.n_iterations_mcmc):
            print "cycle = ", j
            self.str_curr_cycle = "current cycle = "+ str(j)
            np.random.shuffle(list_frags)
            for i in list_frags:
                # print "id_frag =", i
                #if bool(glutMainLoopEvent):
                #    glutMainLoopEvent()
                #else:
                #    glutCheckLoop()
                o, n_contigs, min_len, mean_len, max_len, temp, dist = self.simulation.sampler.step_metropolis_hastings_s_a(i, np.float32(j), n_iter, self.dt)
                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(n_contigs)
                self.str_curr_id_frag = "current frag = "+ str(i)
                self.str_curr_temp = "current temperature = "+ str(temp)
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_dist_from_init_genome.append(dist)

        self.simulation.export_new_fasta()
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.output_matrix_mcmc)

        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_n_contigs, self.file_n_contigs,
                                       "n_contigs")

        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_mean_len, self.file_mean_len,
                                       "mean length contigs")

        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_dist_from_init_genome, self.file_dist_init_genome,
                                       "distance from init genome")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_slope, self.file_slope,
                                       "slope")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_fact, self.file_fact,
                                       "scale factor")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d_nuc, self.file_d_nuc,
                           "val trans")

        self.save_behaviour_to_txt()


    def start_MTM(self,):
        print "set jumping distribution..."
        delta = 5
        self.simulation.sampler.set_jumping_distributions_parameters(delta)
        self.simulation.sampler.init_likelihood()
        print "start sampling launched ... "
        print self.simulation.n_iterations
        delta = range(5, 5 + self.simulation.n_iterations * 2, 2)
        print delta
        # if self.scrambled:
        #     self.simulation.sampler.modify_genome(500)
        # o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.input_matrix)
        if self.scrambled:
        #     self.simulation.sampler.modify_genome(500)
            self.simulation.sampler.explode_genome(self.dt)
        n_iter = np.float32(self.simulation.n_iterations)
        list_frags = np.arange(0, self.n_frags, dtype=np.int32)
        for j in xrange(0, self.n_iterations_mcmc):
            print "cycle = ", j
            self.str_curr_cycle = "current cycle = "+ str(j)
            np.random.shuffle(list_frags)
            for i in list_frags:
                # print "id_frag =", i
                #if bool(glutMainLoopEvent):
                #    glutMainLoopEvent()
                #else:
                #    glutCheckLoop()
                o, n_contigs, min_len, mean_len, max_len, temp, dist = self.simulation.sampler.step_mtm(i, np.float32(j), n_iter, self.dt)
                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(n_contigs)
                self.str_curr_id_frag = "current frag = "+ str(i)
                self.str_curr_temp = "current temperature = "+ str(temp)
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_dist_from_init_genome.append(dist)

        self.simulation.export_new_fasta()
        o, d, d_high = self.simulation.sampler.display_current_matrix(self.simulation.output_matrix_mcmc)

        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_n_contigs, self.file_n_contigs,
                                       "n_contigs")

        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_mean_len, self.file_mean_len,
                                       "mean length contigs")

        self.simulation.plot_info_simu(self.collect_likelihood, self.collect_dist_from_init_genome, self.file_dist_init_genome,
                                       "distance from init genome")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_slope, self.file_slope,
                                       "slope")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_fact, self.file_fact,
                                       "scale factor")
        self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d_nuc, self.file_d_nuc,
                           "val trans")

        self.save_behaviour_to_txt()


    def remote_update(self):
        pass
        #if bool(glutMainLoopEvent):
        #    glutMainLoopEvent()
        #else:
        #    glutCheckLoop()

    def setup_simu(self, id_f_ins):
        self.simulation.sampler.insert_repeats(id_f_ins)
        self.simulation.sampler.simulate_rippe_contacts()
        plt.imshow(self.simulation.sampler.hic_matrix, interpolation='nearest', vmin=0, vmax=100)
        plt.show()

    def test_model(self, id_fi, delta):
        id_fi = np.int32(id_fi)
        id_neighbors = np.copy(self.simulation.sampler.return_neighbours(id_fi, delta))
        id_neighbors.sort()
        np.sort(id_neighbors)
        print "physic model = ", self.simulation.sampler.param_simu
        j = 0
        n_iter = self.n_iterations_em
        self.simulation.sampler.step_max_likelihood(id_fi, delta, 512, self.dt, np.float32(j), n_iter)
        nscore = np.copy(self.simulation.sampler.score)
        plt.figure()
        plt.plot(id_neighbors, nscore[range(0, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(1, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(2, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(3, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(4, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(5, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(6, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(7, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        # plt.plot(id_neighbors, nscore[range(12, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)

        plt.legend([self.simulation.sampler.modification_str[i] for i in range(0, 8)])
        plt.ylabel('log likelihood')
        plt.xlabel('fragments id')
        # plt.show()
        plt.figure()
        for i in range(0, self.simulation.sampler.n_tmp_struct):
            print i
            plt.plot(id_neighbors, nscore[range(i, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.legend(self.simulation.sampler.modification_str)
        plt.show()

    def debug_test_model(self, id_fi, delta):
        id_fi = np.int32(id_fi)
        id_neighbors = np.copy(self.simulation.sampler.return_neighbours(id_fi, delta))
        id_neighbors.sort()
        np.sort(id_neighbors)
        print "physic model = ", self.simulation.sampler.param_simu
        self.simulation.sampler.debug_step_max_likelihood(id_fi, delta, 512, self.dt)
        nscore = np.copy(self.simulation.sampler.score)
        plt.figure()
        plt.plot(id_neighbors, nscore[range(0, len(nscore), self.simulation.sampler.n_tmp_struct)], '-', markersize=10)
        plt.plot(id_neighbors, nscore[range(1, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.plot(id_neighbors, nscore[range(2, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.plot(id_neighbors, nscore[range(3, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.plot(id_neighbors, nscore[range(4, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.plot(id_neighbors, nscore[range(5, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.plot(id_neighbors, nscore[range(6, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.legend([self.simulation.sampler.modification_str[i] for i in range(0, 7)])
        # plt.show()
        plt.figure()
        for i in range(0, self.simulation.sampler.n_tmp_struct):
            print i
            plt.plot(id_neighbors, nscore[range(i, len(nscore), self.simulation.sampler.n_tmp_struct)], '-*', markersize=10)
        plt.legend(self.simulation.sampler.modification_str)
        plt.show()

    def debug_test_EM(self, delta):
        for id_fA in xrange(0, self.simulation.sampler.n_new_frags):
            self.simulation.sampler.debug_step_max_likelihood(id_fA, delta, 512, self.dt)


    def cuda_gl_init(self,):
        cuda.init()
        if False:
            print "----SELECT DISPLAY DEVICE----"
            num_gpu = cuda.Device.count()
            for i in range(0,num_gpu):
                tmp_dev = cuda.Device(i)
                print "device_id = ",i,tmp_dev.name()
            id_gpu = raw_input("Select GPU: ")
            # id_gpu = 0
            id_gpu = int(id_gpu)
            curr_gpu = cuda.Device(id_gpu)
            print "you have selected ", curr_gpu.name()
            self.ctx_gl = cudagl.make_context(curr_gpu, flags=cudagl.graphics_map_flags.NONE)
        else:
            #import pycuda.gl.autoinit
            #curr_gpu = cudagl.autoinit.device
            curr_gpu = cuda.Device(0)
            self.ctx_gl = curr_gpu.make_context()

    def glut_print(self, x,  y,  font,  text, r,  g , b , a):

        blending = False
        if glIsEnabled(GL_BLEND) :
            blending = True

        glEnable(GL_BLEND)
        glColor3f(r, g, b)
        glRasterPos2f(x, y)
        for ch in text :
            glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )


        if not blending :
            glDisable(GL_BLEND)

    # def glinit(self):
    #     glViewport(0, 0, self.width, self.height)
    #     glMatrixMode(GL_PROJECTION)
    #     glLoadIdentity()
    #     gluPerspective(60., self.width / float(self.height), .1, 1000.)
    #     glMatrixMode(GL_MODELVIEW)


    ###GL CALLBACKS
    def timer(self, t):
        glutTimerFunc(t, self.timer, t)
        glutPostRedisplay()

    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE:
            self.simulation.release()
            sys.exit()
        elif args[0] == 'p':
            self.size_points += 1
        elif args[0] == 'm':
            self.size_points -= 1
        elif args[0] == 's':
            self.start_EM()
        elif args[0] == 'w':
            self.white *= -1
        # elif args[0] == 'd':
        #     self.simulation.sampler.thresh += 1
        # elif args[0] == 'c':
        #     self.simulation.sampler.thresh -=1
        elif args[0] == 'b':
            self.modify_image_thresh(-1)

        elif args[0] == 'd':
            self.modify_image_thresh(1)

        # elif args[0] == 'e':
        #     self.simulation.plot_info_simu(self.collect_likelihood, self.collect_n_contigs, self.file_n_contigs,
        #                                    "n_contigs")
        #     self.simulation.plot_info_simu(self.collect_likelihood, self.collect_mean_len, self.file_mean_len,
        #                                    "mean length contigs")
        #     self.simulation.plot_info_simu(self.collect_likelihood, self.collect_dist_from_init_genome, self.file_dist_init_genome,
        #                                "distance from init genome")
        #     self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_slope, self.file_slope,
        #                                "slope")
        #     self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_fact, self.file_fact,
        #                                "scale factor")
        #     self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d_nuc, self.file_d_nuc,
        #                    "val trans")
        #     self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d, self.file_d,
        #                    "d")
        #     self.simulation.plot_info_simu(self.collect_likelihood_nuisance, self.collect_d_max, self.file_d_max,
        #                    "dist max intra")
        #     plt.figure()
        #     plt.loglog(self.bins_rippe, self.y_eval, '-b')
        #     plt.loglog(self.bins_rippe, self.simulation.sampler.mean_contacts, '-*r')
        #     plt.xlabel("genomic separation ( kb)")
        #     plt.ylabel("n contacts")
        #     plt.title("rippe curve")
        #     plt.legend(["fit", "obs"])
        #     plt.show()
        #     # self.simulation.sampler.display_modif_vect(0, 0, -1, 100)
        #     self.save_behaviour_to_txt()


    def on_click(self, button, state, x, y):
        if state == GLUT_DOWN:
            self.mouse_down = True
            self.button = button
        else:
            self.mouse_down = False
        self.mouse_old.x = x
        self.mouse_old.y = y


    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old.x
        dy = y - self.mouse_old.y
        if self.mouse_down and self.button == 0: #left button
            self.rotate.x += dy * .2
            self.rotate.y += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            self.translate.z -= dy * .01
        self.mouse_old.x = x
        self.mouse_old.y = y
    ###END GL CALLBACKS

    def modify_image_thresh(self, val):
        """ modify threshold of the matrix """
        self.simulation.sampler.modify_image_thresh(val)
        # print "threshold = ", self.simulation.sampler.im_thresh

    def draw(self):
        """Render the particles"""
        #update or particle positions by calling the OpenCL kernel
        # self.cle.execute(10)
        glFlush()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.white == -1:
            glClearColor(1, 1, 1, 1)
        else:
            glClearColor(0, 0, 0, 0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #handle mouse transformations
        glTranslatef(self.initrans.x, self.initrans.y, self.initrans.z)
        glRotatef(self.rotate.x, 1, 0, 0)
        glRotatef(self.rotate.y, 0, 1, 0) #we switched around the axis so make this rotate_z
        glTranslatef(self.translate.x, self.translate.y, self.translate.z)


        # render the matrix
        self.render_image()
        #render the particles
        self.render()

        #draw the x, y and z axis as lines
        glutil.draw_axes()
        ############## enable 2d display #######################
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0.0, self.width, self.height, 0.0, -1.0, 10.0)
        glMatrixMode(GL_MODELVIEW)
        # glPushMatrix()        ----Not sure if I need this
        glLoadIdentity()
        glDisable(GL_CULL_FACE)

        glClear(GL_DEPTH_BUFFER_BIT)

        # glBegin(GL_LINES)
        # glColor3f(0.0, 1.0, 0.0)
        # glVertex2f(0.0, 0.0)
        # glVertex2f(100.0, 0.0)
        # glVertex2f(100.0, 100.0)
        # glVertex2f(0.0, 100.0)
        # glEnd()

        if self.white == 1:
            self.glut_print( 10 , 15 , GLUT_BITMAP_9_BY_15 , self.str_curr_cycle , 0.0 , 1.0 , 0.0 , 1.0)
            self.glut_print( 10 , 30 , GLUT_BITMAP_9_BY_15 , self.str_curr_temp , 0.0 , 1.0 , 0.0 , 1.0)
            self.glut_print( 10 , 45 , GLUT_BITMAP_9_BY_15 , self.str_curr_id_frag , 0.0 , 1.0 , 0.0 , 1.0)
            self.glut_print( 10 , 60 , GLUT_BITMAP_9_BY_15 , self.str_likelihood , 0.0 , 1.0 , 0.0 , 1.0)
            self.glut_print( 10 , 75 , GLUT_BITMAP_9_BY_15 , self.str_n_contigs , 0.0 , 1.0 , 0.0 , 1.0)
            self.glut_print( 10 , 90 , GLUT_BITMAP_9_BY_15 , self.str_curr_dist , 0.0 , 1.0 , 0.0 , 1.0)
        else:
            self.glut_print( 10 , 15 , GLUT_BITMAP_9_BY_15 , self.str_curr_cycle , 0.0 , 0.0 , 0.0 , 1.0)
            self.glut_print( 10 , 30 , GLUT_BITMAP_9_BY_15 , self.str_curr_temp , 0.0 , 0.0 , 0.0 , 1.0)
            self.glut_print( 10 , 45 , GLUT_BITMAP_9_BY_15 , self.str_curr_id_frag , 0.0 , 0.0 , 0.0 , 1.0)
            self.glut_print( 10 , 60 , GLUT_BITMAP_9_BY_15 , self.str_likelihood , 0.0 , 0.0 , 0.0 , 1.0)
            self.glut_print( 10 , 75 , GLUT_BITMAP_9_BY_15 , self.str_n_contigs , 0.0 , 0.0 , 0.0 , 1.0)
            self.glut_print( 10 , 90 , GLUT_BITMAP_9_BY_15 , self.str_curr_dist , 0.0 , 0.0 , 0.0 , 1.0)

        # self.glut_print( 10 , 15 , GLUT_BITMAP_9_BY_15 , self.str_curr_cycle , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 10 , 30 , GLUT_BITMAP_9_BY_15 , self.str_curr_temp , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 10 , 45 , GLUT_BITMAP_9_BY_15 , self.str_curr_id_frag , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 10 , 60 , GLUT_BITMAP_9_BY_15 , self.str_likelihood , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 10.1 , 60.1 , GLUT_BITMAP_9_BY_15 , self.str_likelihood , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 9.9 , 59.9, GLUT_BITMAP_9_BY_15 , self.str_likelihood , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 10 , 75 , GLUT_BITMAP_9_BY_15 , self.str_n_contigs , 1.0 , 1.0 , 1.0 , 1.0)
        # self.glut_print( 10 , 90 , GLUT_BITMAP_9_BY_15 , self.str_curr_dist , 1.0 , 1.0 , 1.0 , 1.0)


        # self.glut_print( 10 , 15 , GLUT_BITMAP_9_BY_15 , self.str_curr_cycle , 0.0 , 0.0 , 0.0 , 1.0)
        # self.glut_print( 10 , 30 , GLUT_BITMAP_9_BY_15 , self.str_curr_temp , 0.0 , 0.0 , 0.0 , 1.0)
        # self.glut_print( 10 , 45 , GLUT_BITMAP_9_BY_15 , self.str_curr_id_frag , 0.0 , 0.0 , 0.0 , 1.0)
        # self.glut_print( 10 , 60 , GLUT_BITMAP_9_BY_15 , self.str_likelihood , 0.0 , 0.0 , 0.0 , 1.0)
        # self.glut_print( 10 , 75 , GLUT_BITMAP_9_BY_15 , self.str_n_contigs , 0.0 , 0.0 , 0.0 , 1.0)
        # self.glut_print( 10 , 90 , GLUT_BITMAP_9_BY_15 , self.str_curr_dist , 0.0 , 0.0 , 0.0 , 1.0)



        ## Making sure we can render 3d again
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        # glPopMatrix()##        ----and this?

        #############
        glutSwapBuffers()


    def render(self):


        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glBegin(GL_POINTS)

        # glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        # glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        # glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
        #
        glPointSize(self.size_points)
        # glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        #setup the VBOs
        self.col_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, self.col_vbo)
        self.pos_vbo.bind()
        glVertexPointer(4, GL_FLOAT, 0, self.pos_vbo)


        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        #draw the VBOs
        glDrawArrays(GL_POINTS, 0, int(self.simulation.sampler.n_new_frags))

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        # glEnd(GL_POINTS)

        glDisable(GL_BLEND)

    def render_image(self):
        blending = False
        if glIsEnabled(GL_BLEND):
            blending = True
        else:
            glEnable(GL_BLEND)
        glColor4f(1, 1, 1, 1)

        glEnable(GL_TEXTURE_2D)

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_im_buffer)
        glBindTexture(GL_TEXTURE_2D, self.texid)


        bsize = glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE)

        #Copyng from buffer to texture
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.gl_size_im, self.gl_size_im, GL_LUMINANCE, GL_UNSIGNED_BYTE, None)

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0) # Unbind

        #glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        #

        glBegin(GL_QUADS)
        # glVertex2f(0-1, 0-1)
        # glTexCoord2f(0-1, 0-1)
        # glVertex2f(0-1, 1-1)
        # glTexCoord2f(1-1, 0-1)
        # glVertex2f(1-1, 1-1)
        # glTexCoord2f(1-1, 1-1)
        # glVertex2f(1-1, 0-1)
        # glTexCoord2f(0-1, 1-1)

        glVertex2f(-1, 0)
        glTexCoord2f(-1, 0)

        glVertex2f(0, 0)
        glTexCoord2f(0, 0)

        glVertex2f(0, 1)
        glTexCoord2f(0, 1)

        glVertex2f(-1, 1)
        glTexCoord2f(-1, 1)



        # rotation 45 deg
        # glVertex2f(-1, 0.)
        # glTexCoord2f(0, 0)
        #
        # glVertex2f(0, 1)
        # glTexCoord2f(0, 1)
        #
        # glVertex2f(1, 0.)
        # glTexCoord2f(1, 1)
        #
        # glVertex2f(0, -1)
        # glTexCoord2f(1, 0)


        glEnd()
        # glBindTexture(GL_TEXTURE_2D, 0)
        # glutSwapBuffers()
        # glutPostRedisplay()
        glDisable(GL_TEXTURE_2D)
        # glEnable(GL_DEPTH_TEST)

        if not blending :
            glDisable(GL_BLEND)

    def simple_start(self, n_cycles, n_neighbours, bomb):
        sampler = self.simulation.sampler
        if bomb:
            sampler.bomb_the_genome()
        sampler.gpu_vect_frags.copy_from_gpu()
        list_frags_extremities = list(np.nonzero(sampler.gpu_vect_frags.prev == -1)[0])
        # n_neighbours = 5

        for j in xrange(0, n_cycles):
            sampler.gpu_vect_frags.copy_from_gpu()
            if j > 0:
                list_frags_extremities = list(np.nonzero(sampler.gpu_vect_frags.prev == -1)[0])
                id_extrem_right = list(np.nonzero(sampler.gpu_vect_frags.next == -1)[0])
                list_frags_extremities.extend(id_extrem_right)
            list_frags_extremities = np.array(list_frags_extremities, dtype=np.int32)
            np.random.shuffle(list_frags_extremities)
            print "cycle = ", j
            # np.random.shuffle(list_frags)
            for id_frag in list_frags_extremities:
                #if bool(glutMainLoopEvent):
                #    glutMainLoopEvent()
                #else:
                #    glutCheckLoop()
                o, dist, op_sampled, id_f_sampled, mean_len, n_contigs = sampler.step_sampler(id_frag, n_neighbours, p2.dt)
                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(sampler.n_contigs)
                self.str_curr_id_frag = "current frag = "+ str(id_frag)
                self.str_curr_dist = "current dist = "+ str(dist)
                self.str_curr_cycle = "current cycle = " + str(j)


    def full_em(self, n_cycles, n_neighbours, bomb, id_start_sample_param):
        sampler = self.simulation.sampler
        if bomb:
            sampler.bomb_the_genome()
        list_frags = np.arange(0, sampler.n_new_frags)
        t =0
        n_iter = n_cycles * sampler.n_new_frags
        for j in xrange(0, n_cycles):
            sampler.gpu_vect_frags.copy_from_gpu()

            np.random.shuffle(list_frags)
            print "cycle = ", j
            # np.random.shuffle(list_frags)
            for id_frag in list_frags:
                #if bool(glutMainLoopEvent):
                #    glutMainLoopEvent()
                #else:
                #    glutCheckLoop()

                o, dist, op_sampled, id_f_sampled, mean_len, n_contigs = sampler.step_sampler(id_frag, n_neighbours, p2.dt)
                self.collect_likelihood.append(o)
                self.collect_n_contigs.append(n_contigs)
                self.collect_mean_len.append(mean_len)
                self.collect_op_sampled.append(op_sampled)
                self.collect_id_fB_sampled.append(id_f_sampled)
                self.collect_id_fA_sampled.append(id_frag)
                self.collect_dist_from_init_genome.append(dist)

                self.str_likelihood = "likelihood = " + str(o)
                self.str_n_contigs = "n contigs = " + str(sampler.n_contigs)
                self.str_curr_id_frag = "current frag = "+ str(id_frag)
                self.str_curr_dist = "current dist = "+ str(dist)
                self.str_curr_cycle = "current cycle = " + str(j)
                if self.sample_param and j> id_start_sample_param:
                    fact, d, d_max, d_nuc, slope, self.likelihood_t_nuis, success, y_rippe = sampler.step_nuisance_parameters(p2.dt, t, n_iter)

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
            file_out = os.path.join(self.simulation.output_folder, 'save_simu_step_' + str(j)+ '.txt')
            h = open(file_out, 'w')
            for pos, start_bp, id_c, ori in zip(c.pos, c.start_bp, c.id_c, c.ori):
                h.write(str(pos) + '\t' + str(start_bp) + '\t' + str(id_c) + '\t' + str(ori) + '\n')
            h.close()
            try:
                self.simulation.export_new_fasta()
            except:
                print("Warning, could not write fasta file at cycle "+str(j))
        self.save_behaviour_to_txt()


if __name__ == "__main__":
    # name = 'tricho_qm6a_sparse'
    # name = 's1_de_novo

    perform_em = False
    sample_param = True

    DEFAULT_CYCLES = 20
    DEFAULT_LEVEL = 5
    DEFAULT_NEIGHBOURS = 5
    DEFAULT_BOMB = True
    DEFAULT_ITERATIONS_MCMC = 100
    DEFAULT_ITERATIONS_EM = 30
    DEFAULT_BURN_IN_CYCLES = 2

    parser = argparse.ArgumentParser(description='Perform serpentin binning.')
    parser.add_argument('-p', '--project', help='Project name', required=True, type=str)
    parser.add_argument('-c', '--cycles', help='Number of cycles', type=int, default=DEFAULT_CYCLES)
    parser.add_argument('-l', '--level', help='Pyramid level', type=int,
                        default=DEFAULT_LEVELS)
    parser.add_argument('-n', '--neighbours', help='Neighbours',
                        default=DEFAULT_MIN_THRESHOLD, type=int)
    parser.add_argument('-b', '--bomb', help='Bomb the genome', action='store_true',
                        default=True)
    parser.add_argument('-m', '--mcmc', help='Number of MCMC iterations', type=int, default=DEFAULT_ITERATIONS_MCMC)
    parser.add_argument('-e', '--em', help='Number of EM iterations', type=int, default=DEFAULT_ITERATIONS_EM)
    parser.add_argument('-s', '--size', help='Pyramid size', type=int, default=7)
    parser.add_argument('-r', '--rippe', help='Use a Rippe polymer model', default=True, action='store_true')
    parser.add_argument('-S', '--simulation', help='Run simulation', default=True, action='store_true')
    parser.add_argument('-B', '--burn-in', help='Burn-in cycles', default=DEFAULT_BURN_IN_CYCLES, type=int)

    args = parser.parse_args()
    name = args.project
    n_cycles = args.cycles
    level = args.level
    n_neighbours = args.neighbours
    bomb = args.bomb
    n_iterations_em = args.em
    n_iterations_mcmc = args.mcmc
    size_pyramid = args.size
    use_rippe = args.rippe
    is_simu = args.is_simu
    burn_in = args.burn_in
    # name = 'human_chr4_chr14'
    # name = 'human_chr19_to_22'
    # name = 'human_chr7_chr22'
    # name = 's1_de_novo'
    # name = 's1_de_novo_0_1'

    # name = 'mosquito'
    # name = 'amoeba'1
    # name = 'S1'
    scrambled = False

    #use_rippe = False
    gl_size_im = 1000

    #sample_param = False
    p2 = window(name,
                level,
                n_iterations_em,
                n_iterations_mcmc,
                is_simu,
                scrambled,
                perform_em,
                use_rippe,
                gl_size_im,
                sample_param,
                size_pyramid)
    sampler = p2.simulation.sampler

    p2.full_em(n_cycles, n_cycles, bomb, burn_in)
    # sampler.step_sampler(50)
    # sampler.gpu_vect_frags.copy_from_gpu()
    # max_id = sampler.gpu_vect_frags.id_c.max()
    # frag_a = 0
    # frag_b = 1
    # id_c_a = sampler.gpu_vect_frags.id_c[frag_a]
    # id_c_b = sampler.gpu_vect_frags.id_c[frag_b]
    # print "id_frag a =", frag_a, "id contig = ", id_c_a
    # print "id_frag b =", frag_b, "id contig = ", id_c_b
    # ####################################################################################################################
    # sampler.perform_mutations(frag_a, frag_b, max_id, 1 == 0)
    # ####################################################################################################################
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

