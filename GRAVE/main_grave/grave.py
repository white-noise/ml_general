#######################################
# Zane Rossi
# University of Chicago, Chong Lab
# June, 2018
# Using python 3.6.5, 2.7.13
# Error Correction Program Synthesis
#######################################

import tensorflow as tf
import numpy as np
import scipy.linalg as la
import random as rd
import time
import os

from IPython import display

def grave(H0, Hops, U, total_time, steps, U0 = None, maxA = None, use_gpu = True, show_plots = True, unitary_error = 1e-4, method = 'ADAM', no_scaling = False, freq_unit = 'GHz'):
    
    # start time
    grave_start_time = time.time()

    # time unit for plotting
    freq_time_unit_dict = {"GHz":"ns", "MHz":"us", "KHz":"ms", "Hz":"s"}
    time_unit = freq_time_unit_dict[freq_unit]

    # process for saving data is written here
    ###
    ###

    # process for convergence data is written here
    ###
    ###

    sys_para = SystemParameters()

    # specify gpu (recommended) or cpu session
    if use_gpu:
        dev = '/gpu:0'
    else:
        dev = '/cpu:0'

    with tf.device(dev):
        tfs = TensorflowState(sys_para)
        graph = tfs.build_graph()



    print "successful exit"
    return 0
