#######################################
# Zane Rossi
# University of Chicago, Chong Lab
# June, 2018
# Using python 3.6.5, 2.7.13
# Error Correction Program Synthesis
#######################################

import tensorflow as tf
import numpy as np
import os
import time

from scipy.optimize import minimize

# import analysis helper file for data visualization
###
###

class RunSession:

	def __init__(self, tfs,graph, sys_para, use_gpu = True):

		# within scope of a session call adam_optimizer
		# which then perfoms the true minimize calls for
		# the gradient created in the tensorflow state object
		###
		###


	# currently we will only use the adam_optimizer from tensorflow
	def adam_optimizer(self):

		# exponentially tempering learning rate, calling run with
		# periodic updates to relevant objects in the graph

		return 0

	# the rest of this is helper methods, data saving
