#######################################
# Zane Rossi
# University of Chicago, Chong Lab
# June, 2018
# Using python 3.6.5, 2.7.13
# Error Correction Program Synthesis
#######################################

import tensorflow as tf
import numpy as np
import math
import os

from tensorflow.python.framework import function
from tensorflow.python.framework import ops

# importing loss functions and helper functions
###
###

class TensorflowState:

	def __init__(self, sys_para):
		self.sys_para = sys_para

	def build_graph(self):

		graph = tf.Graph()

		# process to initiate all elements of graph
		###
		###

		return graph