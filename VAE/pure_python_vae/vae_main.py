import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims
from utilities import *
from operators import *

# defining a class
# Q: why is this referred to as the latent attention, or the latent space
# see papers referenced 
class LatentAttention():

    def __init__(self):
        # set of small number recognition images
        # one-hot representation is a way of making sparse labels into
        # dense representations
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        # fetches the number of examples from the MNIST file
        # train data is well known tensorflow object
        self.n_samples = self.mnist.train.num_examples

        # specify net dimensions
        self.n_hidden = 500
        self.n_z = 20
        self.batchsize = 100

        # placeholder variable for constructing graph, 784 is MNIST-specific parameter
        self.images = tf.placeholder(tf.float32, [None, 784])

        # reshape the images (look up syntax for these arguments)
        image_matrix = tf.reshape(self.images,[-1, 28, 28, 1])

        # calls interior function
        z_mean, z_stddev = self.recognition(image_matrix)

        # generate normal noise for the guessed image?
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        # generate images out of the auto-encoder
        self.generated_images = self.generation(guessed_z)
        # restore the original array dimension (flattening them again)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])

        # use something like the self-entropy here to compute loss in the abstract
        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)

        # this is a different sort of loss, having to do with the sum of squares of various quantities (see post)
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)

        # then we average the two of these (as a technique)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)

        # this is the training call which will create the back-propogation graph
        # still don't know much yet what the adam optimizer does
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images):
        # setting the scope of variables to within this method? 
        with tf.variable_scope("recognition"):
            # commands for convolutional networks and reshaping
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
            # command for the same idea, save even larger of an image
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32
            # then we flatten the whole thing again
            h2_flat = tf.reshape(h2,[self.batchsize, 7*7*32])

            # constructing fully connected layers
            w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 7*7*32, self.n_z, "w_stddev")

        return w_mean, w_stddev

    # decoder
    def generation(self, z):
        # basically we do everything from the encoder method in reverse
        # having encoded everything into the z variable
        with tf.variable_scope("generation"):
            z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
            z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 7, 7, 32]))
            h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 14, 14, 16], "g_h1"))
            h2 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

        return h2

    # training
    def train(self):
        visualization = self.mnist.train.next_batch(self.batchsize)[0]
        reshaped_vis = visualization.reshape(self.batchsize,28,28)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # hardcoded epoch managing
            for epoch in range(10):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch = self.mnist.train.next_batch(self.batchsize)[0]
                    _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                    # naive way to print results as we go along
                    if idx % (self.n_samples - 3) == 0:
                        print "epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss))
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize,28,28)
                        ims("results/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))


model = LatentAttention()
model.train()