import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def show_images(X,y):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i], cmap=plt.cm.binary)
        plt.xlabel(y[i])
    plt.show()



def neuron_layer(X, n_neurons, name, lambdaReg, activation=None): #initialises neurons in layer and performs activation function on input
    #X: inputs, n_neurons: outputs
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2.0 / np.sqrt(n_inputs + n_neurons)
        W = tf.get_variable("weights", shape=(n_inputs, n_neurons),
                            initializer=tf.truncated_normal_initializer(stddev=stddev),regularizer=tf.contrib.layers.l2_regularizer(lambdaReg))
        b = tf.Variable(tf.zeros([n_neurons]),name="bias")
        Z = tf.matmul(X,W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z



def heavy_side(z, name=None):
    return tf.nn.relu(tf.math.sign(z), name=name)

def leaky_relu(z, name=None):
    return tf.maximum(0.2*z,z, name=name)

#----------------------------------------------------------------------------------------------------------------------

def conv2d(input_data, num_input_channels, num_filters, filter_shape, stride, pool_shape, name):
    #filter shape = LRF = [5,5]
    #pool_shape = [2,2]
    #num_filters = num of channels (no of outputs)
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    n_inputs = int(input_data.get_shape()[1])
    stddev = 2.0 / np.sqrt(n_inputs + num_filters)

    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=stddev), #stddev = 0.03
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [stride, stride, 1, 1], padding='SAME')

    # add the bias
    out_layer = tf.nn.bias_add(out_layer, bias)

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    return out_layer
