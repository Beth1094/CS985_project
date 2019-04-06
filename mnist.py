import os
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
from helpers import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
num_classes = 10
lambdaReg=0.


# fully connected one layer
def one_hidden_layers(X, n_hidden1= 300, n_outputs=num_classes, activation_func=tf.nn.sigmoid, lambdaReg=lambdaReg):
    print("Network with one hidden layer")
    with tf.name_scope("dnn"):
        with tf.variable_scope("layer1"):
            hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=activation_func, lambdaReg=lambdaReg)
            # hidden1 = tf.nn.dropout(hidden1, dropout)
        with tf.variable_scope("layer2"):
            logits = neuron_layer(hidden1, n_outputs, name="outputs", lambdaReg=lambdaReg)
    return logits

#fully connected two layer
def two_hidden_layers(X, n_hidden1 = 300, n_hidden2 = 100, n_outputs = num_classes, activation_func=tf.nn.sigmoid, lambdaReg=lambdaReg):
    print("Network with two hidden layers")
    with tf.name_scope("dnn"):
        with tf.variable_scope("layer1"):
            hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=activation_func, lambdaReg=lambdaReg)
        with tf.variable_scope("layer2"):
            hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=activation_func, lambdaReg=lambdaReg)
            #hidden2 = tf.nn.dropout(hidden2, dropout)
        with tf.variable_scope("layer3"):
            logits = neuron_layer(hidden2, n_outputs, name="outputs", lambdaReg=lambdaReg)
    return logits

#fully connected three layer
def three_hidden_layers(X, n_hidden1 = 300, n_hidden2 = 100, n_hidden3 = 50, n_outputs = num_classes, activation_func=tf.nn.sigmoid, lambdaReg=lambdaReg):
    print("Network with three hidden layers")
    with tf.name_scope("dnn"):
        with tf.variable_scope("layer1"):
            hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=activation_func, lambdaReg=lambdaReg)
        with tf.variable_scope("layer2"):
            hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=activation_func, lambdaReg=lambdaReg)
            #hidden2 = tf.nn.dropout(hidden2, dropout) #dropout layer
        with tf.variable_scope("layer3"):
            hidden3 = neuron_layer(hidden2, n_hidden3, name="hidden3", activation=activation_func, lambdaReg=lambdaReg)
        with tf.variable_scope("layer4"):
            logits = neuron_layer(hidden3, n_outputs, name="outputs", lambdaReg=lambdaReg)
    return logits


#fully connected network architecture training
def mlp_network(layers, learning_rate, epochs, batches, activation_func):
    n_inputs = 28*28  #image size is 28 by 28 pixels
    learning_rate = learning_rate
    n_epochs = epochs #number of iterations of learning
    batch_size = batches

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    #keep_prob = tf.placeholder(tf.float32) # dropout placeholder(keep probability)


    if layers ==1:
        with tf.variable_scope("one_layer") as scope:
            logits = one_hidden_layers(X, activation_func=activation_func, lambdaReg=lambdaReg)
            scope.reuse_variables()
    elif layers ==2:
        with tf.variable_scope("two_layer") as scope:
            logits = two_hidden_layers(X, activation_func=activation_func, lambdaReg=lambdaReg)
            scope.reuse_variables()
    elif layers ==3:
        with tf.variable_scope("two_layer") as scope:
            logits = three_hidden_layers(X, activation_func=activation_func, lambdaReg=lambdaReg)
            scope.reuse_variables()

    with tf.name_scope("loss"):
        reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss") + reg_losses/784


    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y ,1) #tests whether targets are in top k predictions
        accuracy = tf.reduce_mean( tf.cast( correct, tf.float32) )

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs): #epoch: num of times run whole dataset through network
            for iteration in range(mnist.train.num_examples//batch_size): #run each batch through network iteration times
                X_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
            acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
            acc_val = accuracy.eval(feed_dict={X: mnist.validation.images, y:mnist.validation.labels})

            print(epoch, "Train Accuracy: {:3f}  Validation Accuracy: {:3f}".format(acc_train, acc_val), end="\r")

        save_path = saver.save(sess,"tmp/my_model_final.ckpt")
        print("")
        print("Optimization Finished!")

        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y:mnist.test.labels})
        print("Test Accuracy: {:3f}".format(acc_test))


#----------------------------------------------------------------------------------------------------------------

# Covolutional layer with pooling x2 and 1 fc layer before output
def conv_net(x, dropout, activation, lambdaReg):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # 2 Convolution Layer & max pooling downsampling
    conv1 = conv2d(x, 1, 32, [5,5], 1, [2,2], name = "conv_layer1")

    conv2 = conv2d(conv1, 32, 64, [5,5], 1, [2,2], name = "conv_layer2")

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, tf.Variable(tf.random_normal([7*7*64, 1000])).get_shape().as_list()[0]])
    with tf.variable_scope("layer1"):
        fc1 = neuron_layer(fc1, 1000, name = "fully_connected1", activation=activation, lambdaReg=lambdaReg)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout) #dropout layer

    with tf.variable_scope("layer2"):
        # Output, class prediction
        out = neuron_layer(fc1, num_classes, name = "output_layer", activation=None, lambdaReg=lambdaReg)

    return out


#dropout: probability to keep units
#train CNN with dropout
def conv_network(learning_rate, epochs, batches, activation_func, dropout):
    display_step = 10
    n_inputs = 28*28  #image size is 28 by 28 pixels
    learning_rate = learning_rate
    n_epochs = epochs #number of iterations of learning
    batch_size = batches
    lambdaReg = 0

    # tf Graph input
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    Y = tf.placeholder(tf.int64, shape=(None), name="y")
    keep_prob = tf.placeholder(tf.float32) # dropout placeholder(keep probability)


    # Construct model
    logits = conv_net(X, keep_prob, activation = activation_func, lambdaReg=lambdaReg)
    #prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)


    # Evaluate model
    correct_pred = tf.nn.in_top_k(logits, Y ,1)# tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        init.run()

        for epoch in range(n_epochs):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and dropout at training time
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            if epoch % display_step == 0 or epoch == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0})

                acc_val = sess.run(accuracy, feed_dict={X: mnist.validation.images, Y: mnist.validation.labels, keep_prob: 1.0})


                print(epoch, "Train Accuracy: {:3f}  Validation Accuracy: {:3f}".format(acc, acc_val), end="\r")

        save_path = saver.save(sess, "tmp/my_model_final.ckpt")
        print("")
        print("Optimization Finished!")

        # Calculate accuracy for 256 MNIST test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: mnist.test.images[0:256],
                                      Y: mnist.test.labels[0:256],
                                      keep_prob: 1.0}))


#-----------------------------------------------------------------------------------------
def network_one(learning_rate, epochs, batches):

    print("CNN with two convolutional layers, two pooling layers and one FC layer")
    print("Combination One with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    conv_network(learning_rate, epochs, batches, activation_func=tf.nn.relu, dropout=0.5)


def network_two(learning_rate, epochs, batches):

    print("Sigmoid Network with Three Hidden Layers")
    print("Combination Two with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    mlp_network(3, learning_rate, epochs, batches, activation_func=tf.nn.sigmoid)



def main(combination, learning_rate, epochs, batches, seed):

    # Set Seed
    print("Seed: {}".format(seed))

    if int(combination)==1:
        network_one(learning_rate, epochs, batches)
    if int(combination)==2:
        network_two(learning_rate, epochs, batches)

    print("Done!")

def check_param_is_numeric(param, value):

    try:
        value = float(value)
    except:
        print("{} must be numeric".format(param))
        quit(1)
    return value


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("iterations", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")

    args = arg_parser.parse_args()

    combination = check_param_is_numeric("combination", args.combination)
    learning_rate = check_param_is_numeric("learning_rate", args.learning_rate)
    epochs = check_param_is_numeric("epochs", args.iterations)
    batches = check_param_is_numeric("batches", args.batches)
    seed = check_param_is_numeric("seed", args.seed)

    main(combination, learning_rate, int(epochs), int(batches), int(seed))