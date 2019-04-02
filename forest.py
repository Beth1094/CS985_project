from sklearn.datasets import fetch_covtype
from sklearn.utils import check_array
import numpy as np
import tensorflow as tf
from helpers import heavy_side, leaky_relu
import os
from datetime import datetime
import argparse

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)



#### download data####

print("Loading dataset...")
data = fetch_covtype(download_if_missing=True, shuffle=True,
                         random_state=13)
X = check_array(data['data'], dtype=np.float32, order='C')
y = (data['target'] != 1).astype(np.int)

# Create train-test split
print("Creating train, valide, test split...")
n_train = 522911
X_train = X[:n_train]
y_train = y[:n_train]
X_test = X[n_train:]
y_test = y[n_train:]
X_valid, X_train = X_train[:52000], X_train[52000:]
y_valid, y_train = y_train[:52000], y_train[52000:]

# Standardize first 10 features (the numerical ones)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
mean[10:] = 0.0
std[10:] = 1.0
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
X_valid = (X_valid - mean) / std

# print size of training, valid and test set
print(len(X_train), len(X_valid), len(X_test))



m, n = X_train.shape

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = int(np.ceil(m / batch_size))
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch



#### two hidden layers function #####

def two_hlayers(learning_rate, batch_size, activation_fnc, n_hidden1, n_hidden2, n_epochs):
    n_inputs = 54  # no. of variable
    n_outputs = 7  # no. of class

    logdir = log_dir("forestbook12_dnn")
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    reset_graph()
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=activation_fnc, kernel_initializer=he_init)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=activation_fnc,
                                  kernel_initializer=he_init)
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs", kernel_initializer=he_init)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
        loss_summary = tf.summary.scalar('log_loss', loss)

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    final_model_path = "./my_model_final.ckpt"

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            if epoch % 10 == 0:
                print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

        save_path = saver.save(sess, "./my_model_final.ckpt")

    with tf.Session() as sess:
        saver.restore(sess, final_model_path)
        accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("Test accuracy", accuracy_val)


#### three hidden layers function ####

def three_hlayers(learning_rate, batch_size, activation_fnc, n_hidden1, n_hidden2, n_hidden3, n_epochs):
    n_inputs = 54  # no. of variable
    n_outputs = 7  # no. of class

    logdir = log_dir("forestbook12_dnn")
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    reset_graph()
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    with tf.name_scope("dnn"):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=activation_fnc, kernel_initializer=he_init)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=activation_fnc,
                                  kernel_initializer=he_init)
        hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=activation_fnc,
                                  kernel_initializer=he_init)
        logits = tf.layers.dense(hidden3, n_outputs, name="outputs", kernel_initializer=he_init)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
        loss_summary = tf.summary.scalar('log_loss', loss)

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    final_model_path = "./my_model_final.ckpt"

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            if epoch % 10 == 0:
                print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

        save_path = saver.save(sess, "./my_model_final.ckpt")

    with tf.Session() as sess:
        saver.restore(sess, final_model_path)
        accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("Test accuracy", accuracy_val)



def network_one(learning_rate, epochs, batches):

    print("Network with Two Hidden Layers")
    print("Combination Three with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    two_hlayers(learning_rate, batch_size = batches, activation_fnc = tf.nn.sigmoid, n_hidden1 =50, n_hidden2 =50, n_epochs = epochs)


def network_two(learning_rate, epochs, batches):
    print("Sigmoid Network with Three Hidden Layer")
    print("Combination Four with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    three_hlayers(learning_rate, batch_size = batches, activation_fnc = tf.nn.sigmoid, n_hidden1=50, n_hidden2 =30, n_hidden3 =20, n_epochs = epochs)



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