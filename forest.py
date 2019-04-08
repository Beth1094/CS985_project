from __future__ import division, print_function, unicode_literals
from sklearn.datasets import fetch_covtype
from sklearn.utils import check_array
import numpy as np
import tensorflow as tf
from helpers2 import heavy_side, leaky_relu, neuron_layer, reset_graph, log_dir, shuffle_batch
import os
import argparse

############## download dataset and preprocessing ##################

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



############################ two hidden layers function #################################

def two_hlayers(learning_rate, batch_size, activation_fnc, n_hidden1, n_hidden2, n_epochs, trainoptimizer):
    n_inputs = 54  # no. of variable
    n_outputs = 7  # no. of class
    dropout_rate = 0.5
    lambdaReg = 0.0

    logdir = log_dir("forestbook_dnn")
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    reset_graph()
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    training = tf.placeholder_with_default(False, shape=(), name='training')
    X_drop = tf.layers.dropout(X, dropout_rate, training=training)

    with tf.name_scope("dnn"):
        with tf.variable_scope("layer1"):
            hidden1 = neuron_layer(X, n_hidden1, "hidden1", lambdaReg, activation=activation_fnc)
            hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
        with tf.variable_scope("layer2"):
            hidden2 = neuron_layer(hidden1_drop, n_hidden2, "hidden2", lambdaReg, activation=activation_fnc)
            hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
        with tf.variable_scope("layer3"):
            logits = neuron_layer(hidden2_drop, n_outputs, "outputs", lambdaReg)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
        loss_summary = tf.summary.scalar('log_loss', loss)

    with tf.name_scope("train"):
        optimizer = trainoptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    checkpoint_path = "/tmp/my_deep_forest_model.ckpt"
    checkpoint_epoch_path = checkpoint_path + ".epoch"
    final_model_path = 'tmp/forest-{comb}-{LR}-{epoch}-{batch}.ckpt'.format(comb="3", LR=str(learning_rate),
                                                                       epoch=str(n_epochs), batch=str(batch_size))

    best_loss = np.infty
    epochs_without_progress = 0
    max_epochs_without_progress = 50

    with tf.Session() as sess:
        if os.path.isfile(checkpoint_epoch_path):
            # if the checkpoint file exists, restore the model and load the epoch number
            with open(checkpoint_epoch_path, "rb") as f:
                start_epoch = int(f.read())
            print("Training was interrupted. Continuing at epoch", start_epoch)
            saver.restore(sess, checkpoint_path)
        else:
            start_epoch = 0
            sess.run(init)

        for epoch in range(start_epoch, n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run(
                [accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_valid, y: y_valid})
            file_writer.add_summary(accuracy_summary_str, epoch)
            file_writer.add_summary(loss_summary_str, epoch)
            if epoch % 10 == 0:
                print("Epoch:", epoch,
                      "\tBatch accuracy: {:.3f}%".format(accuracy_batch * 100),
                      "\tLoss: {:.5f}".format(loss_val),
                      "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
                      "\tLoss: {:.5f}".format(loss_val))
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))
                if loss_val < best_loss:
                    saver.save(sess, final_model_path)
                    best_loss = loss_val

                else:
                    epochs_without_progress += 5
                    if epochs_without_progress > max_epochs_without_progress:
                        print("Early stopping")
                        break

    os.remove(checkpoint_epoch_path)

    with tf.Session() as sess:
        saver.restore(sess, final_model_path)
        accuracy_test = accuracy.eval(feed_dict={X: X_test, y: y_test})

    print("\tTest accuracy: {:.3f}%".format(accuracy_test * 100))


############################ three hidden layers function ##############################

def three_hlayers(learning_rate, batch_size, activation_fnc, n_hidden1, n_hidden2, n_hidden3, n_epochs, trainoptimizer):
    n_inputs = 54  # no. of variable
    n_outputs = 7  # no. of class
    dropout_rate = 0.5
    lambdaReg = 0.0

    logdir = log_dir("forestbook_dnn")
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    reset_graph()
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    training = tf.placeholder_with_default(False, shape=(), name='training')
    X_drop = tf.layers.dropout(X, dropout_rate, training=training)

    with tf.name_scope("dnn"):
        with tf.variable_scope("layer1"):
            hidden1 = neuron_layer(X, n_hidden1, "hidden1", lambdaReg, activation=activation_fnc)
            hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
        with tf.variable_scope("layer2"):
            hidden2 = neuron_layer(hidden1_drop, n_hidden2, "hidden2", lambdaReg, activation=activation_fnc)
            hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
        with tf.variable_scope("layer3"):
            hidden3 = neuron_layer(hidden2_drop, n_hidden3, "hidden3", lambdaReg, activation=activation_fnc)
            hidden3_drop = tf.layers.dropout(hidden3, dropout_rate, training=training)
        with tf.variable_scope("layer4"):
            logits = neuron_layer(hidden3_drop, n_outputs, "outputs", lambdaReg)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
        loss_summary = tf.summary.scalar('log_loss', loss)

    with tf.name_scope("train"):
        optimizer = trainoptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    checkpoint_path = "/tmp/my_deep_forest_model.ckpt"
    checkpoint_epoch_path = checkpoint_path + ".epoch"
    final_model_path = 'tmp/forest-{comb}-{LR}-{epoch}-{batch}.ckpt'.format(comb="2", LR=str(learning_rate),
                                                                       epoch=str(n_epochs), batch=str(batch_size))

    best_loss = np.infty
    epochs_without_progress = 0
    max_epochs_without_progress = 50

    with tf.Session() as sess:
        if os.path.isfile(checkpoint_epoch_path):
            # if the checkpoint file exists, restore the model and load the epoch number
            with open(checkpoint_epoch_path, "rb") as f:
                start_epoch = int(f.read())
            print("Training was interrupted. Continuing at epoch", start_epoch)
            saver.restore(sess, checkpoint_path)
        else:
            start_epoch = 0
            sess.run(init)

        for epoch in range(start_epoch, n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run(
                [accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_valid, y: y_valid})
            file_writer.add_summary(accuracy_summary_str, epoch)
            file_writer.add_summary(loss_summary_str, epoch)
            if epoch % 10 == 0:
                print("Epoch:", epoch,
                      "\tBatch accuracy: {:.3f}%".format(accuracy_batch * 100),
                      "\tLoss: {:.5f}".format(loss_val),
                      "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
                      "\tLoss: {:.5f}".format(loss_val))
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))
                if loss_val < best_loss:
                    saver.save(sess, final_model_path)
                    best_loss = loss_val

                else:
                    epochs_without_progress += 5
                    if epochs_without_progress > max_epochs_without_progress:
                        print("Early stopping")
                        break

    os.remove(checkpoint_epoch_path)

    with tf.Session() as sess:
        saver.restore(sess, final_model_path)
        accuracy_test = accuracy.eval(feed_dict={X: X_test, y: y_test})

    print("\tTest accuracy: {:.3f}%".format(accuracy_test * 100))


############################ four hidden layers function ##############################

def four_hlayers(learning_rate, batch_size, activation_fnc, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_epochs,
                 trainoptimizer):
    n_inputs = 54  # no. of variable
    n_outputs = 7  # no. of class
    dropout_rate = 0.5
    lambdaReg = 0.0

    logdir = log_dir("forestbook_dnn")
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    reset_graph()
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    training = tf.placeholder_with_default(False, shape=(), name='training')
    X_drop = tf.layers.dropout(X, dropout_rate, training=training)

    with tf.name_scope("dnn"):
        with tf.variable_scope("layer1"):
            hidden1 = neuron_layer(X, n_hidden1, "hidden1", lambdaReg, activation=activation_fnc)
            hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
        with tf.variable_scope("layer2"):
            hidden2 = neuron_layer(hidden1_drop, n_hidden2, "hidden2", lambdaReg, activation=activation_fnc)
            hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
        with tf.variable_scope("layer3"):
            hidden3 = neuron_layer(hidden2_drop, n_hidden3, "hidden3", lambdaReg, activation=activation_fnc)
            hidden3_drop = tf.layers.dropout(hidden3, dropout_rate, training=training)
        with tf.variable_scope("layer4"):
            hidden4 = neuron_layer(hidden3_drop, n_hidden4, "hidden4", lambdaReg, activation=activation_fnc)
            hidden4_drop = tf.layers.dropout(hidden4, dropout_rate, training=training)
        with tf.variable_scope("layer5"):
            logits = neuron_layer(hidden4_drop, n_outputs, "outputs", lambdaReg)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
        loss_summary = tf.summary.scalar('log_loss', loss)

    with tf.name_scope("train"):
        optimizer = trainoptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    checkpoint_path = "/tmp/my_deep_forest_model.ckpt"
    checkpoint_epoch_path = checkpoint_path + ".epoch"
    final_model_path = 'tmp/forest-{comb}-{LR}-{epoch}-{batch}.ckpt'.format(comb="1", LR=str(learning_rate),
                                                                       epoch=str(n_epochs), batch=str(batch_size))

    best_loss = np.infty
    epochs_without_progress = 0
    max_epochs_without_progress = 50

    with tf.Session() as sess:
        if os.path.isfile(checkpoint_epoch_path):
            # if the checkpoint file exists, restore the model and load the epoch number
            with open(checkpoint_epoch_path, "rb") as f:
                start_epoch = int(f.read())
            print("Training was interrupted. Continuing at epoch", start_epoch)
            saver.restore(sess, checkpoint_path)
        else:
            start_epoch = 0
            sess.run(init)

        for epoch in range(start_epoch, n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run(
                [accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_valid, y: y_valid})
            file_writer.add_summary(accuracy_summary_str, epoch)
            file_writer.add_summary(loss_summary_str, epoch)
            if epoch % 10 == 0:
                print("Epoch:", epoch,
                      "\tBatch accuracy: {:.3f}%".format(accuracy_batch * 100),
                      "\tLoss: {:.5f}".format(loss_val),
                      "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
                      "\tLoss: {:.5f}".format(loss_val))
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))
                if loss_val < best_loss:
                    saver.save(sess, final_model_path)
                    best_loss = loss_val

                else:
                    epochs_without_progress += 5
                    if epochs_without_progress > max_epochs_without_progress:
                        print("Early stopping")
                        break

    os.remove(checkpoint_epoch_path)

    with tf.Session() as sess:
        saver.restore(sess, final_model_path)
        accuracy_test = accuracy.eval(feed_dict={X: X_test, y: y_test})

    print("\tTest accuracy: {:.3f}%".format(accuracy_test * 100))


###########################################################

def network_one(learning_rate, epochs, batches):

    print("ELU Network with Four Hidden Layer")
    print("Combination one with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    four_hlayers(learning_rate, batch_size = batches, activation_fnc = tf.nn.elu, n_hidden1 =50, n_hidden2 =40, n_hidden3 =30, n_hidden4=20, n_epochs =epochs,
                 trainoptimizer = tf.train.AdamOptimizer)


def network_two(learning_rate, epochs, batches):
    print("ReLU Network with Three Hidden Layer")
    print("Combination two with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    three_hlayers(learning_rate, batch_size = batches, activation_fnc = tf.nn.relu, n_hidden1 =50, n_hidden2 =50, n_hidden3 =50, n_epochs = epochs, trainoptimizer = tf.train.GradientDescentOptimizer)




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
