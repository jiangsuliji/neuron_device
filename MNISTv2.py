'''
Save and Restore a model using TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
credit to: Aymeric Damien
Author: Ji Li
'''

# This is for MNIST training but saves the results into model file

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.03
batch_size = 100
model_path = "./MNIST/nn/NN"
file_ending = ".ckpt"
epoch_num = 50 
clip_var = 2.0

# Network Parameters
n_hidden_1 = 300 # 1st layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def clip_grad(x, clip_value, name=None):
    """"
    scales backpropagated gradient so that
    its L2 norm is no more than `clip_value`
    """
    with tf.name_scope(name, "ClipGrad", [x]) as name:
        return py_func(lambda x : x,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=lambda op, g : tf.clip_by_norm(g, clip_value))[0]

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.matmul(x, clip_grad(weights['h1'], clip_var))
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, clip_grad(weights['out'], clip_var))

    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights,biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=pred)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver(max_to_keep=100)

# Running first session
print("Starting 1st session...")
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    accuracies = []
    for epoch in range(epoch_num):
        save_path = model_path + str(epoch+1) + file_ending
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        #if epoch % display_step == 0:
        #    print("Epoch:", '%04d' % (epoch+1), "cost=", \
        #        "{:.9f}".format(avg_cost))
         
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
        accuracies.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))



        # Save model weights to disk
        save_path = saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    print("First Optimization Finished!")
    print("Final accuracies:", accuracies)


