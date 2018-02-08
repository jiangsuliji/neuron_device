# Rebuild the UCI network with low accuracy
from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.1
batch_size = 100
display_step = 1
model_path = "./MNIST/nn/NN"
file_ending = ".ckpt"
epoch_num = 40 
n_bits = 6 # Device resolution--device state--mapping -1.0~1.0 into these states
f_n_bits = 2.0/(2**n_bits)

# Network Parameters
n_hidden_1 = 300 # 1st layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# round operation to convert floating point number to fixed point number
def fround(fnum):
    return round((fnum+1.0)/f_n_bits)*f_n_bits - 1.0

# for testing fround purpose
#for i in np.linspace(-1, 1, num=73):
    #print(fround(i))

# Create model
def multilayer_perceptron(x, weights):
    # Hidden layer with RELU activation
    layer_1 = tf.matmul(x, weights['h1'])
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out'])
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()


# 'Saver' op to save and restore all the variables
saver = tf.train.Saver(max_to_keep=100)

# Running a new session
print("Starting 2nd session...")
with tf.Session() as sess:
    # Initialize variables
    # sess.run(init)
    tf.initialize_all_variables().run() 
    
    real_accu = []
    ideal_accu = []
    for epoch in range(epoch_num):
        save_path = model_path + str(epoch+1) + file_ending

        # Restore model weights from previously saved model
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
    

        #print("valofw:", weights['h1'].eval())
        #nw = {
        #    'h1': weights['h1'].eval(),
        #    'out': weights['out'].eval()}

        #print(nw)
        #nw['h1'] = np.zeros((784,300))
        #weights['h1'].assign(nw['h1']).eval()
        #print("afterchange:",weights['h1'].eval())

        #print(sess.run(weights))
        #print(weights['h1'])
        #print(weights['h1'].get_shape(),weights['out'].get_shape())
        
        # Ideal accuracy test
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval(
            {x: mnist.test.images, y: mnist.test.labels}))
        ideal_accu.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
        # update weight values
        newh1 = weights['h1'].eval()
        newout = weights['out'].eval()
        #print(newh1)
        for i in range(0,n_input):
            for j in range(0,n_hidden_1):
                newh1[i][j] = fround(newh1[i][j])
        #print(newh1)
        for i in range(0,n_hidden_1):
            for j in range(0,n_classes):
                newout[i][j] = fround(newout[i][j])
        
        weights['h1'].assign(newh1).eval()
        weights['out'].assign(newout).eval()


        # Real accuracy test
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval(
            {x: mnist.test.images, y: mnist.test.labels}))
        real_accu.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        



    print("Final ideal accuracies:", ideal_accu)
    print("Final real accuracies:", real_accu)
