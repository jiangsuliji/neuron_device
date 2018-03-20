# Rebuild the UCI network with low accuracy
from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import numpy as np

# Training Parameters
#learning_rate = 0.1
batch_size = 100
display_step = 1
model_path = "./MNIST/nn/NN"
file_ending = ".ckpt"
epoch_num = 50 

# Network Parameters
n_hidden_1 = 300 # 1st layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# Device Parameters
#n_bits = 6 # Device resolution--device state--mapping -1.0~1.0 into these states
#f_n_bits = 2.0/(2**n_bits)

# Gaussian distribution
mu = 0 # Mean 
sigma = 0.0 # Standard deviation
num_of_samples = 10
gaussian_matrix = {
        'h1': np.random.normal(mu,sigma,epoch_num*n_input*n_hidden_1).\
                reshape(epoch_num,n_input,n_hidden_1),
        'out': np.random.normal(mu,sigma,epoch_num*n_hidden_1*n_classes).\
                reshape(epoch_num,n_hidden_1,n_classes)
        } 
        # Stores epoch *[dimension,dimension] for each weight matrix

# normal sample with mean and var
def new_weight(mean, var):
    return var * np.random.randn()+mean

# parse input distribution data
# return distribution, max, min
distribution = []
distribution_path = "./distribution.txt"
distribution_max, distribution_min = None, None
distribution_stage = None
def read_distribution(path):
    rtn = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines[1:]:
            tmp = line.split()
            rtn.append([float(tmp[1]), float(tmp[2])])
    return rtn, rtn[-1][0], rtn[0][0]

distribution, distribution_max, distribution_min = read_distribution(distribution_path)

distribution_stage = len(distribution)
# find distribution parameter 
# conductance -1 to 1 value,distribution array, max, min of conductance, total stages
# return mean, stdv
def check_distribution(c, chi, clo, d, hi, lo, stage):
    idx = int((c-clo)*stage/(chi-clo))
    if idx >= stage-1: 
        m, var = d[-1][0], d[-1][1]
    else:
        m, var = (d[idx][0]+d[idx+1][0])/2.0, (d[idx][1]+d[idx+1][1])/2.0
    m = new_weight(m, var)
    return (m-lo)/(hi-lo)*(chi-clo) + clo

#u, var = check_distribution(-0.1, 4.5, -4.5,  distribution, distribution_max, distribution_min, distribution_stage)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# round operation to convert floating point number to fixed point number
#def fround(fnum):
#    return round((fnum+1.0)/f_n_bits)*f_n_bits - 1.0

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
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

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
        print("Ideal Accuracy:", accuracy.eval(
            {x: mnist.test.images, y: mnist.test.labels}))
        ideal_accu.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
        # update weight values
        newh1 = weights['h1'].eval()
        newout = weights['out'].eval()
        saveh1 = np.copy(newh1) # deep copy
        saveout = np.copy(newout)
        #newh1 += gaussian_matrix['h1'][epoch]
        #newout += gaussian_matrix['out'][epoch]
        #print(newh1)
        print("weight layer 1 max-min:",newh1.max(),newh1.min())
        print("weight layer 2 max-min:",newout.max(),newout.min())
        
        sample_accu = []

        # loop for # of normal distribution samples
        for sample in range(num_of_samples):
            for i in range(0,n_input):
                for j in range(0,n_hidden_1):
                    tmp = saveh1[i][j] 
                    tmp = check_distribution(tmp, 4.8, -4.8,  distribution, distribution_max, distribution_min, distribution_stage)
                    newh1[i][j] = tmp
                    #newh1[i][j] = fround(tmp)
            #print("after-----",newh1)
            for i in range(0,n_hidden_1):
                for j in range(0,n_classes):
                    tmp = saveout[i][j] 
                    tmp = check_distribution(tmp, 4.8, -4.8,  distribution, distribution_max, distribution_min, distribution_stage)
                    newout[i][j] = tmp
                    #newout[i][j] = fround(tmp)
            weights['h1'].assign(newh1).eval()


            # Real accuracy test
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            #print("Accuracy:", accuracy.eval(
                #{x: mnist.test.images, y: mnist.test.labels}))
            sample_accu.append(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print("Real sample_accu: ",sample_accu)
        real_accu.append(sum(sample_accu)/(len(sample_accu)))
        



    print("Final ideal accuracies:", ideal_accu)
    print("Final real accuracies:", real_accu)
