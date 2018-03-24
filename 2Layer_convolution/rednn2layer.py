'''
Save and Restore a model using TensorFlow.
Author:y Ji Li
This file is for ORL dataset, 2 layer DNN implementation training and save model 
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf

# Parameters
#learning_rate = 0.3 
#batch_size = 30
model_path = "./2nn/NN" # 2 layer 
file_ending = ".ckpt"
epoch_num = 15 
num_of_samples = 1

# Network Parameters
n_hidden_1 = 100 # 1st layer number of features
n_input = 112*92 # MNIST data input (img shape: 28*28)
n_classes = 20 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#load dataset
testX = np.load('testX.npy')
testY= np.load('testY.npy')


#images = tf.constant(train_data, dtype=tf.float32) # X is a np.array
#labels = tf.constant(train_labels, dtype=tf.int32)   # y is a np.array

#def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
#
#    # Need to generate a unique name to avoid duplicates:
#    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
#
#    tf.RegisterGradient(rnd_name)(grad)
#    g = tf.get_default_graph()
#    with g.gradient_override_map({"PyFunc": rnd_name}):
#        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
#
#def clip_grad(x, clip_value, name=None):
#    """"
#    scales backpropagated gradient so that
#    its L2 norm is no more than `clip_value`
#    """
#    with tf.name_scope(name, "ClipGrad", [x]) as name:
#        return py_func(lambda x : x,
#                        [x],
#                        [tf.float32],
#                        name=name,
#                        grad=lambda op, g : tf.clip_by_norm(g, clip_value))[0]
#

# normal sample with mean and var
def new_weight(mean, var):
    return var * np.random.randn()+mean

# parse input distribution data
# return distribution, max, min
distribution = []
distribution_path = "../distribution.txt"
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

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.matmul(x, weights['h1'])#+biases['b1']
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out'])#+biases['out']
    #out_layer = tf.layers.dropout(inputs=out_layer,rate=0.5,training=True)
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
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=pred)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# 'Saver' op to save and restore all the variables
saver = tf.train.Saver(max_to_keep=100)




# Running first session
print("Starting 2nd session...")
with tf.Session() as sess:

    #print("------Testing inputs ------------")
    #print(testX, testY)

    print("------staring training eval etc--")

    # Run the initializer
    # sess.run(init)
    tf.initialize_all_variables().run() 

    real_accu = []
    ideal_accu = []

    for epoch in range(epoch_num):
        save_path = model_path + str(epoch+1) + file_ending
        # Restore model weights from previously saved model
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

        # Ideal accuracy test
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Ideal Accuracy:", accuracy.eval(
            {x: testX, y: testY}))
        ideal_accu.append(accuracy.eval({x: testX, y: testY}))
        
        # update weight values
        newh1 = weights['h1'].eval()
        newout = weights['out'].eval()
        saveh1 = np.copy(newh1) # deep copy
        saveout = np.copy(newout)
        #print(newout)
        
        print("weight layer 1 max-min:",newh1.max(),newh1.min())
        print("weight layer 2 max-min:",newout.max(),newout.min())

        sample_accu = []
        
        # loop for # of normal distribution samples
        for sample in range(num_of_samples):
            print("ite:",sample)
            for i in range(0,n_input):
                for j in range(0,n_hidden_1):
                    tmp = saveh1[i][j] 
                    tmp = check_distribution(tmp, 5., -5.,  distribution, distribution_max, distribution_min, distribution_stage)
                    newh1[i][j] = tmp
            for i in range(0,n_hidden_1):
                for j in range(0,n_classes):
                    tmp = saveout[i][j] 
                    tmp = check_distribution(tmp, 5., -5.,  distribution, distribution_max, distribution_min, distribution_stage)
                    newout[i][j] = tmp
            #print("after-----",newout)
            weights['h1'].assign(newh1).eval()
            weights['out'].assign(newout).eval()
            # Real accuracy test
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            sample_accu.append(accuracy.eval({x: testX, y: testY}))
        print("Real sample_accu: ",sample_accu)
        real_accu.append(sum(sample_accu)/(len(sample_accu)))




    print("Final ideal accuracies:", ideal_accu)
    print("Final real accuracies:", real_accu)
