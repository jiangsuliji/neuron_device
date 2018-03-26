'''
Save and Restore a model using TensorFlow.
Author:y Ji Li
retore cnn for face regco
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf

# Parameters
model_path = "./nn2/NN"
file_ending = ".ckpt"
epoch_num = 20 

# Network Parameters
n_hidden_1 = 200 # 1st layer number of features
n_input = 10304#112*92 # MNIST data input (img shape: 28*28)
n_classes = 20 # MNIST total classes (0-9 digits)

# Convolutional Layer 1. -- mannually write 
#conv1_len = 5   # Convolution filters are 5 x 5 pixels.
#conv1_num = 16  # Convolution feature number

# tf Graph input
x = tf.placeholder("float", [None, n_input], name='X')
x_image = tf.reshape(x, [-1, 112, 92, 1])
y = tf.placeholder("float", [None, n_classes])

# Gaussian distribution
num_of_samples = 1

##load dataset
#testX = np.load('testX.npy')
#testY = np.load('testY.npy')
#trainX = np.load('trainX.npy')
#trainY = np.load('trainY.npy')

data = np.load('ORL_faces.npz') #assumption is that the data is unzipped
trainX = data['trainX']
testX = data['testX']
trainY= data['trainY']
testY= data['testY']

#Reshape train and test labels to one hot vector
trainY = np.eye(n_classes)[trainY]
testY  = np.eye(n_classes)[testY]

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


## reorganize rows
#train = np.concatenate((trainX, testX), axis = 0)
#test = np.concatenate((trainY, testY), axis = 0)
#idx = np.random.randint(train.shape[0],size=80)
#trainX = []
#trainY = []
#testX = []
#testY = []
#for i in range(0,400):
#    if i not in idx:
#        trainX.append(train[i])
#        trainY.append(test[i])
#    else:
#        testX.append(train[i])
#        testY.append(test[i])
#
#trainX = np.asarray(trainX)
#testX = np.asarray(testX)
#
## process label to one hot
#trainY_onehot = []
#for i in trainY:
#    trainY_onehot.append([0]*n_classes)
#    trainY_onehot[-1][int(i)] = 1
#trainY = np.asarray(trainY_onehot)
#
#testY_onehot = []
#for i in testY:
#    testY_onehot.append([0]*n_classes)
#    testY_onehot[-1][int(i)] = 1
#testY = np.asarray(testY_onehot)
#
## scaling
#trainX = (trainX-np.mean(trainX))/256
#testX = (testX-np.mean(testX))/256
#
## save test set for later evaluation
#np.save('testX',testX)
#np.save('testY',testY)
#
#print(":::",np.max(testY), np.max(trainX), np.min(trainY))

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

def reduce(tensor):
    #reduce the 4-dim tensor, the output from the
    #conv/maxpooling to 2-dim as input to the fully-connected layer
    features = tensor.get_shape()[1:4].num_elements() # The volume
    reduced = tf.reshape(tensor, [-1, features])
    return reduced, features

#def max_pooling(input_data,size): #max pooling layer
#    out = tf.nn.max_pool(value=input_data, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')
#    return out

# Create model
def multilayer_perceptron(x, weights, biases):
    # conv1
    x = tf.reshape(x, [-1, 112, 92, 1])
    conv1 = tf.nn.conv2d(
      input=x,
      filter=weights['h1'],
      #kernel_size=[5, 5],
      strides = [1,1,1,1],
      padding='SAME')
      #activation=tf.nn.relu
    
    conv1 = tf.nn.max_pool(value = conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    conv1 = tf.nn.relu(conv1)

    # conv2
    conv1 = tf.reshape(conv1, [-1, 112/2, 92/2, 16])
   
    conv1 = tf.nn.conv2d(
      input=conv1,
      filter=weights['h2'],
      #kernel_size=[5, 5],
      strides = [1,1,1,1],
      padding='SAME')
      #activation=tf.nn.relu
    
    conv1 = tf.nn.max_pool(value = conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    conv1 = tf.nn.relu(conv1)

    Xi, features = reduce(conv1)
    #print("-----", Xi.shape)
    # Hidden layer with RELU activation
    layer_1 = tf.matmul(Xi, clip_grad(weights['h3'],5.5))#+biases['b1']
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, clip_grad(weights['out'], 5.5))#+biases['out']
    #out_layer = tf.layers.dropout(inputs=out_layer,rate=0.5,training=True)
    return out_layer

# Store layers weight & bias
weights = {
     # filter size F, F, channel#, filter# K
    'h1': tf.Variable(tf.truncated_normal([5,5,1,16],stddev=0.01)),
    'h2': tf.Variable(tf.truncated_normal([5,5,16,36],stddev=0.01)),
    'h3': tf.Variable(tf.truncated_normal([112*92*36/4/4, n_hidden_1],stddev=0.01)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_classes],stddev=0.01))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x_image, weights,biases)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver(max_to_keep=100)




# Running first session
print("Starting 2nd session...")
with tf.Session() as sess:

    tf.initialize_all_variables().run() 
    real_accu = []
    ideal_accu = []


    #print("------Testing inputs ------------")
    #print("testing shape:", trainX.shape, testX.shape, trainY.shape, testY.shape)
    #print(trainX, trainY)
    #print(testX, testY)

    # Run the initializer
    sess.run(init)

    # Training cycle
    accuracies = []
    for epoch in range(epoch_num):
        save_path = model_path + str(epoch+1) + file_ending
        #avg_cost = 0.

        # Restore model weights from previously saved model
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        ideal_accu.append(accuracy.eval({x: testX, y: testY}))

        print("Ideal Accuracy:", ideal_accu[-1])

        # update weight values
        newh1 = weights['h1'].eval()
        newh2 = weights['h2'].eval()
        newh3 = weights['h3'].eval()
        newout = weights['out'].eval()
        saveh1 = np.copy(newh1) # deep copy
        saveh2 = np.copy(newh2)
        saveh3 = np.copy(newh3)
        saveout = np.copy(newout)
        #newh1 += gaussian_matrix['h1'][epoch]
        #newout += gaussian_matrix['out'][epoch]
        #print(newh2)
        print("weight layer 1 max-min:",newh1.max(),newh1.min())
        print("weight layer 2 max-min:",newh2.max(),newh2.min())
        print("weight layer 3 max-min:",newh3.max(),newh3.min())
        print("weight layer 4 max-min:",newout.max(),newout.min())
        
        sample_accu = []
        
        # loop for # of normal distribution samples
        for sample in range(num_of_samples):
            for i in range(0,saveh1.shape[0]):
                for j in range(0,saveh1.shape[1]):
                    for k in range(0, saveh1.shape[3]):
                        tmp = saveh1[i][j][0][k] 
                    #print("==", i, '--', j, ':',)
                        tmp = check_distribution(tmp, .0201, -.0201,  distribution, distribution_max, distribution_min, distribution_stage)
                        newh1[i][j][0][k] = tmp

            for i in range(0,saveh2.shape[0]):
                for j in range(0,saveh2.shape[1]):
                    for k in range(0, saveh2.shape[2]):
                        for l in range(0, saveh2.shape[3]):
                            tmp = saveh2[i][j][k][l]
                            tmp = check_distribution(tmp, .0201, -.0201,  distribution, distribution_max, distribution_min, distribution_stage)
                            newh2[i][j][k][l] = tmp

            for i in range(0,saveh3.shape[0]):
                for j in range(0,saveh3.shape[1]):
                    tmp = saveh3[i][j] 
                    tmp = check_distribution(tmp, .0201, -.0201,  distribution, distribution_max, distribution_min, distribution_stage)
                    newh3[i][j] = tmp

            for i in range(0,saveout.shape[0]):
                for j in range(0,saveout.shape[1]):
                    tmp = saveout[i][j] 
                    tmp = check_distribution(tmp, .0201, -.0201,  distribution, distribution_max, distribution_min, distribution_stage)
                    newout[i][j] = tmp
            weights['h1'].assign(newh1).eval()
            weights['h2'].assign(newh2).eval()
            weights['h3'].assign(newh3).eval()
            weights['out'].assign(newout).eval()

            #print("after-----",newh2)
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

    



