'''
Save and Restore a model using TensorFlow.
Author:y Ji Li
This file is for ORL dataset, 1 layer DNN implementation training and save model 
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf

# Parameters
learning_rate = 0.1 
batch_size = 20
model_path = "./nn/NN"
file_ending = ".ckpt"
epoch_num = 20 

# Network Parameters
n_hidden_1 = 1000 # 1st layer number of features
n_input = 112*92 # MNIST data input (img shape: 28*28)
n_classes = 20 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#load dataset
data = np.load('ORL_faces.npz') #assumption is that the data is unzipped
trainX = data['trainX']
testX = data['testX']
trainY= data['trainY']
testY= data['testY']

# reorganize rows
train = np.concatenate((trainX, testX), axis = 0)
test = np.concatenate((trainY, testY), axis = 0)
idx = np.random.randint(train.shape[0],size=80)
trainX = []
trainY = []
testX = []
testY = []
for i in range(0,400):
    if i not in idx:
        trainX.append(train[i])
        trainY.append(test[i])
    else:
        testX.append(train[i])
        testY.append(test[i])

trainX = np.asarray(trainX)
testX = np.asarray(testX)

# process label to one hot
trainY_onehot = []
for i in trainY:
    trainY_onehot.append([0]*n_classes)
    trainY_onehot[-1][int(i)] = 1
trainY = np.asarray(trainY_onehot)

testY_onehot = []
for i in testY:
    testY_onehot.append([0]*n_classes)
    testY_onehot[-1][int(i)] = 1
testY = np.asarray(testY_onehot)

# scaling
trainX = (trainX-np.mean(trainX))/256
testX = (testX-np.mean(testX))/256

# save test set for later evaluation
np.save('testX',testX)
np.save('testY',testY)

#print(":::",np.max(testY), np.max(trainX), np.min(trainY))

#images = tf.constant(train_data, dtype=tf.float32) # X is a np.array
#labels = tf.constant(train_labels, dtype=tf.int32)   # y is a np.array

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


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.matmul(x, clip_grad(weights['h1'],5.5))#+biases['b1']
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, clip_grad(weights['out'], 5.5))#+biases['out']
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

    #print("------Testing inputs ------------")
    print("testing shape:", trainX.shape, testX.shape, trainY.shape, testY.shape)
    #print(trainX, trainY)
    #print(testX, testY)

    print("------staring training eval etc--")

    # Run the initializer
    sess.run(init)

    # Training cycle
    accuracies = []
    for epoch in range(epoch_num):
        print("--in epoch ", epoch)
        save_path = model_path + str(epoch+1) + file_ending
        avg_cost = 0.
        total_batch = int(trainX.shape[0]/batch_size)

        # Loop over all batches
        for i in range(total_batch):
            #batch_x, batch_y = tf.train.batch([images, labels], batch_size=batch_size, capacity=300, enqueue_many=True)
            #print("--in ",i,"-th batch")
            training_idx = np.random.randint(trainX.shape[0],size=batch_size)
            batch_x = trainX[training_idx]
            batch_y = trainY[training_idx]

            #print(training_idx.shape,batch_x.shape, batch_y.shape)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                         y: batch_y})
            # Compute average loss
            #avg_cost += c / total_batch
        # Display logs per epoch step
        #if epoch % display_step == 0:
        #    print("Epoch:", '%04d' % (epoch+1), "cost=", \
        #        "{:.9f}".format(avg_cost))
         
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: testX, y: testY}))
        accuracies.append(accuracy.eval({x: testX, y: testY})) 

        # Save model weights to disk
        save_path = saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    print("First Optimization Finished!")
    print("Final accuracies:", accuracies)


