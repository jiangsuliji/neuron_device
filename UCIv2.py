'''
Save and Restore a model using TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
credit to: Aymeric Damien
Author:y Ji Li
This file is for UCI training and dumping model files at each epoch
'''

# This is for MNIST training but saves the results into model file

from __future__ import print_function
import numpy as np
import tensorflow as tf

# Parameters
learning_rate = 0.5
batch_size = 100
model_path = "./UCI/nn/NN"
file_ending = ".ckpt"
epoch_num = 40 

# Network Parameters
n_hidden_1 = 36 # 1st layer number of features
n_input = 64 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


train_data = np.load('UCI/training_data.npy')
train_labels_tmp = np.load('UCI/training_label.npy')
train_labels = []
for i in train_labels_tmp:
    train_labels.append([0]*n_classes)
    train_labels[-1][int(i)] = 1
train_labels = np.asarray(train_labels)

#train_labels = tf.one_hot(indices=tf.cast(train_labels, tf.int32), depth=10)
eval_data = np.load('UCI/test_data.npy')
eval_labels_tmp = np.load('UCI/test_label.npy')
eval_labels = []
for i in eval_labels_tmp:
    eval_labels.append([0]*n_classes)
    eval_labels[-1][int(i)] = 1
eval_labels =  np.asarray(eval_labels)

    
#images = tf.constant(train_data, dtype=tf.float32) # X is a np.array
#labels = tf.constant(train_labels, dtype=tf.int32)   # y is a np.array

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.matmul(x, weights['h1'])#+biases['b1']
    layer_1 = tf.tanh(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out'])#+biases['out']
    out_layer = tf.layers.dropout(inputs=out_layer,rate=0.5,training=True)
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

    
    print("------Testing inputs ------------")
    print(train_data.shape, train_data)
    print(train_labels.shape, train_labels)
    #print(eval_data.shape, eval_labels.shape)
    #print(train_data[500])    
    #print(eval_labels.shape)    

    print("------staring training eval etc--")

    # Run the initializer
    sess.run(init)

    # Training cycle
    accuracies = []
    for epoch in range(epoch_num):
        print("--in epoch ", epoch)
        save_path = model_path + str(epoch+1) + file_ending
        avg_cost = 0.
        total_batch = int(train_data.shape[0]/batch_size)

        # Loop over all batches
        for i in range(total_batch):
            #batch_x, batch_y = tf.train.batch([images, labels], batch_size=batch_size, capacity=300, enqueue_many=True)
            #print("--in ",i,"-th batch")
            training_idx = np.random.randint(train_data.shape[0],size=batch_size)
            batch_x = train_data[training_idx]
            batch_y = train_labels[training_idx]
            
            # print(training_idx.shape,batch_x.shape, batch_y.shape)
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
        print("Accuracy:", accuracy.eval({x: eval_data, y: eval_labels}))
        accuracies.append(accuracy.eval({x: eval_data, y: eval_labels})) 

        # Save model weights to disk
        save_path = saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    print("First Optimization Finished!")
    print("Final accuracies:", accuracies)


