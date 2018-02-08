from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
"""Runing 2layer NN for MNIST"""
tf.logging.set_verbosity(tf.logging.INFO)



def cnn_model_fn(features, labels, mode):
    """model func for CNN"""
    # Input layer
    input_layer = tf.reshape(features["x"], [-1, 28*28])
    
    
    # dense layer
    dense = tf.layers.dense(inputs=input_layer, units=300, activation=None)
    #dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # logits layer
    logits = tf.layers.dense(inputs=dense, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Loading training eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # np array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    print("------Testing inputs ------------")
    print(train_data.shape, train_data.size)
    print(eval_data.shape, eval_labels.shape)
    #print(train_data[500])    
    print(eval_labels)    

    print("------staring training eval etc--")

    # Creat the Estimator
    mnist_classifier = tf.estimator.Estimator(
           model_fn=cnn_model_fn, model_dir="/tmp/mnist_nn_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
           tensors=tensors_to_log, every_n_iter=500)
    
    accuracy_results = []
    for i in range(0,40):
        print("In Epoch", i, ":")
        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=1,
            shuffle=True)
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=None,
            #hooks=[logging_hook]
            )
    
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print("---",eval_results,"---")
        accuracy_results.append(eval_results["accuracy"])

    print("Final results:", accuracy_results)

if __name__ == "__main__":
    tf.app.run()
