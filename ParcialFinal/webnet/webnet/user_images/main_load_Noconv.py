'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
import time
import cv2
import numpy
# Import MINST data
import sys

import tensorflow as tf

t = time.time()
# Parameters
learning_rate = 0.001 # 0.001
training_epochs = 2 # 15
batch_size = 10 # 
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer num features # 256 
n_hidden_2 = 256 # 2nd layer num features # 256
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
    return tf.matmul(layer_2, weights['out']) + biases['out']

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name="w_h1"),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name="w_h2"),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]),name="w_out")
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]),name="b_h1"),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]),name="b_h2"),
    'out': tf.Variable(tf.random_normal([n_classes]),name="b_out")
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()
# Launch the graph
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    '''
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
            
    print "Optimization Finished!"
    '''
    # Restore variables from disk.
    saver.restore(sess, "./variablesNormal.ckpt")
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

    #image = cv2.imread("./Samples/"+sys.argv[1]+".jpg",0)
    image = cv2.imread("./digit.jpg",0)
    image = image.astype(numpy.float32)
    image = numpy.multiply(image, 1.0 / 255.0)
    x_test = image.flatten()
    
    #x_test, batch_ys = mnist.train.next_batch(1)
    final = sess.run(pred,{x:[x_test]})
    #print("\nNoT Convolutional: ")	
    #print(final)    
    print "\nNormal Network  : ", numpy.argmax(final), '\n'
