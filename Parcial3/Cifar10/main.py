"""Functions for reading CIFAR10 data."""

import tensorflow as tf
import numpy as np
import time

t = time.time()


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def norm(data):
    data = data.astype(np.float32)
    data = np.multiply(data, 1.0 / 255.0)
    return data

def num2vec(data):
    vec = np.zeros((len(data),10), dtype=np.float32)
    for i in range(len(data)):
        tvec = np.zeros((10,), dtype=np.float32)
        tvec[data[i]] = 1
        vec[i] = tvec
        #print i, ' ... ' , tvec 
    return vec

def batch_sample(data,labels,batch_size):
    import random
    index = random.sample(range(len(data)), batch_size)
    x = np.zeros((batch_size,3072), dtype=np.float32)
    y = np.zeros((batch_size,10), dtype=np.float32)
    #print index
    for i in range(batch_size):
        y[i] = labels[index[i]]
        x[i] = data[index[i]]
    return x,y

""" load training data """
b1 = unpickle('./data/data_batch_1')
b2 = unpickle('./data/data_batch_2')
b3 = unpickle('./data/data_batch_3')
b4 = unpickle('./data/data_batch_4')
b5 = unpickle('./data/data_batch_5')
train_data = norm(np.vstack((b1['data'],b2['data'],b3['data'],b4['data'],b5['data'])))
train_labels = np.asarray(b1['labels'] + b2['labels'] + b3['labels'] + b4['labels'] + b5['labels'])
train_labels = num2vec(train_labels)

"""" load test data"""
test = unpickle('./data/test_batch')
test_data = norm(test['data'])
test_labels = num2vec(test['labels'])

#x,y = batch_sample(train_data,train_labels,1)

'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the CIFAR10 database 
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Parameters
learning_rate = 0.001 # 0.001
training_epochs = 3000 # 15
batch_size = 200# 100
display_step = 1
train_num_examples = len(train_data)

# Network Parameters
n_hidden_1 = 250 # 1st layer num features # 256 
n_hidden_2 = 150 # 2nd layer num features # 256
n_input = 3072 # Cifar10 data input (img shape: 32*32*3)
n_classes = 10 # Cifat10 total classes (0-9 classes)

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

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = batch_sample(train_data,train_labels,batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: test_data, y: test_labels})
    save_path = saver.save(sess, "./variables.ckpt")
    print "Model saved in file: ", save_path

end = time.time() - t
print ("-----------test!-----------")
print ("Tiempo empleado : ", end)
print ("learning: ", learning_rate )
print ("Training Epoch: ", training_epochs )
print ("batch_size: ", batch_size)
print ("train_num_examples: ", train_num_examples )
print ("n_hidden_1: ", n_hidden_1 )
print ("n_hidden_2: ", n_hidden_2 )
