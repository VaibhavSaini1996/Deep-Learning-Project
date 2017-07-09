# -*- coding: utf-8 -*-
import tensorflow as tf
import scipy.io as scp
import numpy as np
import random
from scipy import misc
import matplotlib.pyplot as plt
import datetime

config = tf.ConfigProto()
config.gpu_options.allocator_type = "BFC"
config.log_device_placement = True


# Step 1: IMPORTING THE DATASET
trainData = scp.loadmat('Data/train_32x32.mat')
testData=scp.loadmat('Data/test_32x32.mat')

# print(trainData['X'].shape)
# print(trainData['y'].shape)
# print(testData['X'].shape)
# print(testData['y'].shape)


# Step 2: Preprocessing the Dataset
# Changing the type of data to model understandable format

trainDataX = trainData['X'].astype('float32') / 128.0 - 1                                                                                                                     
testDataX = testData['X'].astype('float32') / 128.0 - 1 

trainDataY=trainData['y']
testDataY=testData['y']

# ONE HOT ENCODING

def OnehotEndoding(Y):
    Ytr=[]
    for el in Y:
        temp=np.zeros(10)
        if el==10:
            temp[0]=1
        else:
            temp[el] = 1
        Ytr.append(temp)

    return np.asarray(Ytr)

trainDataY = OnehotEndoding(trainDataY)
testDataY = OnehotEndoding(testDataY)

# Formatting the labels for in order to be model friendly
def transposeArray(data):
    xtrain = []
    trainLen = data.shape[3]
    
  
    for x in range(trainLen):
        xtrain.append(data[:,:,:,x])
      

    xtrain = np.asarray(xtrain)
    return xtrain

trainDataX = transposeArray(trainDataX)
testDataX = transposeArray(testDataX)

# print(trainDataX.shape)
# print(testDataX.shape)

# Step 2: Setting up Tensorflow

# Function to initialize weights.
def weight_variable(shape):
    """Args:
            shape: a list of 4 integer [patch size,patch size,channels,depth]"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Function to initialize Bias
def bias_variable(shape):
    """Args:
            shape: a list containing depth"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution Function. Stride of 1 is used.
def conv2d(x, W):
    """Args:
            input: matrix of input to the convolution layer 
            weights: weights Matrix"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Pooling Function. 2x2 max_pooling 
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

# x,y_ are the placeholders for input values for tensorflow

x = tf.placeholder(tf.float32, shape=[None, 32,32,3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Step 3: Making Convolution Model.

# First layer
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,32,32,3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Layer
W_conv2=weight_variable([5, 5, 32, 64])
b_conv2=bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
# Now that the image size has been reduced to 8x8, we add a fully-connected layer with 1024 neurons
# to allow processing on the entire image.
# We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.

W_fc1=weight_variable([8*8*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout. to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ReadOut Layer (a Softmax Layer)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv=tf.nn.softmax(logits)


with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = y_))


# Optimizer. Using Adams Optimizer for controlling learning rate than steepest gradient descent optimizer

with tf.name_scope('optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) 

# Correct Predictions
with tf.name_scope('correct_predictions'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

# accuracy
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create a summary to monitor cost tensor
tf.summary.scalar("cross_entropy", cross_entropy)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()


# Runnng the model
#config = tf.configProto()
#config.gpu_options.allocator_type = "BFC"
#config.log_device_placement = True


print('started at : ', str(datetime.datetime.now()))
with tf.Session(config = config) as sess:
    epoch=20000
    batch_size=64
    
    #train_writer = tf.train.SummaryWriter('/Users/prince/Desktop/Svhn/tensorflow',graph=tf.get_default_graph())

    train_writer = tf.summary.FileWriter('/tmp/mnist_logs',
                                      sess.graph)
    
    sess.run(tf.global_variables_initializer())
    
    
    p = np.random.permutation(range(len(trainDataX)))
    trX, trY = trainDataX[p], trainDataY[p]
  
    start = 0
    end = 0  
    for step in range(epoch):
        start = end
        end = start + batch_size

        if start >= len(trainDataX):
            start = 0
            end = start + batch_size

        if end >= len(trainDataX):
            end = len(trainDataX) - 1
            
        if start == end:
            start = 0
            end = start + batch_size
        
        inX, outY = trX[start:end], trY[start:end]
        #_, summary=sess.run([train_step, merged_summaries], feed_dict={x: inX, y_: outY, keep_prob: 0.5})
        
        _, summary = sess.run([train_step, merged_summary_op], feed_dict= {x: inX, y_: outY, keep_prob:0.5})
        train_writer.add_summary(summary, step)
        #train_step.run(feed_dict={x: inX, y_: outY, keep_prob: 0.5})
        if step % 500 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: inX, y_: outY, keep_prob:1})
            print("step %d, training accuracy %g"%(step, train_accuracy))
            
        
        
   # Splitting due to hardware limitations.  
    pp = np.random.permutation(range(len(testDataX)))
    tstX, tstY = testDataX[pp], testDataY[pp]
    batch_size = 50
    pstart = 0
    pend = 0  
    for pstep in range(10):
        pstart = pend
        pend = pstart + batch_size

        if pstart >= len(testDataX):
            pstart = 0
            pend = pstart + batch_size

        if pend >= len(testDataX):
            pend = len(testDataX) - 1
            
        if pstart == pend:
            pstart = 0
            pend = pstart + batch_size
        
        tsX, tsY = tstX[pstart:pend], tstY[pstart:pend]
       
        test_accuracy = accuracy.eval(feed_dict={x: tsX, y_: tsY, keep_prob:1})
        print("step %d, testing accuracy %g"%(pstep, test_accuracy))
    # print("test accuracy %g"%accuracy.eval(feed_dict={x: testDataX, y_:testDataY , keep_prob: 1.0}))
    

print('Ended at : ', str(datetime.datetime.now()))


# Results on Hp Laptop with intel i5 + 8GB Ram + 2 Gb Nvida GPU.(but only 1Gb is allocated to the tensor due to which i had to split my test set too.)
# if you have more than 4 Gb of GPU available feel free to run it once on complete test set just replace the above code from line 221 with:
#print("test accuracy %g"%accuracy.eval(feed_dict={x: testDataX, y_:testDataY , keep_prob: 1.0}))

# For 20000 epochs and a batch size of 64
# Testing Accuracy 90%
# total time 20 mins

#For 10000 epochs and a batch size of 16
#Testing Accuracy 86%
#Total time 10 mins
