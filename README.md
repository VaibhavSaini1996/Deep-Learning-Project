# Deep-Learning-Project
Training a Convolutional Neural Network on the SVHN Dataset. 

This project is a tensorflow implementation of CNN to decode a sequence of digits. The model is trained on Google Street View Images Dataset known as [SVHN dataset](http://ufldl.stanford.edu/housenumbers/). 

## Libraries and Dependencies
For the above code to work following Python3.5 libraries are required:
* matplotlib
* numpy
* PIL
* tensorflow
* scipy
* six
* sklearn

## Problem Statement
The goal is to train a model so that it can recognise digits (0-9) in an image with good accuracy.


The task involved are following:
  1. Download and load training	and	testing	datasets (.mat)	file in memory
  2. This will give a dictionary of two variables:
      -'X' : a 4D matrix of image
      -'y' : labels for corresponding image.
  3. Reshape the images in format of [no.images, height, width, channels] for simplification purpose.
  4. One hot encode the labels.
  5. Create a Convolutional nueral network.
  6. Train the network with data.
  7. Find Accuracy and Optimize it for better results.
 
## Algorithms and Techniques
The algorithm I used is a Convolutional Nueral Network which consists of multiple layers of small neuron collections which process portions of the input image. The output of one layer is feeded as input to the next layer.
A CNN architecture is formed by a stack of distinct layers such as:
  1. Convolution Layer: Each convolutional neuron processes data only for its receptive field.
                        Tiling allows CNNs to tolerate translation of the input image.The convolution operation reduces the                           number of free parameters and improve generalization.
  2. ReLu {Rectified linear Units}: It will apply an element wise activation function.
  3. Pooling Layer: This layer performs the downsampling operation along height and width, resulting in reduced volume.
  4. Fully Connected Layer: This layer will finally computer the output variables which is of the type [1x1x10].
  5. Loss Layer: This layer specifies hoe the network training penalizes the deviation between the predicted and true labels and is normally the last layer in the network. Softmax function is applied in this layer.
  
Certain parameters are are tuned in order to improve the results are:
 --Kernel Size, Stride, Padding, Training Length,Batch Size, Learning Rate, Dropout Probability. 

## Results:
These are the results that I obtained on my machine with GPU support.

For 10000 epochs and a batch size of 16
Testing Accuracy 86%
Total time 10 mins

For 20000 epochs and a batch size of 64
Testing Accuracy 90%
total time 20 mins

Time to get the results may vary from machine to machine.

## Conclusions:
This is what i conclude from doing this project:
  1. Stochastic gradient descent give comparable accuracy and decreases training time by lot
  2. More the hidden layers, more the training time. We can make our model deeper for better results.
  3. A better system with a very powerful GPU helps a lot in computaions and faster results.

 Thanks for reading :)

## Steps to run the code

Step 1: Download the dataset URL. 
       training : http://ufldl.stanford.edu/housenumbers/train_32x32.mat
       testing : http://ufldl.stanford.edu/housenumbers/test_32x32.mat
       
Step 2: Run SVHN_model.py file.
  -make sure you give the correct path of training and testing data as mentioned in the code.
  
## References:
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011.[PDF](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)

Ian J. Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, and Vinay Shet (2013). Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks.[cs.CV](https://arxiv.org/abs/1312.6082)

Pierre Sermanet, Soumith Chintala, and Yann LeCun (2012). Convolutional Neural Networks Applied to House Numbers Digit Classification. [Link](https://arxiv.org/abs/1204.3968v1)

https://github.com/hangyao/street_view_house_numbers

  

  
