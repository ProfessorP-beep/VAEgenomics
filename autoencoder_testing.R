#I am adapting the code from Brandon Yeoph and the Liquid Brain Channel on youtube. Link to the original markdown below.
# url: https://github.com/brandonyph/LiquidBrain_Scripts/blob/main/VariationalAutoEncoder/VAE2.Rmd

#Also using scripts and examples from Tensorflow
# https://tensorflow.rstudio.com/tutorials/quickstart/beginner

#First time install and load of tensorflow and keras
tensorflow::install_tensorflow() #requires a fresh R instance. Just restart session if already loaded any packages

#This should install keras if it is not already in your installed packages.
##otherwise it does nothing.
if(! 'keras' %in% installed.packages()){
install.packages('keras')
}

#load libraries
library('keras') #for ML pipeline
library('tensorflow') #for ML pipeline
library('magrittr') #for piping

#Load tensorflow practice MNIST dataset - converts sample data from integers -> floating-point numbers
c(c(x_train, y_train), c(x_test, y_test)) %>% keras::dataset_mnist()
x_train <- x_train / 255
x_test <-  x_test / 255




# Batch Size- number of samples that go forward and backward through the network
# epochs is when all the samples in test dataset go forward and backwward through the neural net - usually stochastic
# latent neurons are the ones compressing the data dimensions are usually 
# Determining intermediate dimensions are usually multiples of 8 for the number of neurons you want and for the epoch sizes
# Papers on similar problems people have tackled and use that parameter as a starting point 0 and adjust by 8ths
# Ideally want to use a step function but cannot do in examples because you can't use the back propogation algo which req to calc derivative of all the weights of the activation function
# Thats why the activation function is sigmoidal or relu, can test different activation functions
# objective function = loss function
# # about a quarter for test set and rest are training dataset