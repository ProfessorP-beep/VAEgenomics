#I am adapting the code from Brandon Yeoph and the Liquid Brain Channel on youtube. Link to the original markdown below.
# url: https://github.com/brandonyph/LiquidBrain_Scripts/blob/main/VariationalAutoEncoder/VAE2.Rmd

#Also using scripts and examples from Tensorflow
# https://tensorflow.rstudio.com/tutorials/quickstart/beginner
# https://tensorflow.rstudio.com/install/

##### Installing tensorflow by command line (Terminal) ####

#You'll first want to install tensorflow using the terminal. Copy the pip strings without the hashtag for each step
# Requires the latest pip
# pip install --upgrade pip
# python.exe -m pip install --upgrade pip

## Current stable release for CPU and GPU
# pip install tensorflow

install.packages("tensorflow")

#Make sure you download Rtools 4.2 https://cran.r-project.org/bin/windows/Rtools/

#### First time isntallation of keras and tensorflow in R console ####
#First time install and load of tensorflow and keras
#If running on Windows make sure you're running R as admin
devtools::install_github("rstudio/tensorflow") #updated all dependent packages

#Install Python to configure R for tensorflow - just need to do this once
library(reticulate)
path_to_python <- install_python()
virtualenv_create("r-reticulate", python = path_to_python)

#Installing keras with the following lines should also finish installing tensorflow
#Also installs scipy and tensorflow-datasets - handy!
#This should install keras if it is not already in your installed packages.
##otherwise it does nothing. Should only need to do this once in fresh R env before loading any libraries
if(! 'keras' %in% installed.packages()){
  install.packages('keras')
}

library(keras)
install_keras(envname = "r-reticulate")

#Confirm installation
library('tensorflow')
tf$constant("Hello Tensorflow!")

#load libraries
library('keras') #for ML pipeline
library('tensorflow') #for ML pipeline
library('magrittr') #for piping

####Load tensorflow practice MNIST dataset - converts sample data from integers -> floating-point numbers ####
# mnist <- dataset_mnist()
# x_train <- mnist$train$x
# y_train <- mnist$train$y
# x_test <- mnist$test$x
# y_test <- mnist$test$y

#Same as commented out code above but in one line 
c(c(x_train, y_train), c(x_test, y_test)) %<-% keras::dataset_mnist()

#rescale the training and test data from integers ranging from 0 - 255 to range 0 - 1
x_train <- x_train / 255
x_test <-  x_test / 255

#Build a sequential model by stacking layers.
model <- keras_model_sequential(input_shape = c(28, 28)) %>% #integer vector of dimensions
  layer_flatten() %>%
  layer_dense(128, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(10)

#In each example the model returns a vector of:
#logits (layer that feeds into a softmax or some other normalization (?))
#log odds (convert logistic regression, a probability based model, to a likelihood-based model)
predictions <- predict(model, x_train[1:2, , ])
predictions

#tf$nn$softmax function converts these logits to probabilities for each class
tf$nn$softmax(predictions)

# Define a loss function for training using loss_sparse_categorical_crossentropy()
# which takes a vector of logits and an integer index of which are TRUE and returns a scalar loss 
# for each example.
loss_fn <- loss_sparse_categorical_crossentropy(from_logits = TRUE)

loss_fn(y_train[1:2], predictions)




#### Notes ####
# Batch Size- number of samples that go forward and backward through the network
# epochs is when all the samples in test dataset go forward and backwward through the neural net - usually stochastic
# latent neurons are the ones compressing the data dimensions are usually 
# Determining intermediate dimensions are usually multiples of 8 for the number of neurons you want and for the epoch sizes
# Papers on similar problems people have tackled and use that parameter as a starting point 0 and adjust by 8ths
# Ideally want to use a step function but cannot do in examples because you can't use the back propogation algo which req to calc derivative of all the weights of the activation function
# Thats why the activation function is sigmoidal or relu, can test different activation functions
# objective function = loss function
# # about a quarter for test set and rest are training dataset