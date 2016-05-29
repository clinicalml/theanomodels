# theanomodels
A lightweight wrapper around theano for rapid-prototyping of machine learning models. 

## Requirements
python2.7

[Theano](https://github.com/Theano/Theano)

## What does this do?
This wrapper takes care of some basic bookkeeping 
for any model and leaves the rest up to the user. In this case, 
this package defines a base model that can be inherited from. The
base model provides functionality to (1) save weights
intermittently (2) load from checkpoints (3) use common initialization schemes 
for matrices used in deep learning. The rest is up to the user to specify in theano.

## Defining a Model

The user provides a dictionary of parameters that is consistently associated
with a model. It contains configurations like the size of hidden units etc. 
It is stored in self.params

The other important dictionary is self.tWeights which is a dictionary containing all the weights in the model
and self.optWeights which contains all the optimization parameters.

Defining a model involves a few steps: 

- Inherit a class from BaseModel 
- Define self.createParams() which is expected to return dictionary (self.npWeights) containing numpy weight matrices representing the model parameters. The keys of the dictionary will be used to represent the names of each of the parameters in the model.
- The base class turns the aforementioned numpy matrices into theano shared variables and saves them in a dictionary (self.tWeights) with the same keys as self.npWeights
- Define self.buildModel() using theano shared variables (from self.tWeights) and create theano functions to train/evaluate/introspect and anything else you might be interested in
- Define other functions like self.learn to perform learning in the model, self.evaluate for validation, self.sample for generative models etc.

A few examples are provided in models/static.py (Variational Autoencoder) and models/temporal.py (LSTM)

## Optimization Strategies

When building the model, use self.setupOptimizer and specify one of rmsprop/adam during model initialization.
The opimization modules are defined in utils/optimizer.py. They use the names of the parameters to determine
whether or not to add regularization

**IMPORTANT**: The T.grad function is called on the model parameters which are defined as variable in self.tWeights whose name
contains a string in ['W_','b_','U_','_W','_b','_U']. The code prints out the variables that the cost is
being differentiated with respect to. It is worthwhile to sanity check this while building the model.

The naming convention I usually use is 
'model_W_modelsubpart'. For example 'p_W_input' 


models/__init__.py contains a simple LSTM unit.
It requires the definition of parameters named: 'W_lstm', 'U_lstm' and 'b_lstm'

## Datasets
Included are datasets for MNIST/ Binarized MNIST and the Polyphonic Music Datasets.
The hope is to be able to retrieve datasets quickly from memory. To that end datasets/process.py
contains scripts to process the raw data into an HDF5 format that can easily be read from
during training time. datasets/load.py returns a dictionary containing the pre-split train/valid/test data.

If the datasets are very large and need special attention (like HDF5 Tables), you might have to modify
the process/load and the method inside the model that is used to perform learning. 

Data is typically represented as a N x dim_observation sized matrix. For time series data,
we specify a N x maxT x dim_observation sized tensor as the data and a N x maxT sized mask matrix for variable
length sequences. 

## Experimental Setup
See expt-bmnist for details on setting up an experiment. 

## Installation
Clone this repository to /path/to/theanomodels and then append the following to your ~/.profile or ~/.bashrc file
```
export PYTHONPATH=$PYTHONPATH:/path/to/theanomodels
```
