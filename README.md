# theanomodels
A lightweight wrapper around theano for rapid-prototyping of machine learning models. 

## Requirements
python2.7

[Theano](https://github.com/Theano/Theano)

## Overview
This wrapper takes care of some basic bookkeeping 
a machine learning model built in theano. It is designed to be lightweight, easy to extend and minimalistic. 

In this case, 
this package defines a base model that can be inherited from.
* save weights/model parameters intermittently 
* load from checkpoints 
* use common initialization schemes, batch normalization, dropout etc. 

The wrapper can be trivially extended to include more complex functionality depending on the project.

## Installation
Clone this repository to your the root of your project directory.  Install using pip in editable mode:
```
git clone https://github.com/clinicalml/theanomodels
cd theanomodels
pip install -e .
```
To install locally, use `pip install -e . --user` instead.

## Defining a Model

The user provides a dictionary of parameters that is consistently associated
with a model. It contains configurations like the size of hidden units etc. 
It is stored in self.params

The other important dictionary is self.tWeights which is a dictionary containing all the weights in the model
and self.optWeights which contains all the optimization parameters.

Defining a model involves a few steps: 

- Inherit a class from BaseModel 
- Define self.createParams() which is expected to return an Ordered dictionary (self.npWeights) containing numpy weight matrices representing the model parameters. The keys of the dictionary will be used to represent the names of each of the parameters in the model.
- The base class turns the aforementioned numpy matrices into theano shared variables and places them in a dictionary (self.tWeights) with the same keys as self.npWeights
- Define self.buildModel() using theano shared variables (from self.tWeights) and create theano functions to train/evaluate/introspect and anything else you might be interested in
- Define other functions like self.learn to perform learning in the model, self.evaluate for validation, self.sample for generative models etc.

A few examples are provided in models/static.py (Variational Autoencoder) and models/temporal.py (LSTM)

## Optimization

When building the model, use self.setupOptimizer and specify one of rmsprop/adam during model initialization.
The opimization modules are defined in utils/optimizer.py. They use parameter names to determine
regularization and define updates to parameters with variable learning rates. 

**IMPORTANT**: The T.grad function is called on the model parameters which are defined as variable in self.tWeights whose name
contains a string in ['W_','b_','U_','_W','_b','_U']. The code prints out the variables that the cost is
being differentiated with respect to. It is worthwhile to sanity check this while building the model to make sure the relevant parameters are being differentiated.

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


## Contributors
* Rahul G. Krishnan (rahul@cs.nyu.edu)
* Justin Mao-Jones
