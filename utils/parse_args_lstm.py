"""
Parse command line and store result in params

Model : LSTM
"""
import argparse,copy
from collections import OrderedDict
p = argparse.ArgumentParser(description="Arguments for variational autoencoder")
parser = argparse.ArgumentParser()

#Model
parser.add_argument('-lr','--lr', action='store',default = 0.0001, help='Learning rate', type=float)
parser.add_argument('-dset','--dataset', action='store',default = '', help='Dataset', type=str)
parser.add_argument('-rd','--rnn_dropout', action='store',default = 0.5, help='Dropout after each RNN output layer', type=float)
parser.add_argument('-rl','--rnn_layers', action='store',default = 1, help='Number of RNN layers', type=int, choices = [1,2])
parser.add_argument('-rs','--rnn_size', action='store', default = 800, help='Hidden dimensions in RNN', type=int)
parser.add_argument('-nl','--nonlinearity', action='store',default = 'relu', help='Nonlinarity',type=str, choices=['relu','tanh','softplus','maxout'])

parser.add_argument('-ischeme','--init_scheme', action='store',default = 'uniform', help='Type of initialization for weights', type=str, choices=['uniform','normal','xavier','he'])
parser.add_argument('-mstride','--maxout_stride', action='store',default = 4, help='Stride for maxout',type=int)
parser.add_argument('-lky','--leaky_param', action='store',default =0., help='Leaky ReLU parameter',type=float)
parser.add_argument('-iw','--init_weight', action='store',default = 0.1, help='Range to initialize weights during learning',type=float)

#Optimization
parser.add_argument('-opt','--optimizer', action='store',default = 'adam', help='Optimizer',choices=['adam','rmsprop'])
parser.add_argument('-bs','--batch_size', action='store',default = 20, help='Batch Size',type=int)
parser.add_argument('-fg','--forget_bias', action='store',default = 10., help='Bias for forget gates', type=float)
#Regularization
parser.add_argument('-reg','--reg_type', action='store',default = 'l2', help='Type of regularization',type=str,choices=['l1','l2'])
parser.add_argument('-rv','--reg_value', action='store',default = 0.05, help='Amount of regularization',type=float)
parser.add_argument('-rspec','--reg_spec', action='store',default = '_', help='String to match parameters',type=str)

#Save/Load
parser.add_argument('-uid','--unique_id', action='store',default = 'uid',help='Unique Identifier',type=str)
parser.add_argument('-seed','--seed', action='store',default = 1, help='Random Seed',type=int)
parser.add_argument('-dir','--savedir', action='store',default = './chkpt', help='Prefix for savedir',type=str)
parser.add_argument('-ep','--epochs', action='store',default = 800, help='MaxEpochs',type=int)
parser.add_argument('-reload','--reloadFile', action='store',default = './NOSUCHFILE', help='Reload from saved model',type=str)
parser.add_argument('-params','--paramFile', action='store',default = './NOSUCHFILE', help='Reload parameters from saved model',type=str)
parser.add_argument('-sfreq','--savefreq', action='store',default = 50, help='Frequency of saving',type=int)

parser.add_argument('-vonly','--validate_only', action='store_true', help='Only build fxn for validation')
params = vars(parser.parse_args())

hmap       = OrderedDict() 
hmap['lr']='lr'
hmap['nonlinearity']='nl'
hmap['optimizer']='opt'
hmap['batch_size']='bs'
hmap['epochs']='ep'
hmap['rnn_size']='rs'
hmap['rnn_dropout']='rd'
hmap['reg_value']='rv'
hmap['forget_bias']='fb'
combined   = ''
for k in hmap:
    if k in params:
        if type(params[k]) is float:
            combined+=hmap[k]+'-'+('%.4e')%(params[k])+'-'
        else:
            combined+=hmap[k]+'-'+str(params[k])+'-'
params['unique_id'] = combined[:-1]+'-'+params['unique_id']
params['unique_id'] = 'LSTM_'+params['unique_id'].replace('.','_')
