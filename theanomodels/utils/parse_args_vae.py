"""
Parse command line and store result in params
"""
import argparse,copy
from collections import OrderedDict
p = argparse.ArgumentParser(description="Arguments for variational autoencoder")
parser = argparse.ArgumentParser()

#Model specification
parser.add_argument('-ph','--p_dim_hidden', action='store', default = 200, help='Hidden dimensions (in p)', type=int)
parser.add_argument('-pl','--p_layers', action='store',default = 2, help='#Layers in Generative Model', type=int)
parser.add_argument('-ds','--dim_stochastic', action='store',default = 50, help='Stochastic dimensions', type=int)
parser.add_argument('-ql','--q_layers', action='store',default = 2, help='#Layers in Recognition Model', type=int)
parser.add_argument('-qh','--q_dim_hidden', action='store', default = 400, help='Hidden dimensions (in q)', type=int)

parser.add_argument('-idrop','--input_dropout', action='store',default = 0.1, help='Dropout at input',type=float)
parser.add_argument('-nl','--nonlinearity', action='store',default = 'softplus', help='Nonlinarity',type=str, choices=['relu','tanh','softplus','maxout'])
parser.add_argument('-ischeme','--init_scheme', action='store',default = 'uniform', help='Type of initialization for weights', type=str, choices=['uniform','normal','xavier','he'])
parser.add_argument('-mstride','--maxout_stride', action='store',default = 4, help='Stride for maxout',type=int)
parser.add_argument('-lky','--leaky_param', action='store',default =0., help='Leaky ReLU parameter',type=float)
parser.add_argument('-iw','--init_weight', action='store',default = 0.1, help='Range to initialize weights during learning',type=float)

#Optimization
parser.add_argument('-dset','--dataset', action='store',default = '', help='Dataset', type=str)
parser.add_argument('-lr','--lr', action='store',default = 8e-4, help='Learning rate', type=float)
parser.add_argument('-opt','--optimizer', action='store',default = 'adam', help='Optimizer',choices=['adam','rmsprop'])
parser.add_argument('-bs','--batch_size', action='store',default = 200, help='Batch Size',type=int)

#Setup 
parser.add_argument('-viz','--visualize_model', action='store',default = False,help='Visualize Model',type=bool)
parser.add_argument('-uid','--unique_id', action='store',default = 'uid',help='Unique Identifier',type=str)
parser.add_argument('-seed','--seed', action='store',default = 1, help='Random Seed',type=int)
parser.add_argument('-dir','--savedir', action='store',default = './chkpt', help='Prefix for savedir',type=str)
parser.add_argument('-ep','--epochs', action='store',default = 3000, help='MaxEpochs',type=int)
parser.add_argument('-reload','--reloadFile', action='store',default = './NOSUCHFILE', help='Reload from saved model',type=str)
parser.add_argument('-params','--paramFile', action='store',default = './NOSUCHFILE', help='Reload parameters from saved model',type=str)
parser.add_argument('-sfreq','--savefreq', action='store',default = 100, help='Frequency of saving',type=int)
parser.add_argument('-infm','--inference_model', action='store',default = 'single', help='Inference Model',type=str, choices=['single'])

#Regularization
parser.add_argument('-reg','--reg_type', action='store',default = 'l2', help='Type of regularization',type=str,choices=['l1','l2'])
parser.add_argument('-rv','--reg_value', action='store',default = 0.01, help='Amount of regularization',type=float)
parser.add_argument('-rspec','--reg_spec', action='store',default = '_', help='String to match parameters (Default is generative model)',type=str)
params = vars(parser.parse_args())

hmap       = OrderedDict() 
hmap['lr']='lr'
hmap['q_dim_hidden']='qh'
hmap['p_dim_hidden']='ph'
hmap['dim_stochastic']='ds'
hmap['p_layers']='pl'
hmap['q_layers']='ql'
hmap['nonlinearity']='nl'
hmap['optimizer']='opt'
hmap['batch_size']='bs'
hmap['epochs']='ep'
hmap['inference_model']='infm'
hmap['input_dropout']='idrop'
hmap['reg_type']    = 'reg'
hmap['reg_value']   = 'rv'
hmap['reg_spec']    = 'rspec'
combined   = ''
for k in hmap:
    if k in params:
        if type(params[k]) is float:
            combined+=hmap[k]+'-'+('%.4e')%(params[k])+'-'
        else:
            combined+=hmap[k]+'-'+str(params[k])+'-'
params['unique_id'] = combined[:-1]+'-'+params['unique_id']
params['unique_id'] = 'VAE_'+params['unique_id'].replace('.','_')
