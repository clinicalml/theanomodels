import os

exptdir = 'example'
script = 'train_mlp_mnist.py'
outputdir = 'output'
session = exptdir
savedir = os.path.join(outputdir,exptdir)
gpu = 0 #gpu selector

theano_flags='device=gpu{gpuid}'
run_flags=['--savedir=%s'%savedir,
           '--epochs=50',
           '--savefreq=50',
           '--evalfreq=1',
           '--nonlinearity=sigmoid',
           '--nlayers=2',
           '--dim_hidden=1000',
           '--init_scheme=normal',
           '--optimizer=adam',
           '--input_dropout=0.1',
           '--lr=0.01', #learning rate
           '-rv 0', #regularization strength (defaults to L2 regularization)
           ]

# grid of parameters settings
batchsizes = [5,100]
normalizations = [None,'batchnorm','layernorm']
seeds = range(1,2)

# setup args from grid 
var_flags = {}
for bs in batchsizes:
    for nm in normalizations:
        for s in seeds:
            if nm is not None:
                norm = '--%s=True' % nm
                flag = '%s_' % nm
            else:
                norm = ''
                flag = ''
            flag = '%sbs%s_seed%s' % (flag,bs,s)
            val = '%s -bs %s -seed %s' % (norm,bs,s)
            var_flags[flag] = val

# run each experiment sequentially
for tag,val in var_flags.iteritems():
    cmd='THEANO_FLAGS="{theano_flags},exception_verbosity=high" python {script} {run_flags}'.format(
            theano_flags=theano_flags.format(gpuid=gpu),
            script=script,
            run_flags=' '.join(run_flags+[val]))
    #session_name = '%s_%s' % (session,tag)
    #cmd = 'tmux new -d -s {session_name}; tmux send-keys -t {session_name} "{cmd}" Enter'.format(**locals())
    #cmd = 'tmux kill-session -t {session_name}'.format(**locals())
    print cmd
    os.system(cmd)
    

