# MLP Example

A simple example showing how to build an experiment using theanomodels.  This includes a simple baseline model, a model with batch-normalization[1], and a model with layer-normalization[2].

* To train a single module, execute command `python train.py`.  This will run a series of 6 experiments sequentially and output the results into a new directory `output`.
* To reproduce the results in `example-results.ipynb`, execute command `python run_gridsearch.py`, and then go get some coffee. 

[1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015). http://arxiv.org/abs/1502.03167

[2] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer Normalization." arXiv preprint arXiv:1607.06450 (2016). http://arxiv.org/abs/1607.06450
