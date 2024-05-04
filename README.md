# Manifold Analysis for Equivariant Models

Course project for Applied Math 220, Spring 2024, Harvard SEAS

Compute manifold dimension, radius and classification capacity for a CNN and GCNN (group-equivariant convolutional network). Uses the manifold analysis methods developed in Chung et al, PRX 2018, Cohen et al, Nat. Comm. 2020, and Stephenson et al NeurIPS 2019.

Perform the analysis for GCNN- equivariant wrt cyclic groups of order 4,8 and 12.


The following files train the models and compute manifold properties: 
1. cnn_mlp_classifier.ipynb
2. gcnn4_mlp_classifier.ipynb
3. gcnn8_mlp_classifier.ipynb
4. gcnn12_mlp_classifier.ipynb

To run these notebooks, download the rotated MNIST data into data/mnist-rot folder. Instructions to download rotated mnist are found here (borrowed from https://github.com/tscohen/gconv_experiments/tree/master?tab=readme-ov-file#readme):

```
$ cd [datadir]
$ wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
$ unzip mnist_rotation_new.zip 
$ rm mnist_rotation_new.zip
$ ipython /path/to/gconv_experiments/gconv_experiments/MNIST_ROT/mnist_rot.py -- --datadir=./
```

The corresponding models are defined in the models/ folder, with the g-convolution layer in layers/gconvlayer.py (credit to Theodosis, E., Helwani, K. and Ba, D., 2023. Learning Linear Groups in Neural Networks. arXiv preprint arXiv:2305.18552 for this layer).

The file make_figures.ipynb generates the figures included in the report.

Requires the following modifications in  packages to work:

1. In autograd, remove _np.int wherever an error pops up
2. In file autograd/differential_operators.py , replace *from inspect import getargspec* with *from inspect import getfullargspec as getargspec*
