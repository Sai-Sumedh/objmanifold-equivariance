# Manifold Analysis for Equivariant Models

Course project for Applied Math 220, Spring 2024, Harvard SEAS

Compute manifold dimension, radius and classification capacity for a CNN and GCNN (group-equivariant convolutional network). Uses the manifold analysis methods developed in Chung et al, PRX 2018, Cohen et al, Nat. Comm. 2020, and Stephenson et al NeurIPS 2019.

Perform the analysis for GCNN- equivariant wrt cyclic groups of order 4,8 and 12.


Requires the following modifications in  packages to work:

1. In autograd, remove _np.int wherever an error pops up
2. In file autograd/differential_operators.py , replace *from inspect import getargspec* with *from inspect import getfullargspec as getargspec*
