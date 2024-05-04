# Manifold Analysis for Equivariant Models





Requires the following modifications in  packages to work:

1. In autograd, remove _np.int wherever an error pops up
2. In file autograd/differential_operators.py , replace *from inspect import getargspec* with *from inspect import getfullargspec as getargspec*