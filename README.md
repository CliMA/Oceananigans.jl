# Oceananigans.jl

[![Build Status](https://travis-ci.com/ali-ramadhan/Oceananigans.jl.svg?branch=master)](https://travis-ci.com/ali-ramadhan/Oceananigans.jl)
[![codecov](https://codecov.io/gh/ali-ramadhan/Oceananigans.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ali-ramadhan/Oceananigans.jl)

A fast non-hydrostatic _n_-dimensional ocean model based on the [MITgcm](https://github.com/MITgcm/MITgcm) algorithm [(Marshall et al., 1997)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/96JC02775) in Julia. The plan is to make it useful as a [large eddy simulation](https://en.wikipedia.org/wiki/Large_eddy_simulation) (LES) model or as a fast [super-parameterization](http://hannahlab.org/what-is-super-parameterization/) to be embedded within a global ocean model. As an embedded model it could resolve the sub-grid scale physics and communicate their effects back to the global model or act as a source of training data for statistical learning algorithms [(Campin et al., 2011)](https://www.sciencedirect.com/science/article/pii/S1463500310001496?via%3Dihub).

It can be used as a general-purpose ocean model in a hydrostatic or non-hydrostatic configuration. A big aim is to have a friendly and intuitive user interface allowing users to focus on the science and not on fixing compiler errors. Thanks to high-level zero-cost abstractions in Julia we think we can make the model look and behave the same no matter the dimension or grid of the underlying simulation.

 Just found out about the [Zen of Python](https://www.python.org/dev/peps/pep-0020/) which might be a good guide for this package
```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Now is better than never.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
```
