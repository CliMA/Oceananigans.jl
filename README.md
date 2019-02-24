# Oceananigans.jl

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://mit-license.org/)
[![Latest documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://ali-ramadhan.github.io/Oceananigans.jl/latest)
[![Travis-CI build status](https://travis-ci.com/ali-ramadhan/Oceananigans.jl.svg?branch=master)](https://travis-ci.com/ali-ramadhan/Oceananigans.jl)
[![GitLab CI GPU pipeline status](https://gitlab.com/JuliaGPU/Oceananigans-jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/Oceananigans-jl/commits/master)
[![Appveyor build status](https://ci.appveyor.com/api/projects/status/jd7kctgj3c0mt957?svg=true)](https://ci.appveyor.com/project/ali-ramadhan/oceananigans-jl)
[![Coveralls](https://coveralls.io/repos/github/ali-ramadhan/Oceananigans.jl/badge.svg?branch=master)](https://coveralls.io/github/ali-ramadhan/Oceananigans.jl?branch=master)
[![codecov](https://codecov.io/gh/ali-ramadhan/Oceananigans.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ali-ramadhan/Oceananigans.jl)
[![Ask us anything](https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg)](https://github.com/ali-ramadhan/Oceananigans.jl/issues)

A fast non-hydrostatic _n_-dimensional ocean model in Julia. The plan is to make it useful as a [large eddy simulation](https://en.wikipedia.org/wiki/Large_eddy_simulation) (LES) model or as a 2D/3D [super-parameterization](http://hannahlab.org/what-is-super-parameterization/) to be embedded within a global ocean model. As an embedded model it could resolve the sub-grid scale physics and communicate their effects back to the global model or act as a source of training data for statistical learning algorithms [(Campin et al., 2011)](https://www.sciencedirect.com/science/article/pii/S1463500310001496?via%3Dihub).

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
