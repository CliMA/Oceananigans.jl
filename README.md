# Oceananigans.jl

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://mit-license.org/)
[![Ask us anything](https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg)](https://github.com/ali-ramadhan/Oceananigans.jl/issues)

| **Documentation**             | **Build Status** (CPU, GPU, Windows)                                                                                 | **Code coverage**                                                                   |
|:------------------------------|:---------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
| [![docs][docs-img]][docs-url] | [![travis][travis-img]][travis-url] [![gitlab][gitlab-img]][gitlab-url] [![appveyor][appveyor-img]][appveyor-url]    | [![coveralls][coveralls-img]][coveralls-url] [![codecov][codecov-img]][codecov-url] |

[docs-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-url]: https://ali-ramadhan.github.io/Oceananigans.jl/latest

[travis-img]: https://travis-ci.com/ali-ramadhan/Oceananigans.jl.svg?branch=master
[travis-url]: https://travis-ci.com/ali-ramadhan/Oceananigans.jl

[gitlab-img]: https://gitlab.com/JuliaGPU/Oceananigans-jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/Oceananigans-jl/commits/master

[appveyor-img]: https://ci.appveyor.com/api/projects/status/jd7kctgj3c0mt957?svg=true
[appveyor-url]: https://ci.appveyor.com/project/ali-ramadhan/oceananigans-jl

[coveralls-img]: https://coveralls.io/repos/github/ali-ramadhan/Oceananigans.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/ali-ramadhan/Oceananigans.jl?branch=master

[codecov-img]: https://codecov.io/gh/ali-ramadhan/Oceananigans.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/ali-ramadhan/Oceananigans.jl

A fast non-hydrostatic ocean model in Julia that can be run in 2 or 3 dimensions on CPUs and GPUs. The plan is to develop it as a stand-alone [large eddy simulation](https://en.wikipedia.org/wiki/Large_eddy_simulation) (LES) model which can be used as a source of training data for statistical learning algorithms and/or embedded within a global ocean model as a super-parameterization of small-scale processes, as in [Campin et al., 2011](https://www.sciencedirect.com/science/article/pii/S1463500310001496?via%3Dihub).

Our goal is to develop friendly and intuitive code allowing users to focus on the science and not on fixing compiler errors. Thanks to high-level, zero-cost abstractions that the Julia programming language makes possible, the model can have the same look and feel no matter the dimension or grid of the underlying simulation, or whether running on CPUs or GPUs.
