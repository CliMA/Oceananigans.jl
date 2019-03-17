# Oceananigans.jl

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://mit-license.org/)
[![Ask us anything](https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg)](https://github.com/climate-machine/Oceananigans.jl/issues/new)

| **Documentation**             | **Build Status** (CPU, GPU, Windows)                                                                                 | **Code coverage**                                                                   |
|:------------------------------|:---------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
| [![docs][docs-img]][docs-url] | [![travis][travis-img]][travis-url] [![gitlab][gitlab-img]][gitlab-url] [![appveyor][appveyor-img]][appveyor-url]    | [![coveralls][coveralls-img]][coveralls-url] [![codecov][codecov-img]][codecov-url] |

[docs-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-url]: https://climate-machine.github.io/Oceananigans.jl/latest/

[travis-img]: https://travis-ci.com/climate-machine/Oceananigans.jl.svg?branch=master
[travis-url]: https://travis-ci.com/climate-machine/Oceananigans.jl

[gitlab-img]: https://gitlab.com/JuliaGPU/Oceananigans-jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/Oceananigans-jl/commits/master

[appveyor-img]: https://ci.appveyor.com/api/projects/status/jd7kctgj3c0mt957?svg=true
[appveyor-url]: https://ci.appveyor.com/project/ali-ramadhan/oceananigans-jl

[coveralls-img]: https://coveralls.io/repos/github/climate-machine/Oceananigans.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/climate-machine/Oceananigans.jl?branch=master

[codecov-img]: https://codecov.io/gh/ali-ramadhan/Oceananigans.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/ali-ramadhan/Oceananigans.jl

A fast non-hydrostatic ocean model in Julia that can be run in 2 or 3 dimensions on CPUs and GPUs. The plan is to develop it as a stand-alone [large eddy simulation](https://en.wikipedia.org/wiki/Large_eddy_simulation) (LES) model which can be used as a source of training data for statistical learning algorithms and/or embedded within a global ocean model as a super-parameterization of small-scale processes, as in [Campin et al., 2011](https://www.sciencedirect.com/science/article/pii/S1463500310001496?via%3Dihub).

Our goal is to develop friendly and intuitive code allowing users to focus on the science and not on fixing compiler errors. Thanks to high-level, zero-cost abstractions that the Julia programming language makes possible, the model can have the same look and feel no matter the dimension or grid of the underlying simulation, or whether running on CPUs or GPUs.

### Getting help

If you are interested in using Oceananigans.jl or are trying to figure out how to use it, please feel free to ask us questions and get in touch! Check out the [examples](https://github.com/climate-machine/Oceananigans.jl/tree/master/examples) and [open an issue](https://github.com/climate-machine/Oceananigans.jl/issues/new) with the _question_ label if you have any questions, comments, suggestions, etc.

### Examples

Watch the deep convection example in action on a GPU ([YouTube link](https://www.youtube.com/watch?v=kpUrxnKKMjI)):
[![Watch the video](https://raw.githubusercontent.com/ali-ramadhan/ali-ramadhan.Github.io/master/img/surface_temp_3d_00130_halfsize.png)](https://www.youtube.com/watch?v=kpUrxnKKMjI)
