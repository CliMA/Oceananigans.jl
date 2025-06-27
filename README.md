<!-- Title -->
<h1 align="center">
  Oceananigans.jl
</h1>

<!-- description -->
<p align="center">
  <strong>🌊 Fast and friendly ocean-flavored Julia software for simulating incompressible fluid dynamics in Cartesian and spherical shell domains on CPUs and GPUs. https://clima.github.io/OceananigansDocumentation/stable</strong>
</p>

<!-- Information badges -->
<p align="center">
  <a href="https://www.repostatus.org/#active">
    <img alt="Repo status" src="https://www.repostatus.org/badges/latest/active.svg?style=flat-square" />
  </a>
  <a href="https://mit-license.org">
    <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square">
  </a>
  <a href="https://github.com/CliMA/Oceananigans.jl/discussions">
    <img alt="Ask us anything" src="https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg?style=flat-square">
  </a>
  <a href="https://github.com/SciML/ColPrac">
    <img alt="ColPrac: Contributor's Guide on Collaborative Practices for Community Packages" src="https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet?style=flat-square">
  </a>
  <a href="https://doi.org/10.21105/joss.02018">
    <img alt="JOSS" src="https://joss.theoj.org/papers/10.21105/joss.02018/status.svg">
  </a>
</p>

<!-- Version and documentation badges -->
<p align="center">
  <a href="https://github.com/CliMA/Oceananigans.jl/releases">
    <img alt="GitHub tag (latest SemVer pre-release)" src="https://img.shields.io/github/v/tag/CliMA/Oceananigans.jl?include_prereleases&label=latest%20version&logo=github&sort=semver&style=flat-square">
  </a>
  <a href="https://clima.github.io/OceananigansDocumentation/stable">
    <img alt="Stable documentation" src="https://img.shields.io/badge/documentation-stable%20release-blue?style=flat-square">
  </a>
  <a href="https://clima.github.io/OceananigansDocumentation/dev">
    <img alt="Development documentation" src="https://img.shields.io/badge/documentation-in%20development-orange?style=flat-square">
  </a>
</p>

<!-- Downloads -->
<!-- counts downloads from individual IPs excluding bots (eg, CI) -->
<!-- see https://discourse.julialang.org/t/announcing-package-download-stats/69073 -->

<p align="center">
  <a href="https://juliapkgstats.com/pkg/Oceananigans">
    <img alt="Downloads per month" src="https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FOceananigans&query=total_requests&suffix=%2Fmonth&label=Downloads&style=flat-square">
  </a>
  <a href="https://juliapkgstats.com/pkg/Oceananigans">
    <img alt="Downloads per month" src="https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FOceananigans&query=total_requests&&label=Total%20Downloads&style=flat-square">
  </a>
</p>

<!-- CI/CD badges -->
<p align="center">
  <a href="https://buildkite.com/clima/oceananigans">
    <img alt="Buildkite CPU+GPU build status" src="https://img.shields.io/buildkite/4d921fc17b95341ea5477fb62df0e6d9364b61b154e050a123/main?logo=buildkite&label=Buildkite%20CPU%2BGPU&style=flat-square">
  </a>
  <a href="https://hub.docker.com/r/aliramadhan/oceananigans">
    <img alt="Docker build status" src="https://img.shields.io/docker/cloud/build/aliramadhan/oceananigans?label=Docker&logo=docker&logoColor=white&style=flat-square">
  </a>
</p>

Oceananigans is a fast, friendly, flexible software package for finite volume simulations of the nonhydrostatic
and hydrostatic Boussinesq equations on CPUs and GPUs.
It runs on GPUs (wow, [fast!](http://arxiv.org/abs/2309.06662)), though we believe Oceananigans makes the biggest waves
with its ultra-flexible user interface that makes simple simulations easy, and complex, creative simulations possible.

Oceananigans.jl is developed by the [Climate Modeling Alliance](https://clima.caltech.edu) and heroic external collaborators.

## Contents

- [Contents](#contents)
- [Installation instructions](#installation-instructions)
- [Running your first model](#running-your-first-model)
- [The Oceananigans knowledge base](#the-oceananigans-knowledge-base)
- [Citing and otherwise spreading the word](#citing-and-otherwise-spreading-the-word)
- [Contributing](#contributing)
- [Movies](#movies)
  - [Deep convection](#deep-convection)
  - [Free convection](#free-convection)
  - [Winds blowing over the ocean](#winds-blowing-over-the-ocean)
  - [Free convection with wind stress](#free-convection-with-wind-stress)
- [Performance benchmarks](#performance-benchmarks)

## Installation instructions

Oceananigans is a [registered Julia package](https://julialang.org/packages/). So to install it,

1. [Download Julia](https://julialang.org/downloads/) (version 1.9 or later).

2. Launch Julia and type

```julia
julia> using Pkg

julia> Pkg.add("Oceananigans")
```

This installs the latest version that's _compatible with your current environment_.
Don't forget to *be careful* 🏄 and check which Oceananigans you installed:

```julia
julia> Pkg.status("Oceananigans")
```

## Running your first model

Let's run a two-dimensional, horizontally-periodic simulation of turbulence using 128² finite volume cells for 4 non-dimensional time units:

```julia
using Oceananigans
grid = RectilinearGrid(CPU(), size=(128, 128), x=(0, 2π), y=(0, 2π), topology=(Periodic, Periodic, Flat))
model = NonhydrostaticModel(; grid, advection=WENO())
ϵ(x, y) = 2rand() - 1
set!(model, u=ϵ, v=ϵ)
simulation = Simulation(model; Δt=0.01, stop_time=4)
run!(simulation)
```

But there's more: changing `CPU()` to `GPU()` makes this code run on a CUDA-enabled Nvidia GPU.

Dive into [the documentation](https://clima.github.io/OceananigansDocumentation/stable/) for more code examples and tutorials.
Below, you'll find movies from GPU simulations along with CPU and GPU [performance benchmarks](https://github.com/clima/Oceananigans.jl#performance-benchmarks).

## The Oceananigans knowledge base

It's _deep_ and includes:

* [Documentation](https://clima.github.io/OceananigansDocumentation/stable) that provides
    * example Oceananigans scripts,
    * tutorials that describe key Oceananigans objects and functions,
    * explanations of Oceananigans finite-volume-based numerical methods,
    * details of the dynamical equations solved by Oceananigans models, and
    * a library documenting all user-facing Oceananigans objects and functions.
* [Discussions on the Oceananigans github](https://github.com/CliMA/Oceananigans.jl/discussions), covering topics like
    * ["Computational science"](https://github.com/CliMA/Oceananigans.jl/discussions/categories/computational-science), or how to science and set up numerical simulations in Oceananigans, and
    * ["Experimental features"](https://github.com/CliMA/Oceananigans.jl/discussions?discussions_q=experimental+features), which covers new and sparsely-documented features for those who like to live dangerously.

    If you've got a question or something, anything! to talk about, don't hesitate to [start a new discussion](https://github.com/CliMA/Oceananigans.jl/discussions/new?).
* The [Oceananigans wiki](https://github.com/CliMA/Oceananigans.jl/wiki) contains practical tips for [getting started with Julia](https://github.com/CliMA/Oceananigans.jl/wiki/Installation-and-getting-started-with-Oceananigans), [accessing and using GPUs](https://github.com/CliMA/Oceananigans.jl/wiki/Accessing-GPUs-and-using-Oceananigans-on-GPUs), and [productive workflows when using Oceananigans](https://github.com/CliMA/Oceananigans.jl/wiki/Productive-Oceananigans-workflows-and-Julia-environments).
* The `#oceananigans` channel on the [Julia Slack](https://julialang.org/slack/), which accesses "institutional knowledge" stored in the minds of the amazing Oceananigans community.
* [Issues](https://github.com/CliMA/Oceananigans.jl/issues) and [pull requests](https://github.com/CliMA/Oceananigans.jl/pulls) also contain lots of information about problems we've found, solutions we're trying to implement, and dreams we're dreaming to make tomorrow better 🌈.

## Citing and otherwise spreading the word

If you use Oceananigans for your research, teaching, or fun 🤩, everyone in our community will be grateful
if you credit Oceananigans by name.

The community has published a number of articles describing the development of Oceananigans,
including a recent [preprint submitted to the Journal of Advances in Modeling Earth Systems](https://arxiv.org/abs/2502.14148) that presents an overview of all the things that make Oceananigans unique:

> "High-level, high-resolution ocean modeling at all scales with Oceananigans"
>
> by Gregory L. Wagner, Simone Silvestri, Navid C. Constantinou, Ali Ramadhan, Jean-Michel Campin,
> Chris Hill, Tomas Chor, Jago Strong-Wright, Xin Kai Lee, Francis Poulin, Andre Souza, Keaton J. Burns,
> John Marshall, Raffaele Ferrari
>
> submitted to the Journal of Advances in Modeling Earth Systems, arXiv:2502.14148

<details><summary>bibtex</summary>
  <pre><code>@article{Oceananigans-overview-paper-2025,
  title = {{High-level, high-resolution ocean modeling at all scales with Oceananigans}},
  author = {G. L. Wagner and S. Silvestri and N. C. Constantinou and A. Ramadhan and J.-M. Campin and C. Hill and T. Chor and J. Strong-Wright and X. K. Lee and F. Poulin and A. Souza and K. J. Burns and J. Marshall and R. Ferrari},
  journal = {arXiv preprint},
  year = {2025},
  archivePrefix = {arXiv},
  eprint = {2502.14148},
  doi = {10.48550/arXiv.2502.14148},
  notes = {submitted to the Journal of Advances in Modeling Earth Systems},
}</code></pre>
</details>

Please cite this 👆 overview paper if you use Oceananigans in published work.

We've also published/submitted several model development papers. Please cite these below 👇 if you use
the features they describe! Also, if you have developed a new feature in Oceananigans and describe it in a paper, make sure to open a pull request to add it to this list:

* **Silvestri et al., ["A new WENO-Based momentum advection scheme for simulations of ocean mesoscale turbulence"](https://doi.org/10.1029/2023MS004130).**

  *This paper describes the development of `WENOVectorInvariant()` advection scheme, which can be used as the
  `momentum_advection` scheme for `HydrostaticFreeSurfaceModel`.*

* **Silvestri et al., ["A GPU-based ocean dynamic core for routine mesoscale-resolving climate simulations"](https://doi.org/10.1029/2024MS004465).**

  *This paper describes the optimization of the `HydrostaticFreeSurfaceModel` algorithm, including the implementation
  of a new `SplitExplicitFreeSurface` algorithm for `Distributed` architectures for multiple GPUs. As a result of this work,
  global simulations with O(10 km) grid spacing can be run on 16-20 nodes, achieving 10 simulated years per day (SYPD).*

* **Wagner et al., ["Formulation and calibration of CATKE, a one-equation parameterization for microscale ocean mixing"](https://doi.org/10.1029/2024MS004522).**

  *This paper describes the development of `CATKEVerticalDiffusivity()`, including how it was automatically calibrated to
  a suite of 35 large eddy simulations (also run with Oceananigans). It additionally features solutions from `TKEDissipationVerticalDiffusivity` (also known as "k-epsilon").*

* **Ramadhan et al., ["Oceananigans.jl: Fast and friendly geophysical fluid dynamics on GPUs"](https://doi.org/10.21105/joss.02018).**

  *This article in the Journal of Open Source Software describes an early version of Oceananigans' `NonhydrostaticModel`.*

We also maintain a [list of publications using Oceananigans.jl](https://clima.github.io/OceananigansDocumentation/stable/#Papers-and-preprints-using-Oceananigans).
If you have work using Oceananigans that you would like to have listed there, please open a pull request to add it or let us know!

## Contributing

If you're interested in contributing to the development of Oceananigans we want your help no matter how big or small a contribution you make!
Cause we're all in this together.

If you'd like to work on a new feature, or if you're new to open source and want to crowd-source neat projects that fit your interests, you should [start a discussion](https://github.com/CliMA/Oceananigans.jl/discussions/new?) right away.

For more information check out our [contributor's guide](https://clima.github.io/OceananigansDocumentation/stable/contributing/).

## Movies

### [Deep convection](https://www.youtube.com/watch?v=kpUrxnKKMjI)

[![Watch deep convection in action](https://raw.githubusercontent.com/ali-ramadhan/ali-ramadhan.Github.io/master/img/surface_temp_3d_00130_halfsize.png)](https://www.youtube.com/watch?v=kpUrxnKKMjI)

### [Free convection](https://www.youtube.com/watch?v=yq4op9h3xcU)

[![Watch free convection in action](https://raw.githubusercontent.com/ali-ramadhan/ali-ramadhan.Github.io/master/img/free_convection_0956.png)](https://www.youtube.com/watch?v=yq4op9h3xcU)

### [Winds blowing over the ocean](https://www.youtube.com/watch?v=IRncfbvuiy8)

[![Watch winds blowing over the ocean](https://raw.githubusercontent.com/ali-ramadhan/ali-ramadhan.Github.io/master/img/wind_stress_0400.png)](https://www.youtube.com/watch?v=IRncfbvuiy8)

### [Free convection with wind stress](https://www.youtube.com/watch?v=ob6OMQgPfI4)

[![Watch free convection with wind stress in action](https://raw.githubusercontent.com/ali-ramadhan/ali-ramadhan.Github.io/master/img/wind_stress_unstable_7500.png)](https://www.youtube.com/watch?v=ob6OMQgPfI4)

## Performance benchmarks

We perform some performance benchmarks (see the [performance benchmarks](https://clima.github.io/OceananigansDocumentation/stable/appendix/benchmarks/) section of the documentation) by initializing models of various sizes and measuring the wall clock time taken per model iteration (or time step).

To make full use of or fully saturate the computing power of a GPU such as an Nvidia Tesla V100 or
a Titan V, the model should have around ~10 million grid points or more.

Sometimes counter-intuitively running with `Float32` is slower than `Float64`. This is likely due
to type mismatches causing slowdowns as floats have to be converted between 32-bit and 64-bit, an
issue that needs to be addressed meticulously. Due to other bottlenecks such as memory accesses and
GPU register pressure, `Float32` models may not provide much of a speedup so the main benefit becomes
lower memory costs (by around a factor of 2).

![Performance benchmark plots](https://user-images.githubusercontent.com/20099589/89906791-d2c85b00-dbb9-11ea-969a-4b8db2c31680.png)
