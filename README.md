<!-- Title -->
<h1 align="center">
  Oceananigans.jl
</h1>

<!-- description -->
<p align="center">
  <strong>🌊 A fast and friendly incompressible fluid flow solver in Julia that can be run in 1-3 dimensions on CPUs and GPUs. https://clima.github.io/OceananigansDocumentation/stable</strong>
</p>

<!-- Information badges -->
<p align="center">
  <a href="https://www.repostatus.org/#active">
    <img alt="Repo status" src="https://www.repostatus.org/badges/latest/active.svg?style=flat-square" />
  </a>
  <a href="https://mit-license.org">
    <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square">
  </a>
  <a href="https://github.com/clima/Oceananigans.jl/issues/new">
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

<!-- CI/CD badges -->
<p align="center">
  <a href="https://buildkite.com/clima/oceananigans">
    <img alt="Buildkite CPU+GPU build status" src="https://img.shields.io/buildkite/4d921fc17b95341ea5477fb62df0e6d9364b61b154e050a123/master?logo=buildkite&label=Buildkite%20CPU%2BGPU&style=flat-square">
  </a>
  <a href="https://travis-ci.com/clima/Oceananigans.jl">
    <img alt="CPU build status" src="https://img.shields.io/travis/com/clima/Oceananigans.jl/master?label=Travis&logo=travis&logoColor=white&style=flat-square">
  </a>
  <a href="https://hub.docker.com/r/aliramadhan/oceananigans">
    <img alt="Docker build status" src="https://img.shields.io/docker/cloud/build/aliramadhan/oceananigans?label=Docker&logo=docker&logoColor=white&style=flat-square">
  </a>
  <a href="https://codecov.io/gh/clima/Oceananigans.jl">
    <img alt="Codecov coverage" src="https://img.shields.io/codecov/c/github/clima/Oceananigans.jl/master?label=Codecov&logo=codecov&logoColor=white&style=flat-square">
  </a>
</p>

Oceananigans.jl is a fast and friendly incompressible fluid flow solver written in Julia that can be run in 1-3 dimensions on CPUs and GPUs. It is designed to solve the rotating Boussinesq equations used in non-hydrostatic ocean modeling but can be used to solve for any incompressible flow.

Our goal is to develop a friendly and intuitive package allowing users to focus on the science. Thanks to high-level, zero-cost abstractions that the Julia programming language makes possible, the model can have the same look and feel no matter the dimension or grid of the underlying simulation, and the same code is shared between the CPU and GPU.

## Contents

* [Installation instructions](#installation-instructions)
* [Running your first model](#running-your-first-model)
  * [More interesting example](#more-interesting-example)
* [Getting help](#getting-help)
* [Movies](#movies)
  * [Deep convection](#deep-convection)
  * [Free convection](#free-convection)
  * [Winds blowing over the ocean](#winds-blowing-over-the-ocean)
  * [Free convection with wind stress](#free-convection-with-wind-stress)
* [Performance benchmarks](#performance-benchmarks)
* [Development team](#development-team)

## Installation instructions

You can install the latest version of Oceananigans using the built-in package manager (accessed by pressing `]` in the Julia command prompt) to add the package and instantiate/build all depdendencies

```julia
julia>]
(v1.5) pkg> add Oceananigans
(v1.5) pkg> instantiate
```

We recommend installing Oceananigans with the built-in Julia package manager, because this installs a stable, tagged release. Oceananigans.jl can be updated to the latest tagged release from the package manager by typing

```julia
(v1.5) pkg> update Oceananigans
```

At this time, updating should be done with care, as Oceananigans is under rapid development and breaking changes to the user API occur often. But if anything does happen, please open an issue!

**Note**: Oceananigans requires at least Julia v1.5 to run. Installing Oceananigans with an older version of Julia will install an older version of Oceananigans (the latest version compatible with your version of Julia).

## Running your first model

Let's initialize a 3D horizontally periodic model with 100×100×50 grid points on a 2×2×1 km domain and simulate it for 1 hour using a constant time step of 60 seconds.

```julia
using Oceananigans
grid = RegularCartesianGrid(size=(100, 100, 50), extent=(2π, 2π, 1))
model = IncompressibleModel(grid=grid)
simulation = Simulation(model, Δt=60, stop_time=3600)
run!(simulation)
```

You just simulated what might have been a 3D patch of ocean, it's that easy! It was a still lifeless ocean so nothing interesting happened but now you can add interesting dynamics and visualize the output.

### More interesting example

Let's add something to make the dynamics a bit more interesting. We can add a hot bubble in the middle of the domain and watch it rise to the surface. This example also shows how to set an initial condition.

```julia
using Oceananigans

N = Nx = Ny = Nz = 128   # Number of grid points in each dimension.
L = Lx = Ly = Lz = 2000  # Length of each dimension.
topology = (Periodic, Periodic, Bounded)

model = IncompressibleModel(
    architecture = CPU(),
            grid = RegularCartesianGrid(topology=topology, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz)),
         closure = IsotropicDiffusivity(ν=4e-2, κ=4e-2)
)

# Set a temperature perturbation with a Gaussian profile located at the center.
# This will create a buoyant thermal bubble that will rise with time.
x₀, z₀ = Lx/2, Lz/2
T₀(x, y, z) = 20 + 0.01 * exp(-100 * ((x - x₀)^2 + (z - z₀)^2) / (Lx^2 + Lz^2))
set!(model, T=T₀)

simulation = Simulation(model, Δt=10, stop_iteration=5000)
run!(simulation)
```

By changing `architecture = CPU()` to `architecture = GPU()`, the example will run on an Nvidia GPU!

GPU model output can be plotted on-the-fly and animated using [Makie.jl](https://github.com/JuliaPlots/Makie.jl)! This [NextJournal notebook](https://nextjournal.com/sdanisch/oceananigans) has an example. Thanks [@SimonDanisch](https://github.com/SimonDanisch)! Some Makie.jl isosurfaces from a rising spherical thermal bubble (the GPU example):
<p align="center">
  <img src="https://raw.githubusercontent.com/ali-ramadhan/ali-ramadhan.Github.io/master/img/Rising%20spherical%20thermal%20bubble%20Makie.png">
</p>

You can see some movies from GPU simulations below along with CPU and GPU [performance benchmarks](https://github.com/clima/Oceananigans.jl#performance-benchmarks).

## Getting help

If you are interested in using Oceananigans.jl or are trying to figure out how to use it, please feel free to ask us questions and get in touch! Check out the [examples](https://github.com/clima/Oceananigans.jl/tree/master/examples) and [open an issue](https://github.com/clima/Oceananigans.jl/issues/new) if you have any questions, comments, suggestions, etc.

## Contributing

If you're interested in contributing to the development of Oceananigans we want your help no matter how big or small a contribution you make! It's always great to have new people look at the code with fresh eyes: you will see errors that other developers have missed.

Let us know by [opening an issue](https://github.com/clima/Oceananigans.jl/issues/new) if you'd like to work on a new feature or if you're new to open-source and want to find a cool little project or issue to work on that fits your interests! We're more than happy to help along the way.

For more information, check out our [contributor's guide](https://github.com/clima/Oceananigans.jl/blob/master/CONTRIBUTING.md).

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

We've performed some preliminary performance benchmarks (see the [performance benchmarks](https://clima.github.io/OceananigansDocumentation/stable/benchmarks/) section of the documentation) by initializing models of various sizes and measuring the wall clock time taken per model iteration (or time step).

This is not really a fair comparison as we haven't parallelized across all the CPU's cores so we will revisit these benchmarks once Oceananigans.jl can run on multiple CPUs and GPUs.

To make full use of or fully saturate the computing power of a GPU such as an Nvidia Tesla V100 or
a Titan V, the model should have around ~10 million grid points or more.

Sometimes counter-intuitively running with `Float32` is slower than `Float64`. This is likely due
to type mismatches causing slowdowns as floats have to be converted between 32-bit and 64-bit, an
issue that needs to be addressed meticulously. Due to other bottlenecks such as memory accesses and
GPU register pressure, `Float32` models may not provide much of a speedup so the main benefit becomes
lower memory costs (by around a factor of 2).

![Performance benchmark plots](https://user-images.githubusercontent.com/20099589/89906791-d2c85b00-dbb9-11ea-969a-4b8db2c31680.png)

## Development team

* [Ali Ramadhan](http://aliramadhan.me/) ([@ali-ramadhan](https://github.com/ali-ramadhan))
* [Greg Wagner](https://glwagner.github.io/) ([@glwagner](https://github.com/glwagner))
* Chris Hill ([@christophernhill](https://github.com/christophernhill))
* Jean-Michel Campin ([@jm-c](https://github.com/jm-c))
* [John Marshall](http://oceans.mit.edu/JohnMarshall/) ([@johncmarshall54](https://github.com/johncmarshall54))
* Andre Souza ([@sandreza](https://github.com/sandreza))
* On the Julia side, big thanks to Valentin Churavy ([@vchuravy](https://github.com/vchuravy)), Tim Besard ([@maleadt](https://github.com/maleadt)) and Peter Ahrens ([@peterahrens](https://github.com/peterahrens))!
