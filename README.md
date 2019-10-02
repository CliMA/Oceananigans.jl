<!-- Title -->
<h1 align="center">
  Oceananigans.jl
</h1>

<!-- description -->
<p align="center">
  <strong>🌊 A fast and friendly incompressible fluid flow solver in Julia that can be run in 1-3 dimensions on CPUs and GPUs. http://bit.ly/oceananigans</strong>
</p>

<!-- Information badge -->
<p align="center">
  <a href="https://www.repostatus.org/#active">
    <img alt="Project Status" src="https://www.repostatus.org/badges/latest/active.svg?style=flat-square" />
  </a>
  <a href="https://mit-license.org">
    <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square">
  </a>
  <a href="https://github.com/climate-machine/Oceananigans.jl/issues/new">
    <img alt="Ask us anything" src="https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg?style=flat-square">
  </a>
  <a href="https://climate-machine.github.io/Oceananigans.jl/latest">
    <img alt="Documents" src="https://img.shields.io/badge/docs-latest-blue.svg?style=flat-square">
  </a>
</p>

<!-- CI/CD badge -->
<p align="center">
  <a href="https://travis-ci.com/climate-machine/Oceananigans.jl">
    <img alt="Build Status for CPU" src="https://img.shields.io/travis/com/climate-machine/Oceananigans.jl/master?label=CPU&logo=travis&logoColor=white&style=flat-square">
  </a>
  <a href="https://gitlab.com/JuliaGPU/Oceananigans-jl/commits/master">
    <img alt="Build Status for GPU" src="https://img.shields.io/gitlab/pipeline/JuliaGPU/Oceananigans-jl/master?label=GPU&logo=gitlab&logoColor=white&style=flat-square">
  </a>
  <a href="https://ci.appveyor.com/project/ali-ramadhan/oceananigans-jl">
    <img alt="Build Status for Window" src="https://img.shields.io/appveyor/ci/ali-ramadhan/oceananigans-jl/master?label=window&logo=appveyor&logoColor=white&style=flat-square">
  </a>
  <a href="https://hub.docker.com/r/aliramadhan/oceananigans">
    <img alt="Build Status for Docker" src="https://img.shields.io/docker/cloud/build/aliramadhan/oceananigans?logo=docker&logoColor=white&style=flat-square">
  </a>
  <a href="https://coveralls.io/github/climate-machine/Oceananigans.jl?branch=master">
    <img alt="Coverage Status for Coveralls" src="https://img.shields.io/coveralls/github/climate-machine/Oceananigans.jl/master?style=flat-square">
  </a>
  <a href="https://codecov.io/gh/climate-machine/Oceananigans.jl">
    <img alt="Coverage Status for Codecov" src="https://img.shields.io/codecov/c/github/climate-machine/Oceananigans.jl/master?logo=codecov&logoColor=white&style=flat-square">
  </a>
</p>

Oceananigans.jl is a fast and friendly incompressible fluid flow solver written in Julia that can be run in 1-3 dimensions on CPUs and GPUs. It is designed to solve the rotating Boussinesq equations used in non-hydrostatic ocean modeling but can be used to solve for any incompressible flow. 

Our goal is to develop a friendly and intuitive package allowing users to focus on the science. Thanks to high-level, zero-cost abstractions that the Julia programming language makes possible, the model can have the same look and feel no matter the dimension or grid of the underlying simulation, and the same code is shared between the CPU and GPU.

# Contents

- [Installation instructions](#installation-instructions)
- [Running your first model](#running-your-first-model)
  - [More interesting example](#more-interesting-example)
- [Getting help](#getting-help)
- [Movies](#movies)
  - [Deep convection](#deep-convection)
  - [Free convection](#free-convection)
  - [Winds blowing over the ocean](#winds-blowing-over-the-ocean)
  - [Free convection with wind stress](#free-convection-with-wind-stress)
- [Performance benchmarks](#performance-benchmarks)
- [Development team](#development-team)

## Installation instructions
You can install the latest version of Oceananigans using the built-in package manager (accessed by pressing `]` in the Julia command prompt) to add the package and instantiate/build all depdendencies
```julia
julia>]
(v1.1) pkg> add Oceananigans
(v1.1) pkg> instantiate
```
We recommend installing Oceananigans with the built-in Julia package manager, because this installs a stable, tagged release. Oceananigans.jl can be updated to the latest tagged release from the package manager by typing
```julia
(v1.1) pkg> update Oceananigans
```
At this time, updating should be done with care, as Oceananigans is under rapid development and breaking changes to the user API occur often. But if anything does happen, please open an issue!

**Note**: Oceananigans requires at least Julia v1.1 to run correctly.

## Running your first model
Let's initialize a 3D model with 100×100×50 grid points on a 2×2×1 km domain and simulate it for 10 time steps using steps of 60 seconds each (for a total of 10 minutes of simulation time).
```julia
using Oceananigans
model = BasicModel(N=(100, 100, 50), L=(2000, 2000, 1000))
time_step!(model; Δt=60, Nt=10)
```
You just simulated what might have been a 3D patch of ocean, it's that easy! It was a still lifeless ocean so nothing interesting happened but now you can add interesting dynamics and visualize the output.

### More interesting example
Let's add something to make the dynamics a bit more interesting. We can add a hot bubble in the middle of the domain and watch it rise to the surface. This example shows how to set an initial condition.
```julia
using Oceananigans

# We'll set up a 2D model with an xz-slice so there's only 1 grid point in y
# and use an artificially high viscosity ν and diffusivity κ.
model = BasicModel(N=(256, 1, 256), L=(2000, 1, 2000), arch=CPU(), ν=4e-2, κ=4e-2)

# Set a temperature perturbation with a Gaussian profile located at the center
# of the vertical slice. This will create a buoyant thermal bubble that will
# rise with time.
Lx, Lz = model.grid.Lx, model.grid.Lz
x₀, z₀ = Lx/2, Lz/2
T₀(x, y, z) = 20 + 0.01 * exp(-100 * ((x - x₀)^2 + (z - z₀)^2) / (Lx^2 + Lz^2))
set!(model; T=T₀)

time_step!(model; Δt=10, Nt=5000)
```
By changing `arch=CPU()` to `arch=GPU()`, the example will run on an Nvidia GPU!

Check out [`rising_thermal_bubble_2d.jl`](https://github.com/climate-machine/Oceananigans.jl/blob/master/examples/rising_thermal_bubble_2d.jl) to see how you can plot a 2D movie with the output.

**Note**: You need to have Plots.jl and ffmpeg installed for the movie to be automatically created by Plots.jl.

GPU model output can be plotted on-the-fly and animated using [Makie.jl](https://github.com/JuliaPlots/Makie.jl)! This [NextJournal notebook](https://nextjournal.com/sdanisch/oceananigans) has an example. Thanks [@SimonDanisch](https://github.com/SimonDanisch)! Some Makie.jl isosurfaces from a rising spherical thermal bubble (the GPU example):
<p align="center">
  <img src="https://raw.githubusercontent.com/ali-ramadhan/ali-ramadhan.Github.io/master/img/Rising%20spherical%20thermal%20bubble%20Makie.png">
</p>

You can see some movies from GPU simulations below along with CPU and GPU [performance benchmarks](https://github.com/climate-machine/Oceananigans.jl#performance-benchmarks).

## Getting help
If you are interested in using Oceananigans.jl or are trying to figure out how to use it, please feel free to ask us questions and get in touch! Check out the [examples](https://github.com/climate-machine/Oceananigans.jl/tree/master/examples) and [open an issue](https://github.com/climate-machine/Oceananigans.jl/issues/new) if you have any questions, comments, suggestions, etc.

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
We've performed some preliminary performance benchmarks (see the [`benchmarks.jl`](https://github.com/climate-machine/Oceananigans.jl/blob/master/benchmark/benchmarks.jl) file) by initializing models of various sizes and measuring the wall clock time taken per model iteration (or time step). The CPU used was a single core of an Intel Xeon CPU E5-2680 v4 @ 2.40GHz while the GPU used was an Nvidia Tesla V100-SXM2-16GB. This isn't really a fair comparison as we haven't parallelized across all the CPU's cores so we will revisit these benchmarks once Oceananigans.jl can run on multiple CPUs and GPUs.
![Performance benchmark plots](https://raw.githubusercontent.com/climate-machine/Oceananigans.jl/master/benchmark/oceananigans_benchmarks.png)

## Development team
* [Ali Ramadhan](http://aliramadhan.me/) ([@ali-ramadhan](https://github.com/ali-ramadhan))
* [Greg Wagner](https://glwagner.github.io/) ([@glwagner](https://github.com/glwagner))
* Chris Hill ([@christophernhill](https://github.com/christophernhill))
* Jean-Michel Campin ([@jm-c](https://github.com/jm-c))
* [John Marshall](http://oceans.mit.edu/JohnMarshall/) ([@johncmarshall54](https://github.com/johncmarshall54))
* Andre Souza ([@sandreza](https://github.com/sandreza))
* On the Julia side, big thanks to Valentin Churavy ([@vchuravy](https://github.com/vchuravy)), Tim Besard ([@maleadt](https://github.com/maleadt)) and Peter Ahrens ([@peterahrens](https://github.com/peterahrens))!
