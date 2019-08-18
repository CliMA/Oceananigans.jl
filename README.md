# Oceananigans.jl

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://mit-license.org/)
[![Ask us anything](https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg)](https://github.com/climate-machine/Oceananigans.jl/issues/new)

| **Documentation**             | **Build Status** (CPU, GPU, Windows, Docker)                                                                                 | **Code coverage**                                                                   |
|:------------------------------|:-----------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
| [![docs][docs-img]][docs-url] | [![travis][travis-img]][travis-url] [![gitlab][gitlab-img]][gitlab-url] [![appveyor][appveyor-img]][appveyor-url] [![docker][docker-img]][docker-url]   | [![coveralls][coveralls-img]][coveralls-url] [![codecov][codecov-img]][codecov-url] |

[docs-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-url]: https://climate-machine.github.io/Oceananigans.jl/latest/

[travis-img]: https://travis-ci.com/climate-machine/Oceananigans.jl.svg?branch=master
[travis-url]: https://travis-ci.com/climate-machine/Oceananigans.jl

[gitlab-img]: https://gitlab.com/JuliaGPU/Oceananigans-jl/badges/master/pipeline.svg
[gitlab-url]: https://gitlab.com/JuliaGPU/Oceananigans-jl/commits/master

[appveyor-img]: https://ci.appveyor.com/api/projects/status/sc488kyni1wp93he?svg=true
[appveyor-url]: https://ci.appveyor.com/project/ali-ramadhan/oceananigans-jl

[docker-img]: https://img.shields.io/docker/cloud/build/aliramadhan/oceananigans.svg
[docker-url]: https://hub.docker.com/r/aliramadhan/oceananigans

[coveralls-img]: https://coveralls.io/repos/github/climate-machine/Oceananigans.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/climate-machine/Oceananigans.jl?branch=master

[codecov-img]: https://codecov.io/gh/climate-machine/Oceananigans.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/climate-machine/Oceananigans.jl

Oceananigans.jl is a fast and friendly incompressible fluid flow solver written in Julia that can be run in 1-3 dimensions on CPUs and GPUs. It is designed to solve the rotating Boussinesq equations used in non-hydrostatic ocean modeling but can be used to solve for any incompressible flow. 

Our goal is to develop a friendly and intuitive package allowing users to focus on the science. Thanks to high-level, zero-cost abstractions that the Julia programming language makes possible, the model can have the same look and feel no matter the dimension or grid of the underlying simulation, and the same code is shared between the CPU and GPU.

## Development team
* [Ali Ramadhan](http://aliramadhan.me/) ([@ali-ramadhan](https://github.com/ali-ramadhan))
* Chris Hill ([@christophernhill](https://github.com/christophernhill))
* Jean-Michel Campin ([@jm-c](https://github.com/jm-c))
* [John Marshall](http://oceans.mit.edu/JohnMarshall/) ([@johncmarshall54](https://github.com/johncmarshall54))
* [Greg Wagner](https://glwagner.github.io/) ([@glwagner](https://github.com/glwagner))
* Andre Souza ([@sandreza](https://github.com/sandreza))
* [James Schloss](http://leios.github.io/) ([@leios](https://github.com/leios))
* [Mukund Gupta](https://mukund-gupta.github.io/) ([@mukund-gupta](https://github.com/mukund-gupta))
* Zhen Wu ([@zhenwu0728](https://github.com/zhenwu0728))
* On the Julia side, big thanks to Valentin Churavy ([@vchuravy](https://github.com/vchuravy)), Tim Besard ([@maleadt](https://github.com/maleadt)) and Peter Ahrens ([@peterahrens](https://github.com/peterahrens))!

## Installation instructions
You can install the latest stable version of Oceananigans using the built-in package manager (accessed by pressing `]` in the Julia command prompt)
```julia
julia>]
(v1.1) pkg> add Oceananigans
```
**Note**: Oceananigans requires the latest version of Julia (1.1) to run correctly.

## Running your first model
Let's initialize a 3D model with 100×100×50 grid points on a 2×2×1 km domain and simulate it for 10 time steps using steps of 60 seconds each (for a total of 10 minutes of simulation time).
```julia
using Oceananigans
model = Model(N=(100, 100, 50), L=(2000, 2000, 1000))
time_step!(model; Δt=60, Nt=10)
```
You just simulated what might have been a 3D patch of ocean, it's that easy! It was a still lifeless ocean so nothing interesting happened but now you can add interesting dynamics and visualize the output.

### More interesting example
Let's add something to make the dynamics a bit more interesting. We can add a hot bubble in the middle of the domain and watch it rise to the surface. This example shows how to set an initial condition.
```julia
using Oceananigans

# We'll set up a 2D model with an xz-slice so there's only 1 grid point in y
# and use an artificially high viscosity ν and diffusivity κ.
model = Model(N=(256, 1, 256), L=(2000, 1, 2000), arch=CPU(), ν=4e-2, κ=4e-2)

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
