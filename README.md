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

[appveyor-img]: https://ci.appveyor.com/api/projects/status/sc488kyni1wp93he?svg=true
[appveyor-url]: https://ci.appveyor.com/project/ali-ramadhan/oceananigans-jl

[coveralls-img]: https://coveralls.io/repos/github/climate-machine/Oceananigans.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/climate-machine/Oceananigans.jl?branch=master

[codecov-img]: https://codecov.io/gh/climate-machine/Oceananigans.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/climate-machine/Oceananigans.jl

A fast non-hydrostatic ocean model in Julia that can be run in 2 or 3 dimensions on CPUs and GPUs. The plan is to develop it as a stand-alone [large eddy simulation](https://en.wikipedia.org/wiki/Large_eddy_simulation) (LES) model which can be used as a source of training data for statistical learning algorithms and/or embedded within a global ocean model as a super-parameterization of small-scale processes, as in [Campin et al., 2011](https://www.sciencedirect.com/science/article/pii/S1463500310001496?via%3Dihub).

Our goal is to develop friendly and intuitive code allowing users to focus on the science and not on fixing compiler errors. Thanks to high-level, zero-cost abstractions that the Julia programming language makes possible, the model can have the same look and feel no matter the dimension or grid of the underlying simulation, or whether running on CPUs or GPUs.

## Development team
* [Ali Ramadhan](http://aliramadhan.me/) ([@ali-ramadhan](https://github.com/ali-ramadhan))
* Chris Hill ([@christophernhill](https://github.com/christophernhill))
* Jean-Michel Campin ([@jm-c](https://github.com/jm-c))
* [John Marshall](http://oceans.mit.edu/JohnMarshall/) ([@johncmarshall54](https://github.com/johncmarshall54))
* [Greg Wagner](https://glwagner.github.io/) ([@glwagner](https://github.com/glwagner))
* [Mukund Gupta](https://mukund-gupta.github.io/) ([@mukund-gupta](https://github.com/mukund-gupta))
* Andre Souza ([@sandreza](https://github.com/sandreza))
* Zhen Wu ([@zhenwu0728](https://github.com/zhenwu0728))
* Also big thanks to Valentin Churavy ([@vchuravy](https://github.com/vchuravy)) and Peter Ahrens ([@peterahrens](https://github.com/peterahrens))!

## Installation instructions
You can install the latest stable version of Oceananigans using the built-in package manager (accessed by pressing `]` in the Julia command prompt)
```julia
julia>]
(v1.1) pkg> add Oceananigans
```
or the latest version via
```julia
julia>]
(v1.1) pkg> add https://github.com/climate-machine/Oceananigans.jl.git
```
**Note**: Oceananigans requires the latest version of Julia (1.1) to run correctly.

## Running your first model
Let's initialize a 3D ocean with 100×100×50 grid points on a 2×2×1 km domain and simulate it for 10 time steps using steps of 60 seconds each (for a total of 10 minutes of simulation time).
```julia
using Oceananigans
Nx, Ny, Nz = 100, 100, 50      # Number of grid points in each dimension.
Lx, Ly, Lz = 2000, 2000, 1000  # Domain size (meters).
Nt, Δt = 10, 60                # Number of time steps, time step size (seconds).

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz))
time_step!(model, Nt, Δt)
```
You just simulated a 3D patch of ocean, it's that easy! It was a still lifeless ocean so nothing interesting happened but now you can add interesting dynamics and plot the output.

### Interesting CPU example
Let's add something to make the ocean dynamics a bit more interesting. We can add a hot bubble in the middle of the ocean and watch it rise to the surface. This example also shows how to set an initial condition and write regular output to NetCDF.
```julia
using Oceananigans

# We'll set up a 2D model with an xz-slice so there's only 1 grid point in y.
Nx, Ny, Nz = 256, 1, 256    # Number of grid points in each dimension.
Lx, Ly, Lz = 2000, 1, 2000  # Domain size (meters).
Nt, Δt = 5000, 10           # Number of time steps, time step size (seconds).

# Set up the model and use an artificially high viscosity ν and diffusivity κ.
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=4e-2, κ=4e-2)

# Set a temperature perturbation with a Gaussian profile located at the center
# of the vertical slice. It roughly corresponds to a background temperature of
# T = 282.99 K and a bubble temperature of T = 283.01 K.
xC, zC = model.grid.xC, model.grid.zC  # Location of the cell centers in x and z.
hot_bubble_perturbation = @. 0.01 * exp(-100 * ((xC - Lx/2)^2 + (zC + Lz/2)'^2) / (Lx^2 + Lz^2))
model.tracers.T.data .= 282.99 .+ 2 .* reshape(hot_bubble_perturbation, (Nx, Ny, Nz))

# Add a NetCDF output writer that saves NetCDF files to the current directory
# "." with a filename prefix of "thermal_bubble_" every 10 iterations.
nc_writer = NetCDFOutputWriter(dir=".", prefix="thermal_bubble_", frequency=10)
push!(model.output_writers, nc_writer)

time_step!(model, Nt, Δt)
```

Check out [`rising_thermal_bubble.jl`](https://github.com/climate-machine/Oceananigans.jl/blob/master/examples/rising_thermal_bubble.jl) to see how you can plot a 2D movie with the output.

**Note**: You need to have Plots.jl and ffmpeg installed for the movie to be automatically created by Plots.jl.

### GPU example
If you have access to an Nvidia CUDA-enabled graphics processing unit (GPU) you can run ocean models on it! To make sure that the CUDA toolkit is properly installed and that Julia can see your GPU, run the `nvidia-smi` comand at the terminal and it should print out some information about your GPU.

Just pass `arch=GPU()` as a keyword argument when constructing the model:
```julia
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=GPU())
```

To see a more advanced example, see free_convection.jl, which should be decently commented and comes with command line arguments to configure the simulation.

**Warning**: Until issue # is resolved, you can only run GPU models with grid sizes where `Nx` and `Ny` are multiples of 16.

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
