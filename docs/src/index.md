# Oceananigans.jl

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://mit-license.org/)
[![Latest documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://ali-ramadhan.github.io/Oceananigans.jl/latest)
[![Build Status](https://travis-ci.com/ali-ramadhan/Oceananigans.jl.svg?branch=master)](https://travis-ci.com/ali-ramadhan/Oceananigans.jl)
[![Pipeline status](https://gitlab.com/JuliaGPU/Oceananigans-jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/Oceananigans-jl/commits/master)
[![codecov](https://codecov.io/gh/ali-ramadhan/Oceananigans.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ali-ramadhan/Oceananigans.jl)
[![Ask us anything](https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg)](https://github.com/ali-ramadhan/Oceananigans.jl/issues)

Oceananigans is a fast and friendly non-hydrostatic _n_-dimensional ocean model that generically runs on CPU and GPU architectures. It is written 100% in Julia.

## Installation instructions


Oceananigans is still not an official Julia package. But you can install it using the built-in package manager (accessed by pressing `]` in the Julia command prompt)
```julia
julia>]
(v1.1) pkg> develop https://github.com/ali-ramadhan/Oceananigans.jl.git
(v1.1) pkg> test Oceananigans
```
**Note**: We recommend using Julia 1.1 with Oceananigans.

## Running your first model
Let's initialize a 3D ocean with $100\times100\times50$ grid points on a $2\times2\times1$ km domain and simulate it for 10 time steps using steps of 60 seconds each (for a total of 10 minutes of simulation time).
```julia
using Oceananigans
Nx, Ny, Nz = 100, 100, 50      # Number of grid points in each dimension.
Lx, Ly, Lz = 2000, 2000, 1000  # Domain size (meters).
Nt, Δt = 10, 60                # Number of time steps, time step size (seconds).

model = Model((Nx, Ny, Nz), (Lx, Ly, Lz))
time_step!(model, Nt, Δt)
```
You just simulated a 3D patch of ocean, it's that easy! It was a still lifeless ocean so nothing interesting happened but now you can add interesting dynamics and plot the output.

### CPU example
Let's add something to make the ocean dynamics a bit more interesting.

### GPU example
If you have access to an Nvidia CUDA-enabled graphics processing unit (GPU) you can run ocean models on it.
