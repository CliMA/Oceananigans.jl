# Oceananigans.jl

Oceananigans is a fast non-hydrostatic ocean model written in Julia that can be run in 2 or 3 dimensions on CPUs and GPUs.

## Installation instructions


Oceananigans is still not an official Julia package. But you can install it using the built-in package manager (accessed by pressing `]` in the Julia command prompt)
```julia
julia>]
(v1.1) pkg> add https://github.com/climate-machine/Oceananigans.jl.git
```
**Note**: We recommend using Julia 1.1 with Oceananigans.

## Running your first model
Let's initialize a 3D ocean with $100\times100\times50$ grid points on a $2\times2\times1$ km domain and simulate it for 10 time steps using steps of 60 seconds each (for a total of 10 minutes of simulation time).
```julia
using Oceananigans
Nx, Ny, Nz = 100, 100, 50      # Number of grid points in each dimension.
Lx, Ly, Lz = 2000, 2000, 1000  # Domain size (meters).
Nt, Δt = 10, 60                # Number of time steps, time step size (seconds).

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz))
time_step!(model, Nt, Δt)
```
You just simulated a 3D patch of ocean, it's that easy! It was a still lifeless ocean so nothing interesting happened but now you can add interesting dynamics and plot the output.

### CPU example
Let's add something to make the ocean dynamics a bit more interesting.

### GPU example
If you have access to an Nvidia CUDA-enabled graphics processing unit (GPU) you can run ocean models on it.
