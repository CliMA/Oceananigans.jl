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

### GPU example
If you have access to an Nvidia CUDA-enabled graphics processing unit (GPU) you can run ocean models on it! Just pass `arch=GPU()` as a keyword argument when constructing the model:
```julia
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=GPU())
```
