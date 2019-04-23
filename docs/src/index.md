# Oceananigans.jl

Oceananigans is a fast non-hydrostatic ocean model written in Julia that can be run in 2 or 3 dimensions on CPUs and GPUs.

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

# Get location of the cell centers in x and z and reshape them to easily
# broadcast over them when calculating hot_bubble_perturbation.
xC, zC = model.grid.xC, model.grid.zC
xC, zC = reshape(xC, (Nx, 1, 1)), reshape(zC, (1, 1, Nz))

# Set a temperature perturbation with a Gaussian profile located at the center
# of the vertical slice. It roughly corresponds to a background temperature of
# T = 282.99 K and a bubble temperature of T = 283.01 K.
hot_bubble_perturbation = @. 0.01 * exp(-100 * ((xC - Lx/2)^2 + (zC + Lz/2)^2) / (Lx^2 + Lz^2))
model.tracers.T.data .= 282.99 .+ 2 .* reshape(hot_bubble_perturbation, (Nx, Ny, Nz))

# Add a NetCDF output writer that saves NetCDF files to the current directory
# "." with a filename prefix of "thermal_bubble_2d_" every 10 iterations.
nc_writer = NetCDFOutputWriter(dir=".", prefix="thermal_bubble_2d_", frequency=10)
push!(model.output_writers, nc_writer)

time_step!(model, Nt, Δt)
```

Check out [`rising_thermal_bubble_2d.jl`](https://github.com/climate-machine/Oceananigans.jl/blob/master/examples/rising_thermal_bubble_2d.jl) to see how you can plot a 2D movie with the output.

**Note**: You need to have Plots.jl and ffmpeg installed for the movie to be automatically created by Plots.jl.

### GPU example
If you have access to an Nvidia CUDA-enabled graphics processing unit (GPU) you can run ocean models on it! To make sure that the CUDA toolkit is properly installed and that Julia can see your GPU, run the `nvidia-smi` comand at the terminal and it should print out some information about your GPU.

To run on your GPU just pass `arch=GPU()` as a keyword argument when constructing the model, and we have to keep in mind that the model uses CuArrays now instead of regular Arrays, although we're working on abstracting this away.

Here is how you can run the rising thermal bubble example from above but in 3D on a GPU.
```julia
using Oceananigans

# We'll set up a 2D model with an xz-slice so there's only 1 grid point in y.
Nx, Ny, Nz = 128, 128, 128     # Number of grid points in each dimension.
Lx, Ly, Lz = 2000, 2000, 2000  # Domain size (meters).
Nt, Δt = 5000, 10              # Number of time steps, time step size (seconds).

# Set up the model and use an artificially high viscosity ν and diffusivity κ.
model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=GPU(), ν=4e-2, κ=4e-2)

# Get location of the cell centers in x, y, z and reshape them to easily
# broadcast over them when calculating hot_bubble_perturbation.
xC, yC, zC = model.grid.xC, model.grid.yC, model.grid.zC
xC, yC, zC = reshape(xC, (Nx, 1, 1)), reshape(yC, (Ny, 1, 1)), reshape(zC, (1, 1, Nz))

# Set a temperature perturbation with a Gaussian profile located at the center
# of the vertical slice. It roughly corresponds to a background temperature of
# T = 282.99 K and a bubble temperature of T = 283.01 K.
hot_bubble_perturbation = @. 0.01 * exp(-100 * ((xC - Lx/2)^2 + (yC - Ly/2)^2 + (zC + Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))
model.tracers.T.data .= 282.99 .+ 2 .* CuArray(hot_bubble_perturbation)

# Add a NetCDF output writer that saves NetCDF files to the current directory
# "." with a filename prefix of "thermal_bubble_3d_" every 10 iterations.
nc_writer = NetCDFOutputWriter(dir=".", prefix="thermal_bubble_3d_", frequency=25)
push!(model.output_writers, nc_writer)

time_step!(model, Nt, Δt)
```

**Warning**: Until issue [#64](https://github.com/climate-machine/Oceananigans.jl/issues/64) is resolved, you can only run GPU models with grid sizes where `Nx` and `Ny` are multiples of 16.

To see a more advanced example, see [`free_convection.jl`](https://github.com/climate-machine/Oceananigans.jl/blob/master/examples/free_convection.jl), which should be decently commented and comes with command line arguments to configure the simulation.

You can movie output from GPU simulations below along with CPU and GPU [performance benchmarks](https://github.com/climate-machine/Oceananigans.jl#performance-benchmarks).

## Getting help
If you are interested in using Oceananigans.jl or are trying to figure out how to use it, please feel free to ask us questions and get in touch! Check out the [examples](https://github.com/climate-machine/Oceananigans.jl/tree/master/examples) and [open an issue](https://github.com/climate-machine/Oceananigans.jl/issues/new) if you have any questions, comments, suggestions, etc.
```
