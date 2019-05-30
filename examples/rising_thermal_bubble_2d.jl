using Oceananigans

# We'll want the make_vertical_slice_movie function from utils.jl to turn our
# rising thermal bubble output into a movie.
include("utils.jl")

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
data(model.tracers.T) .= 282.99 .+ 2 .* reshape(hot_bubble_perturbation, (Nx, Ny, Nz))

# Add a NetCDF output writer that saves NetCDF files to the current directory
# "." with a filename prefix of "thermal_bubble_2d_" every 10 iterations.
nc_writer = NetCDFOutputWriter(dir=".", prefix="thermal_bubble_2d_", frequency=10)
push!(model.output_writers, nc_writer)

time_step!(model, Nt, Δt)

make_vertical_slice_movie(model, nc_writer, "T", Nt, Δt, model.eos.T₀)
