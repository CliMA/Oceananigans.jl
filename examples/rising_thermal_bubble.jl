using Oceananigans

include("utils.jl")

Nx, Ny, Nz = 256, 1, 256
Lx, Ly, Lz = 2000, 1, 1000
Nt, Δt = 5000, 10

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=4e-2, κ=4e-2)

# Set a temperature perturbation with a Gaussian profile located at the center
# of the vertical slice. It roughly corresponds to a background temperature of
# T = 282.99 K and a bubble temperature of T = 283.01 K.
xC, zC = model.grid.xC, model.grid.zC
hot_bubble_perturbation = 0.01 * exp.(-100 * ((xC .- Lx/2).^2 .+ (zC .+ Lz/2)'.^2) / (Lx^2 + Lz^2))
model.tracers.T.data .= 282.99 .+ 2 .* reshape(hot_bubble_perturbation, (Nx, Ny, Nz))

# Add a NetCDF output writer that saves NetCDF files to the current directory
# "." with a filename prefix of "thermal_bubble_" every 10 iterations.
nc_writer = NetCDFOutputWriter(dir=".", prefix="thermal_bubble_", frequency=10)
push!(model.output_writers, nc_writer)

time_step!(model, Nt, Δt)

make_vertical_slice_movie(model, nc_writer, "T", Nt, Δt, model.eos.T₀)
