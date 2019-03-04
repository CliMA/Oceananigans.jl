using Oceananigans

include("utils.jl")

Nx, Ny, Nz = 128, 128, 1
Lx, Ly, Lz = 2000, 2000, 10
Nt, Δt = 25, 1

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=4e-2, κ=4e-2)

@. model.tracers.T.data[70:90, 10:30, 1] += 0.01
# @. model.velocities.u.data = 0.1
# @. model.velocities.v.data = 0.1

# Add a NetCDF output writer that saves NetCDF files to the current directory
# "." with a filename prefix of "vertical_profile_" every 200 iterations.
nc_writer = NetCDFOutputWriter(dir=".", prefix="barotropic_tracer_", frequency=1)
push!(model.output_writers, nc_writer)

time_step!(model, Nt, Δt)

make_horizontal_slice_movie(model, nc_writer, "T", Nt, Δt, model.eos.T₀)
