using Oceananigans

include("utils.jl")

Nx, Ny, Nz = 1, 1, 512
Lx, Ly, Lz = 1, 1, 5000
Nt, Δt = 10000, 60

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=4e-2, κ=4e-2)

@. model.tracers.T.data[:, :, 1:Int(Nz/2)]   -= 0.01
@. model.tracers.T.data[:, :, Int(Nz/2):end] += 0.01

# Add a NetCDF output writer that saves NetCDF files to the current directory
# "." with a filename prefix of "vertical_profile_" every 200 iterations.
nc_writer = NetCDFOutputWriter(dir=".", prefix="column_model_", frequency=200)
push!(model.output_writers, nc_writer)

time_step!(model, Nt, Δt)

make_vertical_profile_movie(model, nc_writer, "T", Nt, Δt)
