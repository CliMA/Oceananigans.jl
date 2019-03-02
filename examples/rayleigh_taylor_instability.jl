using Oceananigans

Nx, Ny, Nz = 1024, 1, 512
Lx, Ly, Lz = 20000, 1, 10000
Nt, Δt = 500, 10

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=4e-2, κ=4e-2)

# Set the temperature of the upper half layer to T₀-ΔT and the temperature
# of the lower half layer to T₀-ΔT. Random noise of order ε is added to
# ensure the onset of turbulence.
ΔT = 0.02  # Temperature difference.
ε = 0.0001  # Small temperature perturbation.
@. model.tracers.T.data[:, :, 1:Int(Nz/2)]   = 283 - (ΔT/2) + ε*rand()
@. model.tracers.T.data[:, :, Int(Nz/2):end] = 283 + (ΔT/2) + ε*rand()

# Add a NetCDF output writer that saves NetCDF files to the current directory
# "." with a filename prefix of "rayleigh_taylor_" every 10 iterations.
nc_writer = NetCDFOutputWriter(dir=".", prefix="rayleigh_taylor_", frequency=10)
push!(model.output_writers, nc_writer)

time_step!(model, Nt, Δt)

make_vertical_slice_movie(model, nc_writer, "T", Nt, Δt, model.eos.T₀)
