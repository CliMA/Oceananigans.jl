using Statistics, Printf
using FFTW
using Plots
using Oceananigans

include("utils.jl")

Nx, Ny, Nz = 256, 1, 128
Lx, Ly, Lz = 2000, 1, 1000
Nt, Δt = 1000, 10*86400  # Time step of 10 days.

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), κ=1e-2)

# Set a +1 g/kg salinity anomaly in a square located at the center of the
# vertical slice with width 0.2Lx and height 0.2Lz.
i1, i2 = floor(Int, 4Nx/10), floor(Int, 6Nx/10)
k1, k2 = floor(Int, 4Nz/10), floor(Int, 6Nz/10)
@. model.tracers.S.data[i1:i2, 1, k1:k2] += 1

# Add a NetCDF output writer that saves NetCDF files to the current directory
# "." with a filename prefix of "diffusion_2d_" every 100 iterations.
nc_writer = NetCDFOutputWriter(dir=".", prefix="isotropic_diffusion_2d_", frequency=100)
push!(model.output_writers, nc_writer)

time_step!(model, Nt, Δt)

make_vertical_slice_movie(model, nc_writer, "S", Nt, Δt, model.eos.S₀)
