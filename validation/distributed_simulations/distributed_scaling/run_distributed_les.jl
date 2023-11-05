using MPI
MPI.Init()

using Oceananigans

rx = parse(Int, get(ENV, "RX", "1"))
ry = parse(Int, get(ENV, "RY", "1"))

ranks = (rx, ry, 1)

Nx = parse(Int, get(ENV, "NX", "256"))
Ny = parse(Int, get(ENV, "NY", "256"))
Nz = parse(Int, get(ENV, "NZ", "256"))

grid_size = (Nx, Ny, Nz)

include("distributed_nonhydrostatic.jl")
include("distributed_hydrostatic.jl")

@info "Running Nonhydrostatic model with ranks $ranks"
run_nonhydrostatic_simulation!(grid_size, ranks)
