using Oceananigans
using Oceananigans.Grids: xspacing

struct SinkingParticles <: AbstractBiogeochemistry end
biogeochemical_drift_velocity(::SinkingParticles, ::Val{:A}) = (u = ConstantField(-1), v = ZeroField(), w = ZeroField())

grid = RectilinearGrid(CPU(); size = 40, halo = 5, x = (0, 1), topology = (Periodic, Flat, Flat))

u(args...) = -1

model = NonhydrostaticModel(; grid, 
                              tracers=:A, 
                              advection = UpwindBiased(),
                              background_fields = (; u))

A₀(x, y, z) = x > 0.4 && x < 0.6 ? 1.0 : 0.0

set!(model, u = 1, A = A₀)

Nx, _, _ = size(grid)
A_hist = zeros(1000, Nx)

A_truth = [A₀(x, 0, 0) for x in xnodes(grid, Center(), Center(), Center())]

model = NonhydrostaticModel(; grid, 
                              tracers=:A, 
                              advection = UpwindBiased(),
                              biogeochemistry = SinkingParticles())

set!(model, u = 1, A = A₀)

A_hist = zeros(1000, Nx)

model = NonhydrostaticModel(; grid, 
                              tracers=:A, 
                              advection = UpwindBiased(),
                              biogeochemistry = SinkingParticles(),
                              background_fields = (; u))

set!(model, u = 2, A = A₀)

A_hist = zeros(1000, Nx)
