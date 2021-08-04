#pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Oceananigans
using Oceananigans: short_show
using Oceananigans.TimeSteppers: time_step!
using BenchmarkTools

Nx = 64

xy_grid = RegularRectilinearGrid(size = (Nx, Nx, 1), halo = (3, 3, 3), extent = (2π, 2π, 2π), topology = (Periodic, Periodic, Bounded))
xz_grid = RegularRectilinearGrid(size = (Nx, 1, Nx), halo = (3, 3, 3), extent = (2π, 2π, 2π), topology = (Periodic, Periodic, Bounded))
yz_grid = RegularRectilinearGrid(size = (1, Nx, Nx), halo = (3, 3, 3), extent = (2π, 2π, 2π), topology = (Periodic, Periodic, Bounded))
                              
function nsteps!(model, n)
    for _ = 1:n
        time_step!(model, 1e-6)
    end
    return nothing
end

arch = CPU() #for arch in (CPU(), GPU())

model_kwargs = (architecture = arch,
                timestepper = :QuasiAdamsBashforth2,
                advection = nothing,
                closure = nothing,
                buoyancy = nothing,
                tracers = nothing)

grids = [xy_grid, xz_grid, yz_grid]
models = [NonhydrostaticModel(; grid = grid, model_kwargs...) for grid in grids]
xy_model, xz_model, yz_model = models
    
for model in models
    time_step!(model, 1e-6) # warmup
end

for model in models
    @info "Benchmarking model with $(short_show(model.grid))..."
    @btime nsteps!($model, 100)
end
