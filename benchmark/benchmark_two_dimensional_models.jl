using Oceananigans
using Oceananigans: short_show
using Oceananigans.TimeSteppers: time_step!
using BenchmarkTools

xy_grid = RegularRectilinearGrid(size = (512, 512, 1), halo = (3, 3, 3), extent = (2π, 2π, 2π), topology = (Periodic, Periodic, Bounded))
xz_grid = RegularRectilinearGrid(size = (512, 1, 512), halo = (3, 3, 3), extent = (2π, 2π, 2π), topology = (Periodic, Periodic, Bounded))
yz_grid = RegularRectilinearGrid(size = (1, 512, 512), halo = (3, 3, 3), extent = (2π, 2π, 2π), topology = (Periodic, Periodic, Bounded))
                              
function ten_steps!(model)
    for _ = 1:10
        time_step!(model, 1e-6)
    end
    return nothing
end

arch = CPU() #for arch in (CPU(), GPU())

for grid in (xy_grid,
             xz_grid,
             yz_grid)

    model = NonhydrostaticModel(architecture = arch,
                                timestepper = :QuasiAdamsBashforth2,
                                grid = grid,
                                advection = nothing,
                                closure = nothing,
                                buoyancy = nothing,
                                tracers = nothing)
    
    time_step!(model, 1e-6) # warmup

    @info "Benchmarking model with $(short_show(grid))..."
    @btime ten_steps!($model)
end
