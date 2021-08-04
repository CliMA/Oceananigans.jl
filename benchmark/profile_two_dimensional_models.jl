pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Oceananigans
using Oceananigans: short_show
using Oceananigans.TimeSteppers: time_step!
using Profile
using StatProfilerHTML

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

arch = CPU()

xz_model = NonhydrostaticModel(architecture = arch,
                               timestepper = :QuasiAdamsBashforth2,
                               grid = xz_grid,
                               advection = nothing,
                               closure = nothing,
                               buoyancy = nothing,
                               tracers = nothing)

xy_model = NonhydrostaticModel(architecture = arch,
                               timestepper = :QuasiAdamsBashforth2,
                               grid = xy_grid,
                               advection = nothing,
                               closure = nothing,
                               buoyancy = nothing,
                               tracers = nothing)


@info "Running one time-step..."
time_step!(xz_model, 1e-6) # warmup
time_step!(xy_model, 1e-6) # warmup

@info "Profiling..."
@profile nsteps!(xz_model, 1)
statprofilehtml()
run(`mv statprof xz_profile_results`)

@profile nsteps!(xy_model, 1)
statprofilehtml()
run(`mv statprof xy_profile_results`)
