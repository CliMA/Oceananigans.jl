pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity

using Profile
using StatProfilerHTML
using BenchmarkTools

grid = RegularRectilinearGrid(size=64, z=(-64, 0), topology=(Flat, Flat, Bounded))

closure = CATKEVerticalDiffusivity()
                                      
Qᵇ = 1e-8
Qᵘ = - 1e-4
Qᵛ = 0.0

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵛ))
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=1e-4),
                                    boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                    closure = closure)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

function nsteps!(model, n)
    for _ = 1:n
        time_step!(model, 1e-9)
    end
    return nothing
end

@info "Running one time-step..."
nsteps!(model, 1) # warmup

@info "Benchmarking..."
@btime time_step!(model, 1e-9)

n = 100
@info "Profiling $n time-steps..."
@profile nsteps!(model, n)

statprofilehtml()
