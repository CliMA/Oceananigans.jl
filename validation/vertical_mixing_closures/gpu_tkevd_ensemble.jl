pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, TKEBasedVerticalDiffusivity
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize

using CUDA
using BenchmarkTools

Ex, Ey = (400, 20)
sz = ColumnEnsembleSize(Nz=128, ensemble=(Ex, Ey))
halo = ColumnEnsembleSize(Nz=sz.Nz)

grid = RegularRectilinearGrid(size=sz, halo=halo, z=(-128, 0), topology=(Flat, Flat, Bounded))

closure = CuArray([TKEBasedVerticalDiffusivity() for i=1:Ex, j=1:Ey])
                                      
Qᵇ = CuArray([+1e-8 for i=1:Ex, j=1:Ey])
Qᵘ = CuArray([-1e-4 for i=1:Ex, j=1:Ey])
Qᵛ = CuArray([0.0   for i=1:Ex, j=1:Ey])

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵛ))
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

model = HydrostaticFreeSurfaceModel(architecture = GPU(),
                                    grid = grid,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=1e-4),
                                    boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                    closure = closure)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)


function one_step!(model)
    CUDA.@sync time_step!(model, 1e-9)
    return nothing
end

@info "Running one time-step..."
one_step!(model)

@info "Benchmarking..."
@btime one_step!(model)
