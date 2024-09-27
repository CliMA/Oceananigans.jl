using Oceananigans
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.TimeSteppers: time_step!

grid = RectilinearGrid(size=64, z=(-256, 0), topology=(Flat, Flat, Bounded))
coriolis = FPlane(f=1e-4)
closure = CATKEVerticalDiffusivity()

boundary_conditions = (b = FieldBoundaryConditions(top = FluxBoundaryCondition(1e-8)),
                       u = FieldBoundaryConditions(top = FluxBoundaryCondition(-2e-4)))

model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis, boundary_conditions,
                                    tracers = (:b, :e), buoyancy = BuoyancyTracer())
                                    
bᵢ(z) = 1e-6 * z
set!(model, b=bᵢ, e=1e-6)

# Compile
time_step!(model, 600)

@time for n = 1:100
    time_step!(model, 600)
end


