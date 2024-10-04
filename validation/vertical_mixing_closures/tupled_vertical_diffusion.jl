using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity

grid = RectilinearGrid(size=128, z=(-128, 0), topology=(Flat, Flat, Bounded))

closure = (VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ=1e-4), 
           CATKEVerticalDiffusivity())

model = HydrostaticFreeSurfaceModel(; grid, closure,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer())

bᵢ(z) = 1e-5 * z
set!(model, b = bᵢ)
simulation = Simulation(model, Δt=1minute, stop_iteration=10)

run!(simulation)

