ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
ENV["JULIA_DEBUG"] = "Reactant, Reactant_jll"

using Reactant

# Required for Reactant MLIR compilation of sharded models (see GB-25)
Reactant.Compiler.WHILE_CONCAT[] = true
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: first_time_step!
using JLD2
using CUDA

function sharding_test_model(arch)
    grid = LatitudeLongitudeGrid(arch, size=(40, 40, 10),
                                 longitude=(0, 360),
                                 latitude=(-10, 10),
                                 z=(-1000, 0),
                                 halo=(5, 5, 5))

    model = HydrostaticFreeSurfaceModel(grid;
                free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                tracers = :c,
                tracer_advection = WENO(),
                momentum_advection = WENOVectorInvariant(order=3),
                coriolis = HydrostaticSphericalCoriolis())

    ηᵢ(λ, φ, z) = exp(- (φ - 90)^2 / 10^2) + exp(- φ^2 / 10^2)
    set!(model, c=ηᵢ, η=ηᵢ)

    Δt = 5minutes
    model.clock.last_Δt = Δt

    return model
end
