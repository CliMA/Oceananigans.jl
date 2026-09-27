# Estimate of the GPU kernel-parameter size of the tracer kernels for a 24-tracer model with a slow group (gate A0).
#
# CUDA does not convert arrays on a machine without a GPU, so this script adapts the kernel arguments with an
# adaptor that replaces every `Array` by a struct with the memory layout of `CUDA.CuDeviceArray`
# (pointer, maxsize, dims, length). The result is an estimate of `sizeof(typeof(cudaconvert(args)))`.
#
# Usage: julia --project validation/split_tracer_stepping/kernel_argument_size.jl

using Adapt
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Models.HydrostaticFreeSurfaceModels: TransportOnlyBiogeochemistry
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity

include(joinpath(@__DIR__, "..", "..", "test", "setup", "split_tracer_stepping_test_utils.jl"))

struct DeviceArrayLayout{T, N} <: AbstractArray{T, N}
    pointer :: Ptr{T}
    maxsize :: Int
    dims :: NTuple{N, Int}
    len :: Int
end

Base.size(a::DeviceArrayLayout) = a.dims

struct DeviceLayoutAdaptor end

Adapt.adapt_storage(::DeviceLayoutAdaptor, a::Array{T, N}) where {T, N} =
    DeviceArrayLayout{T, N}(Ptr{T}(0), sizeof(a), size(a), length(a))

kernel_argument_size(args) = sizeof(typeof(Adapt.adapt(DeviceLayoutAdaptor(), args)))

z = MutableVerticalDiscretization(collect(range(-4000, 0, length=51)))
underlying_grid = LatitudeLongitudeGrid(size = (36, 18, 50), halo = (7, 7, 4), longitude = (0, 360), latitude = (-80, 80), z = z)
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom((λ, φ) -> -4000 + 1000 * cosd(2λ)))

bgc_names = Tuple(Symbol(:B, n) for n in 1:18)
tracer_names = (:T, :S, :N, :P, :Z, :D, bgc_names...)
slow_names = tracer_names[3:end]

biogeochemistry = MinimalNPZD(grid; sinking_speed = 10 / day)
splitting = TracerTimeStepSplitting(tracers = slow_names, ratio = 8, biogeochemistry_substeps = 4)

model = HydrostaticFreeSurfaceModel(grid; tracers = tracer_names, biogeochemistry,
                                    buoyancy = SeawaterBuoyancy(),
                                    closure = CATKEVerticalDiffusivity(),
                                    timestepper = :SplitRungeKutta3,
                                    tracer_advection = WENO(order=5),
                                    momentum_advection = WENOVectorInvariant(),
                                    tracer_time_step_splitting = splitting)

s = model.tracer_time_step_splitting

function tendency_arguments(model, velocities, biogeochemistry, closure_fields, name, index)
    return (model.timestepper.Gⁿ[name], model.grid, Val(index), Val(name), model.advection[name], model.closure,
            immersed_boundary_condition(model.tracers[name]), model.buoyancy, biogeochemistry, velocities,
            model.free_surface, model.tracers, closure_fields, model.auxiliary_fields, model.clock, model.forcing[name])
end

fast_arguments = tendency_arguments(model, model.transport_velocities, model.biogeochemistry, model.closure_fields, :T, 1)
slow_arguments = tendency_arguments(model, s.velocities, TransportOnlyBiogeochemistry(model.biogeochemistry), s.closure_fields, :D, 6)
source_arguments = (s.slow_tracers, model.grid, model.biogeochemistry, s.clock, Oceananigans.fields(model),
                    s.previous_tracers, 1.0, map(Val, keys(s.slow_tracers)))

println("Estimated GPU kernel-parameter size, 24-tracer LatitudeLongitude ImmersedBoundaryGrid with z-star, CATKE and 22 slow tracers:")
println("  fast tracer tendency kernel (`compute_hydrostatic_free_surface_Gc!`): ", kernel_argument_size(fast_arguments), " bytes")
println("  slow tracer tendency kernel (same kernel, long-step arguments):     ", kernel_argument_size(slow_arguments), " bytes")
println("  pointwise biogeochemical source kernel (22 slow tracers):           ", kernel_argument_size(source_arguments), " bytes")
println("  of which the full `model.tracers` NamedTuple:                        ", kernel_argument_size(model.tracers), " bytes")
println("CUDA limit: 4096 bytes by default, 32764 bytes with CUDA ≥ 12.1 on Volta or newer.")

# Exercise the split step once, with SplitRungeKutta3 and z-star, through the new code path
set!(model, T = 10, S = 35, N = 1, P = 0.1, Z = 0.1, D = 0.1)
for step in 1:8
    time_step!(model, 10minutes)
end
println("After 8 steps (one long step): iteration = ", model.clock.iteration,
        ", accumulated steps = ", s.accumulated_steps, ", max |N| = ", maximum(abs, interior(model.tracers.N)))
