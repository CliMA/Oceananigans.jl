"""
    BuoyancyTracer <: AbstractBuoyancyModel{Nothing}

Type indicating that the tracer `b` represents buoyancy.
"""
struct BuoyancyTracer <: AbstractBuoyancyModel{Nothing} end

const BuoyancyTracerModel = Buoyancy{<:BuoyancyTracer}

required_tracers(::BuoyancyTracer) = (:b,)

@inline buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, ::BuoyancyTracer, C) = @inbounds C.b[i, j, k]

@inline ∂x_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂xᶠᶜᶜ(i, j, k, grid, C.b)
@inline ∂y_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂yᶜᶠᶜ(i, j, k, grid, C.b)
@inline ∂z_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂zᶜᶜᶠ(i, j, k, grid, C.b)

@inline    top_buoyancy_flux(i, j, grid, ::BuoyancyTracer, top_tracer_bcs, clock, fields) = getbc(top_tracer_bcs.b, i, j, grid, clock, fields)
@inline bottom_buoyancy_flux(i, j, grid, ::BuoyancyTracer, bottom_tracer_bcs, clock, fields) = getbc(bottom_tracer_bcs.b, i, j, grid, clock, fields)
