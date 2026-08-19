"""
    BuoyancyTracer <: AbstractBuoyancyFormulation{Nothing}

Type indicating that the tracer `b` represents buoyancy.
"""
struct BuoyancyTracer <: AbstractBuoyancyFormulation{Nothing} end

const BuoyancyTracerFormulation = BuoyancyForce{<:BuoyancyTracer}

required_tracers(::BuoyancyTracer) = (:b,)

@inline buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, ::BuoyancyTracer, C) = @inbounds C.b[i, j, k]

@inline ∂xᵣ_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂xᵣᶠᶜᶜ(i, j, k, grid, C.b)
@inline ∂yᵣ_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂yᵣᶜᶠᶜ(i, j, k, grid, C.b)
@inline  ∂x_b(i, j, k, grid, ::BuoyancyTracer, C) =  ∂xᶠᶜᶜ(i, j, k, grid, C.b)
@inline  ∂y_b(i, j, k, grid, ::BuoyancyTracer, C) =  ∂yᶜᶠᶜ(i, j, k, grid, C.b)
@inline  ∂z_b(i, j, k, grid, ::BuoyancyTracer, C) =  ∂zᶜᶜᶠ(i, j, k, grid, C.b)

@inline    top_buoyancy_flux(i, j, grid, ::BuoyancyTracer, top_tracer_bcs, clock, fields) = total_boundary_flux(top_tracer_bcs.b, i, j, size(grid, 3), grid, clock, fields, fields.b)
@inline bottom_buoyancy_flux(i, j, grid, ::BuoyancyTracer, bottom_tracer_bcs, clock, fields) = total_boundary_flux(bottom_tracer_bcs.b, i, j, 1, grid, clock, fields, fields.b)
