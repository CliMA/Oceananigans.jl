"""
    BuoyancyTracer <: AbstractBuoyancy{Nothing}

Type indicating that the tracer `b` represents buoyancy.
"""
struct BuoyancyTracer <: AbstractBuoyancy{Nothing} end

required_tracers(::BuoyancyTracer) = (:b,)

@inline buoyancy_perturbation(i, j, k, grid, ::BuoyancyTracer, C) = @inbounds C.b[i, j, k]

@inline x_dot_g_b(i, j, k, grid, buoyancy::BuoyancyModel{<:BuoyancyTracer}, C) = @inbounds buoyancy.gravity_unit_vector[1] * C.b[i, j, k]
@inline y_dot_g_b(i, j, k, grid, buoyancy::BuoyancyModel{<:BuoyancyTracer}, C) = @inbounds buoyancy.gravity_unit_vector[2] * C.b[i, j, k]
@inline z_dot_g_b(i, j, k, grid, buoyancy::BuoyancyModel{<:BuoyancyTracer}, C) = @inbounds buoyancy.gravity_unit_vector[3] * C.b[i, j, k]

@inline ∂x_b(i, j, k, grid, ::BuoyancyModel{<:BuoyancyTracer}, C) = ∂xᶠᵃᵃ(i, j, k, grid, C.b)
@inline ∂y_b(i, j, k, grid, ::BuoyancyModel{<:BuoyancyTracer}, C) = ∂yᵃᶠᵃ(i, j, k, grid, C.b)
@inline ∂z_b(i, j, k, grid, ::BuoyancyModel{<:BuoyancyTracer}, C) = ∂zᵃᵃᶠ(i, j, k, grid, C.b)
