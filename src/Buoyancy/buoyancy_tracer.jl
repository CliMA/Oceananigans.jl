using Oceananigans.Operators

"""
    BuoyancyTracer <: AbstractBuoyancy{Nothing}

Type indicating that the tracer `b` represents buoyancy.
"""
struct BuoyancyTracer <: AbstractBuoyancy{Nothing} end

required_tracers(::BuoyancyTracer) = (:b,)

@inline buoyancy_perturbation(i, j, k, grid, ::BuoyancyTracer, C) = @inbounds C.b[i, j, k]

@inline ∂x_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂xᶠᵃᵃ(i, j, k, grid, C.b)
@inline ∂y_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂yᵃᶠᵃ(i, j, k, grid, C.b)
@inline ∂z_b(i, j, k, grid, ::BuoyancyTracer, C) = ∂zᵃᵃᶠ(i, j, k, grid, C.b)
