"""
    BuoyancyTracer <: AbstractBuoyancyModel{Nothing}

Type indicating that the tracer `b` represents buoyancy.
"""
struct BuoyancyTracer <: AbstractBuoyancyModel{Nothing} end

const BuoyancyTracerModel = Buoyancy{<:BuoyancyTracer}

required_tracers(::BuoyancyTracer) = (:b,)

@inline buoyancy_perturbation(i, j, k, grid, ::BuoyancyTracer, C) = @inbounds C.b[i, j, k]

@inline x_dot_g_b(i, j, k, grid, buoyancy_model::BuoyancyTracerModel, C) = @inbounds ĝ_x(buoyancy_model) * C.b[i, j, k]
@inline y_dot_g_b(i, j, k, grid, buoyancy_model::BuoyancyTracerModel, C) = @inbounds ĝ_y(buoyancy_model) * C.b[i, j, k]
@inline z_dot_g_b(i, j, k, grid, buoyancy_model::BuoyancyTracerModel, C) = @inbounds ĝ_z(buoyancy_model) * C.b[i, j, k]

@inline ∂x_b(i, j, k, grid, ::BuoyancyTracerModel, C) = ∂xᶠᵃᵃ(i, j, k, grid, C.b)
@inline ∂y_b(i, j, k, grid, ::BuoyancyTracerModel, C) = ∂yᵃᶠᵃ(i, j, k, grid, C.b)
@inline ∂z_b(i, j, k, grid, ::BuoyancyTracerModel, C) = ∂zᵃᵃᶠ(i, j, k, grid, C.b)
