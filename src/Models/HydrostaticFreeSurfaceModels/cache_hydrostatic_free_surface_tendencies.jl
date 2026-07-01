using KernelAbstractions: @index, @kernel
using Oceananigans: prognostic_fields
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Utils: launch!

import Oceananigans.TimeSteppers: cache_previous_tendencies!

#####
##### Storing previous tendencies for the AB2 update
#####

""" Store source terms for `η`. """
@kernel function _cache_free_surface_tendency!(Gη⁻, grid, Gη⁰)
    i, j = @index(Global, NTuple)
    @inbounds Gη⁻[i, j, grid.Nz+1] = Gη⁰[i, j, grid.Nz+1]
end

cache_free_surface_tendency!(free_surface, model) = nothing

"""
$(TYPEDSIGNATURES)

Store the current free surface tendency `Gⁿ.η` into `G⁻.η` for AB2 time stepping.
Only applicable to `ExplicitFreeSurface` where `η` is a prognostic variable
advanced with the AB2 scheme.
"""
function cache_free_surface_tendency!(::ExplicitFreeSurface, model)
    launch!(model.architecture, model.grid, :xy,
            _cache_free_surface_tendency!,
            model.timestepper.G⁻.η,
            model.grid,
            model.timestepper.Gⁿ.η)
end

@kernel function _cache_field_tendencies!(G⁻, G⁰)
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻[i, j, k] = G⁰[i, j, k]
end

"""
$(TYPEDSIGNATURES)

Store the current tendencies `Gⁿ` into `G⁻` for all prognostic fields.

This function is called after advancing the model state but before computing new tendencies,
preserving the tendencies needed for the AB2 time-stepping scheme.

If CATKE or TD closures are active, their prognostic tracers (`e`, `ϵ`) are skipped.
For `ExplicitFreeSurface`, the free surface tendency is also cached.
"""
function cache_previous_tendencies!(model::HydrostaticFreeSurfaceModel)
    cache_prognostic_field_tendencies!(model, Val(keys(prognostic_fields(model))))
    cache_free_surface_tendency!(model.free_surface, model)
    return nothing
end

@inline cache_prognostic_field_tendencies!(model, ::Val{()}) = nothing

@inline function cache_prognostic_field_tendencies!(model, ::Val{names}) where names
    cache_prognostic_field_tendency!(model, Val(first(names)))
    cache_prognostic_field_tendencies!(model, Val(Base.tail(names)))
    return nothing
end

@inline function cache_prognostic_field_tendency!(model, ::Val{field_name}) where field_name
    closure = model.closure
    skip = field_name == :η ||
           (hasclosure(closure, FlavorOfCATKE) && field_name == :e) ||
           (hasclosure(closure, FlavorOfTD) && (field_name == :ϵ || field_name == :e))
    skip && return nothing
    launch!(model.architecture, model.grid, :xyz,
            _cache_field_tendencies!,
            model.timestepper.G⁻[field_name],
            model.timestepper.Gⁿ[field_name])
    return nothing
end

#####
##### Storing previous fields for the RK3 update
#####

# Tracers are multiplied by the vertical coordinate scaling factor
@kernel function _cache_tracer_fields!(Ψ⁻, grid, Ψⁿ)
    i, j, k = @index(Global, NTuple)
    @inbounds Ψ⁻[i, j, k] = Ψⁿ[i, j, k] * σⁿ(i, j, k, grid, Center(), Center(), Center())
end

"""
$(TYPEDSIGNATURES)

Cache the current prognostic fields at the beginning of a split Runge-Kutta time step.

The cached fields are stored in `model.timestepper.Ψ⁻` and serve as the base state `U⁰`
for all substeps within a single time step. Each substep computes `U = U⁰ + Δτ * G`.

For tracers, the cached quantity is `σ * c` (tracer times grid stretching factor) to
properly handle mutable vertical coordinates (z-star). Velocities and free surface
are cached directly without modification.
"""
function cache_current_fields!(model::HydrostaticFreeSurfaceModel)
    previous_fields = model.timestepper.Ψ⁻
    model_fields = prognostic_fields(model)
    cache_current_fields!(previous_fields, model_fields, model, Val(keys(model_fields)))
    return nothing
end

@inline cache_current_fields!(previous_fields, model_fields, model, ::Val{()}) = nothing

@inline function cache_current_fields!(previous_fields, model_fields, model, ::Val{names}) where names
    name = first(names)
    cache_current_field!(previous_fields[name], model_fields[name], Val(name), model)
    cache_current_fields!(previous_fields, model_fields, model, Val(Base.tail(names)))
    return nothing
end

@inline function cache_current_field!(Ψ⁻, Ψⁿ, ::Val{name}, model) where name
    grid = model.grid
    if name ∈ keys(model.tracers) # Tracers are stored with the grid scaling
        launch!(architecture(grid), grid, :xyz, _cache_tracer_fields!, Ψ⁻, grid, Ψⁿ)
    else # Velocities and free surface are stored without the grid scaling
        parent(Ψ⁻) .= parent(Ψⁿ)
    end
    return nothing
end
