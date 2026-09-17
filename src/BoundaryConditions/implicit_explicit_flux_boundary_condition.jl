"""
    IMEXFluxTimeDiscretization(implicit_coefficient=nothing)

Time-discretization of a `Flux` boundary condition whose linear part is integrated implicitly.
At the boundary-adjacent cell the flux is split into `J(φᵦ) ≈ Fₑ + λ φᵦ`, where `φᵦ` is the
boundary-cell field value. `Fₑ` is integrated through the tendency and `λ φᵦ` by the vertical
tridiagonal solver, which removes the `Δz`-dependent time step restriction of an explicit
dissipative flux:

```julia
FluxBoundaryCondition(J;  time_discretization = IMEXFluxTimeDiscretization())   # J split automatically
FluxBoundaryCondition(Fₑ; time_discretization = IMEXFluxTimeDiscretization(λ))  # J = Fₑ + λ φᵦ
```

Without a coefficient the split is up to the condition: `BulkDragFunction` supplies its own, and any
other condition is split by the Patankar rule, see [`implicit_flux_coefficient`](@ref).
"""
struct IMEXFluxTimeDiscretization{C} <: AbstractTimeDiscretization
    implicit_coefficient :: C
end

IMEXFluxTimeDiscretization() = IMEXFluxTimeDiscretization(nothing)

Base.summary(::IMEXFluxTimeDiscretization) = "IMEXFluxTimeDiscretization"

Adapt.adapt_structure(to, td::IMEXFluxTimeDiscretization) =
    IMEXFluxTimeDiscretization(Adapt.adapt(to, td.implicit_coefficient))

"""
    struct IMEXFlux{E, C}

A flux condition with a user-supplied affine split `J(φᵦ) = explicit_flux + implicit_coefficient φᵦ`.
Built by [`FluxBoundaryCondition`](@ref) with an `IMEXFluxTimeDiscretization(λ)` that carries a coefficient.
"""
struct IMEXFlux{E, C}
    explicit_flux        :: E
    implicit_coefficient :: C
end

const IEFBC = BoundaryCondition{<:Flux{<:IMEXFluxTimeDiscretization}}

function materialize_flux_boundary_condition(flux, time_discretization::IMEXFluxTimeDiscretization;
                                             parameters, discrete_form, field_dependencies)

    condition = materialize_condition(flux, parameters, discrete_form, field_dependencies)
    λ = time_discretization.implicit_coefficient
    isnothing(λ) || (condition = IMEXFlux(condition, materialize_condition(λ, parameters, discrete_form, field_dependencies)))

    return BoundaryCondition(Flux(IMEXFluxTimeDiscretization()), condition)
end

"""
    IMEXFluxBoundaryCondition(flux; kwargs...)
    IMEXFluxBoundaryCondition(explicit_flux, implicit_coefficient; kwargs...)

Shorthand for a `Flux` boundary condition with an [`IMEXFluxTimeDiscretization`](@ref), whose linear part is
integrated implicitly. With one argument, `flux` is the total flux `J`, split at the boundary cell according
to its condition. With two, the boundary condition is the affine flux
`J(φᵦ) = explicit_flux + implicit_coefficient φᵦ`.
"""
IMEXFluxBoundaryCondition(J; kwargs...) =
    FluxBoundaryCondition(J; time_discretization = IMEXFluxTimeDiscretization(), kwargs...)

IMEXFluxBoundaryCondition(Fₑ, λ; kwargs...) =
    FluxBoundaryCondition(Fₑ; time_discretization = IMEXFluxTimeDiscretization(λ), kwargs...)

@inline getbc(condition::IMEXFlux, args...) = getbc(condition.explicit_flux, args...)

#####
##### Evaluating a condition on a vertical boundary: domain boundaries take the two horizontal indices,
##### immersed facets the three indices of the boundary-adjacent cell.
#####

@inline boundary_flux(condition, ::Union{Bottom, Top}, i, j, k, grid, args...) = getbc(condition, i, j, grid, args...)
@inline boundary_flux(condition, ::ImmersedFacet,      i, j, k, grid, args...) = getbc(condition, i, j, k, grid, args...)

# `Nothing` is spelled out for each boundary rather than for all of them at once, which would be
# ambiguous with the two methods above.
@inline boundary_flux(::Nothing, ::Union{Bottom, Top}, i, j, k, grid, args...) = zero(grid)
@inline boundary_flux(::Nothing, ::ImmersedFacet,      i, j, k, grid, args...) = zero(grid)

#####
##### The affine split of a flux condition
#####

# A flux through the top enters the tendency as `-J/Δz`, through the bottom or any immersed facet as `+J/Δz`.
# The implicit step damps `φᵦ` when `λ ≥ 0` on the top and `λ ≤ 0` elsewhere.
@inline dissipative_part(::Top, λ) = max(λ, zero(λ))
@inline dissipative_part(::Union{Bottom, ImmersedFacet}, λ) = min(λ, zero(λ))

# The Patankar rule: `λ = J / φᵦ`, keeping only the part that damps `φᵦ`
@inline function patankar_coefficient(boundary, J, φᵦ)
    λ = ifelse(φᵦ == 0, zero(J), J / φᵦ)
    return dissipative_part(boundary, λ)
end

"""
    implicit_flux_coefficient(condition, boundary, i, j, k, grid, ϕ, args...)

The linear coefficient `λ` of the split `J(φᵦ) ≈ Fₑ + λ φᵦ` of the flux `condition` at the boundary-adjacent
cell `(i, j, k)` of the field `ϕ`, where `boundary` is `Bottom()`, `Top()`, or `ImmersedFacet()` and `args`
are the arguments the condition is evaluated with (`clock, fields, ...`). The vertically implicit solver
embeds `λ` in the boundary-cell diagonal.

The fallback is the Patankar rule: `λ` is the dissipative part of `J / φᵦ`, the part whose sign makes the
implicit step damp `φᵦ`. A flux proportional to `φᵦ`, such as a drag, is thereby integrated implicitly in
full; what is left of a flux that does not vanish with `φᵦ`, such as a stress or a relaxation toward a
nonzero target, is integrated explicitly by [`explicit_flux`](@ref). Conditions that know their own split,
like [`IMEXFlux`](@ref) and `BulkDragFunction`, extend both functions and avoid the division.
"""
@inline function implicit_flux_coefficient(condition, boundary, i, j, k, grid, ϕ, args...)
    J  = boundary_flux(condition, boundary, i, j, k, grid, args...)
    φᵦ = @inbounds ϕ[i, j, k]
    return patankar_coefficient(boundary, J, φᵦ)
end

"""
    explicit_flux(condition, boundary, i, j, k, grid, ϕ, args...)

The explicit part `Fₑ` of the affine split `J(φᵦ) ≈ Fₑ + λ φᵦ` of the flux `condition`, integrated through
the tendency. See [`implicit_flux_coefficient`](@ref) for the arguments. The fallback is the Patankar
remainder `J - λ φᵦ`.
"""
@inline function explicit_flux(condition, boundary, i, j, k, grid, ϕ, args...)
    J  = boundary_flux(condition, boundary, i, j, k, grid, args...)
    φᵦ = @inbounds ϕ[i, j, k]
    λ  = patankar_coefficient(boundary, J, φᵦ)
    return J - λ * φᵦ
end

# A user-supplied split
@inline implicit_flux_coefficient(c::IMEXFlux, boundary, i, j, k, grid, ϕ, args...) = boundary_flux(c.implicit_coefficient, boundary, i, j, k, grid, args...)
@inline explicit_flux(c::IMEXFlux, boundary, i, j, k, grid, ϕ, args...) = boundary_flux(c.explicit_flux, boundary, i, j, k, grid, args...)

#####
##### Boundary conditions: only an implicit-explicit flux has a linear part; any other boundary condition
##### is integrated explicitly in full.
#####

@inline implicit_flux_coefficient(::Nothing,           boundary, i, j, k, grid, ϕ, args...) = zero(grid)
@inline implicit_flux_coefficient(::BoundaryCondition, boundary, i, j, k, grid, ϕ, args...) = zero(grid)
@inline implicit_flux_coefficient(bc::IEFBC,           boundary, i, j, k, grid, ϕ, args...) =
    implicit_flux_coefficient(bc.condition, boundary, i, j, k, grid, ϕ, args...)

@inline explicit_flux(::Nothing,             boundary, i, j, k, grid, ϕ, args...) = zero(grid)
@inline explicit_flux(bc::BoundaryCondition, boundary, i, j, k, grid, ϕ, args...) = boundary_flux(bc, boundary, i, j, k, grid, args...)
@inline explicit_flux(bc::IEFBC,             boundary, i, j, k, grid, ϕ, args...) =
    explicit_flux(bc.condition, boundary, i, j, k, grid, ϕ, args...)

needs_implicit_solver(bc) = false
needs_implicit_solver(bc::IEFBC) = true

"""
    total_boundary_flux(bc, boundary, i, j, k, grid, ϕ, args...)

The realized boundary flux `Fₑ + λ φᵦ` of `bc` for the field `ϕ`, evaluated with the boundary-cell value
`ϕ[i, j, k]` (`k = Nz` on the `Top()`, `k = 1` on the `Bottom()`). A derived boundary condition that needs
the actual flux, such as the friction velocity `u★` of a TKE closure, reconstructs it with this function.
Without an implicit part it is the flux itself.
"""
@inline total_boundary_flux(bc, boundary, i, j, k, grid, ϕ, args...) =
    explicit_flux(bc, boundary, i, j, k, grid, ϕ, args...)

# Only an implicit-explicit flux has a linear part, so only this method needs `ϕ` to be indexable:
# `ϕ` may also be a constant, as the salinity of a `TemperatureSeawaterBuoyancy` is.
@inline total_boundary_flux(bc::IEFBC, boundary, i, j, k, grid, ϕ, args...) =
    explicit_flux(bc, boundary, i, j, k, grid, ϕ, args...) +
    implicit_flux_coefficient(bc, boundary, i, j, k, grid, ϕ, args...) * @inbounds ϕ[i, j, k]

function validate_implicit_explicit_flux_locations(bcs)
    for side in (bcs.west, bcs.east, bcs.south, bcs.north)
        side isa IEFBC && error("IMEXFluxTimeDiscretization is supported only on top and bottom boundaries")
    end
    validate_immersed_implicit_explicit_flux(bcs.immersed)
    return nothing
end

validate_immersed_implicit_explicit_flux(immersed_bc) = nothing

validate_immersed_implicit_explicit_flux(immersed_bc::IEFBC) =
    error("An immersed IMEXFluxTimeDiscretization must be wrapped in an ImmersedBoundaryCondition")

Adapt.adapt_structure(to, c::IMEXFlux) = IMEXFlux(Adapt.adapt(to, c.explicit_flux), Adapt.adapt(to, c.implicit_coefficient))

Architectures.on_architecture(to, c::IMEXFlux) = IMEXFlux(on_architecture(to, c.explicit_flux), on_architecture(to, c.implicit_coefficient))
