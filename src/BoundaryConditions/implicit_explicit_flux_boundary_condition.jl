"""
    struct IMEXFluxTimeDiscretization{C} <: AbstractTimeDiscretization

Time-discretization of a `Flux` boundary condition whose linear part is integrated implicitly.
At the boundary-adjacent cell the flux is split into

```math
J(φᵦ) ≈ Fₑ + λ φᵦ
```

where `φᵦ` is the boundary-cell field value. The explicit part `Fₑ` is integrated through the tendency
and the linear part `λ φᵦ` by the vertical tridiagonal solver, which removes the `Δz`-dependent time step
restriction of an explicit dissipative flux. How a boundary condition is split depends on its condition,
see [`implicit_flux_coefficient`](@ref) and [`explicit_flux`](@ref):

* A condition that carries its own split, such as an [`IMEXFlux`](@ref) or a `BulkDragFunction`, supplies
  `Fₑ` and `λ` directly.
* Any other condition, a number, an array, a `Field`, or a function, is split by the Patankar rule: with `J`
  the flux at the current `φᵦ`, the dissipative part of `J / φᵦ` is the coefficient `λ` and the remainder
  `J - λ φᵦ` is integrated explicitly.

    IMEXFluxTimeDiscretization(implicit_coefficient=nothing)

Build the discretization and pass it to [`FluxBoundaryCondition`](@ref). Without a coefficient, the flux
passed to `FluxBoundaryCondition` is the whole flux `J`, split as described above. With a coefficient `λ`,
the flux passed to `FluxBoundaryCondition` is the explicit part `Fₑ`:

```julia
FluxBoundaryCondition(J;  time_discretization = IMEXFluxTimeDiscretization())   # J split by the Patankar rule
FluxBoundaryCondition(Fₑ; time_discretization = IMEXFluxTimeDiscretization(λ))  # J = Fₑ + λ φᵦ
```
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
    IMEXFluxBoundaryCondition(explicit_flux, implicit_coefficient; kwargs...)

Return a `Flux` boundary condition with the affine flux `J(φᵦ) = explicit_flux + implicit_coefficient φᵦ`.
Shorthand for

```julia
FluxBoundaryCondition(explicit_flux; time_discretization = IMEXFluxTimeDiscretization(implicit_coefficient), kwargs...)
```
"""
IMEXFluxBoundaryCondition(Fₑ, λ; kwargs...) =
    FluxBoundaryCondition(Fₑ; time_discretization = IMEXFluxTimeDiscretization(λ), kwargs...)

@inline getbc(condition::IMEXFlux, args...) = getbc(condition.explicit_flux, args...)

#####
##### Evaluating a condition on a vertical boundary: domain boundaries take the two horizontal indices,
##### immersed facets the three indices of the boundary-adjacent cell.
#####

@inline boundary_flux(condition, ::Union{Bottom, Top}, i, j, k, grid, args...) = getbc(condition, i, j, grid, args...)
@inline boundary_flux(condition, ::ImmersedFacet,      i, j, k, grid, args...) = getbc(condition, i, j, k, grid, args...)
@inline boundary_flux(::Nothing, boundary, i, j, k, grid, args...) = zero(grid)

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

The linear coefficient `λ` of the affine split `J(φᵦ) ≈ Fₑ + λ φᵦ` of the flux `condition` at the
boundary-adjacent cell `(i, j, k)` of the field `ϕ`, where `boundary` is `Bottom()`, `Top()`, or
`ImmersedFacet()` and `args` are the arguments the condition is evaluated with (`clock, fields, ...`).
The vertically implicit solver embeds `λ` in the boundary-cell diagonal.

The fallback is the Patankar rule: with `J` the flux at the current `φᵦ`, `λ` is the dissipative part of
`J / φᵦ`, the part with the sign that makes the implicit step damp `φᵦ`. For a flux proportional to `φᵦ`
(a drag or a sink toward zero) this recovers `λ = J / φᵦ` and the whole flux is integrated implicitly. For a
flux that does not vanish with `φᵦ`, such as a prescribed stress or a relaxation toward a nonzero target,
the Patankar coefficient is zero or a poor estimate of the slope, and the remainder is integrated explicitly.
Conditions that know their own split extend this function together with [`explicit_flux`](@ref), as
[`IMEXFlux`](@ref) and `BulkDragFunction` do, and avoid the division.
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
For a boundary condition without an implicit part it is the flux itself.
"""
@inline total_boundary_flux(bc, boundary, i, j, k, grid, ϕ, args...) =
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
