"""
    IMEXFluxTimeDiscretization(implicit_coefficient=nothing)

Time discretization of a `Flux` boundary condition split at the boundary cell as `J ≈ Fₑ + λ φᵦ`,
where `φᵦ` is the boundary-cell value. `Fₑ` is integrated explicitly and `λ φᵦ` by the vertical
tridiagonal solver. Without `implicit_coefficient`, the condition chooses the split; otherwise
the flux is `Fₑ` and `λ = implicit_coefficient`.
"""
struct IMEXFluxTimeDiscretization{C} <: AbstractTimeDiscretization
    implicit_coefficient :: C
end

IMEXFluxTimeDiscretization() = IMEXFluxTimeDiscretization(nothing)

Base.summary(::IMEXFluxTimeDiscretization) = "IMEXFluxTimeDiscretization"

Adapt.adapt_structure(to, td::IMEXFluxTimeDiscretization) =
    IMEXFluxTimeDiscretization(Adapt.adapt(to, td.implicit_coefficient))

# A user-supplied split J = explicit_flux + implicit_coefficient * φᵦ
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

Shorthand for `FluxBoundaryCondition` with an [`IMEXFluxTimeDiscretization`](@ref).
"""
IMEXFluxBoundaryCondition(J; kwargs...) =
    FluxBoundaryCondition(J; time_discretization = IMEXFluxTimeDiscretization(), kwargs...)

IMEXFluxBoundaryCondition(Fₑ, λ; kwargs...) =
    FluxBoundaryCondition(Fₑ; time_discretization = IMEXFluxTimeDiscretization(λ), kwargs...)

@inline getbc(condition::IMEXFlux, args...) = getbc(condition.explicit_flux, args...)

# Domain boundaries take two horizontal indices, immersed facets the three indices of the adjacent cell.
# `Nothing` is spelled out per boundary to avoid ambiguities.
@inline boundary_flux(condition, ::Union{Bottom, Top}, i, j, k, grid, args...) = getbc(condition, i, j, grid, args...)
@inline boundary_flux(condition, ::ImmersedFacet,      i, j, k, grid, args...) = getbc(condition, i, j, k, grid, args...)
@inline boundary_flux(::Nothing, ::Union{Bottom, Top}, i, j, k, grid, args...) = zero(grid)
@inline boundary_flux(::Nothing, ::ImmersedFacet,      i, j, k, grid, args...) = zero(grid)

# A top flux enters the tendency as -J/Δz, a bottom or immersed flux as +J/Δz,
# so the implicit step damps φᵦ for λ ≥ 0 on the top and λ ≤ 0 elsewhere.
@inline dissipative_part(::Top, λ) = max(λ, zero(λ))
@inline dissipative_part(::Union{Bottom, ImmersedFacet}, λ) = min(λ, zero(λ))

# Patankar split: λ is the dissipative part of J / φᵦ and Fₑ = J - λ φᵦ
@inline function patankar_split(condition, boundary, i, j, k, grid, ϕ, args...)
    J  = boundary_flux(condition, boundary, i, j, k, grid, args...)
    φᵦ = @inbounds ϕ[i, j, k]
    λ  = dissipative_part(boundary, ifelse(φᵦ == 0, zero(J), J / φᵦ))
    return J - λ * φᵦ, λ
end

# The split J ≈ Fₑ + λ φᵦ: `explicit_flux` returns Fₑ and `implicit_flux_coefficient` returns λ.
# Conditions that know their own split (`IMEXFlux`, `BulkDragFunction`) extend both.
@inline implicit_flux_coefficient(condition, boundary, i, j, k, grid, ϕ, args...) = patankar_split(condition, boundary, i, j, k, grid, ϕ, args...)[2]
@inline explicit_flux(condition, boundary, i, j, k, grid, ϕ, args...)             = patankar_split(condition, boundary, i, j, k, grid, ϕ, args...)[1]

@inline implicit_flux_coefficient(c::IMEXFlux, boundary, i, j, k, grid, ϕ, args...) = boundary_flux(c.implicit_coefficient, boundary, i, j, k, grid, args...)
@inline explicit_flux(c::IMEXFlux, boundary, i, j, k, grid, ϕ, args...)             = boundary_flux(c.explicit_flux,        boundary, i, j, k, grid, args...)

# Only an implicit-explicit flux has a linear part
@inline implicit_flux_coefficient(::Nothing,           boundary, i, j, k, grid, ϕ, args...) = zero(grid)
@inline implicit_flux_coefficient(::BoundaryCondition, boundary, i, j, k, grid, ϕ, args...) = zero(grid)
@inline implicit_flux_coefficient(bc::IEFBC,           boundary, i, j, k, grid, ϕ, args...) = implicit_flux_coefficient(bc.condition, boundary, i, j, k, grid, ϕ, args...)

@inline explicit_flux(::Nothing,             boundary, i, j, k, grid, ϕ, args...) = zero(grid)
@inline explicit_flux(bc::BoundaryCondition, boundary, i, j, k, grid, ϕ, args...) = boundary_flux(bc, boundary, i, j, k, grid, args...)
@inline explicit_flux(bc::IEFBC,             boundary, i, j, k, grid, ϕ, args...) = explicit_flux(bc.condition, boundary, i, j, k, grid, ϕ, args...)

needs_implicit_solver(bc) = false
needs_implicit_solver(bc::IEFBC) = true

# The realized flux Fₑ + λ φᵦ, for derived boundary conditions such as the TKE closure's friction velocity.
# Only the IEFBC method indexes ϕ, which may otherwise be a constant (e.g. salinity in TemperatureSeawaterBuoyancy).
@inline total_boundary_flux(bc, boundary, i, j, k, grid, ϕ, args...) = explicit_flux(bc, boundary, i, j, k, grid, ϕ, args...)

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
