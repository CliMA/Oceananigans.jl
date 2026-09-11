"""
    struct IMEXFluxTimeDiscretization{C} <: AbstractTimeDiscretization

Time-discretization of an affine `Flux` boundary condition `J(φᵦ) = Fₑ + λ φᵦ`, where `φᵦ` is the
boundary-cell field value. `Fₑ` is integrated through the tendency and the linear part `λ φᵦ` — with
`λ` the stored `implicit_coefficient` — by the vertical tridiagonal solver.

    IMEXFluxTimeDiscretization(implicit_coefficient)

Build the discretization carrying `λ`, then pass it to [`FluxBoundaryCondition`](@ref):

```julia
FluxBoundaryCondition(Fₑ; time_discretization = IMEXFluxTimeDiscretization(λ))
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

The condition of a `Flux` boundary condition with `IMEXFluxTimeDiscretization`, holding the
`explicit_flux` `Fₑ` and the `implicit_coefficient` `λ` of the affine flux `J(φᵦ) = Fₑ + λ φᵦ`.
"""
struct IMEXFlux{E, C}
    explicit_flux        :: E
    implicit_coefficient :: C
end

const IEFBC = BoundaryCondition{<:Flux{<:IMEXFluxTimeDiscretization}}

function materialize_flux_boundary_condition(explicit_flux, time_discretization::IMEXFluxTimeDiscretization;
                                             parameters, discrete_form, field_dependencies)

    Fₑ = materialize_condition(explicit_flux,                            parameters, discrete_form, field_dependencies)
    λ  = materialize_condition(time_discretization.implicit_coefficient, parameters, discrete_form, field_dependencies)

    return BoundaryCondition(Flux(IMEXFluxTimeDiscretization()), IMEXFlux(Fₑ, λ))
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

"""
    implicit_flux_coefficient(bc, i, j, grid, clock, fields)

The linear coefficient `λ` of an implicit-explicit `Flux` boundary condition, which the vertically-implicit
solver embeds in the boundary-cell diagonal. Zero for any other boundary condition.
"""
@inline implicit_flux_coefficient(bc, i, j, grid, clock, fields) = zero(grid)
@inline implicit_flux_coefficient(bc::IEFBC, i, j, grid, clock, fields) = getbc(bc.condition.implicit_coefficient, i, j, grid, clock, fields)

@inline immersed_implicit_flux_coefficient(bc, i, j, k, grid, clock, fields) = zero(grid)
@inline immersed_implicit_flux_coefficient(bc::IEFBC, i, j, k, grid, clock, fields) = getbc(bc.condition.implicit_coefficient, i, j, k, grid, clock, fields)

needs_implicit_solver(bc) = false
needs_implicit_solver(bc::IEFBC) = true

"""
    total_boundary_flux(bc, i, j, k, grid, clock, fields, ϕ)

The realized boundary flux of `bc` for the field `ϕ`, evaluated with the boundary-cell value `ϕ[i, j, k]`
(`k = Nz` on a top boundary, `k = 1` on a bottom boundary). A derived boundary condition that needs the
actual flux, such as the friction velocity `u★` of a TKE closure, reconstructs `Fₑ + λ φᵦ` with this
function. For any other boundary condition it is `getbc`.
"""
@inline total_boundary_flux(bc, i, j, k, grid, clock, fields, ϕ) = getbc(bc, i, j, grid, clock, fields)
@inline total_boundary_flux(bc::IEFBC, i, j, k, grid, clock, fields, ϕ) = @inbounds getbc(bc, i, j, grid, clock, fields) + implicit_flux_coefficient(bc, i, j, grid, clock, fields) * ϕ[i, j, k]

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
