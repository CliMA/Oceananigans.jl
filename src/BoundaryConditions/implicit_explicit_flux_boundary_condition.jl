"""
    struct IMEXFluxTimeDiscretization{C} <: AbstractTimeDiscretization

An implicit-explicit (IMEX) time-discretization for an affine `Flux` boundary condition
`J(φ_b) = Fₑ + λ φ_b`. The explicit part `Fₑ` is integrated through the tendency like an
ordinary flux boundary condition, while the linear part `λ φ_b` — with `λ` the stored
`implicit_coefficient` — is integrated implicitly by the vertical tridiagonal solver.

    IMEXFluxTimeDiscretization(implicit_coefficient)

Build the discretization carrying the linear coefficient `λ`, then pass it to
[`FluxBoundaryCondition`](@ref) through the `time_discretization` keyword:

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

Condition stored by a `Flux` boundary condition with `IMEXFluxTimeDiscretization`,
representing a flux that is affine in the boundary-cell field value `φ_b`,

```math
J(φ_b) = Fₑ + λ φ_b ,
```

where `Fₑ` is the `explicit_flux` and `λ` is the `implicit_coefficient`. Built by
[`FluxBoundaryCondition`](@ref) when an `IMEXFluxTimeDiscretization` is supplied.
"""
struct IMEXFlux{E, C}
    explicit_flux        :: E
    implicit_coefficient :: C
end

const IEFBC = BoundaryCondition{<:Flux{<:IMEXFluxTimeDiscretization}}

# Affine flux `Fₑ + λ φ_b`: the explicit part enters the tendency, the linear part is integrated
# implicitly by the vertical solver. Selected whenever an `IMEXFluxTimeDiscretization` is supplied.
function materialize_flux_boundary_condition(explicit_flux, time_discretization::IMEXFluxTimeDiscretization;
                                             parameters, discrete_form, field_dependencies)

    Fₑ = materialize_condition(explicit_flux,                            parameters, discrete_form, field_dependencies)
    λ  = materialize_condition(time_discretization.implicit_coefficient, parameters, discrete_form, field_dependencies)

    return BoundaryCondition(Flux(IMEXFluxTimeDiscretization()), IMEXFlux(Fₑ, λ))
end

"""
    IMEXFluxBoundaryCondition(explicit_flux, implicit_coefficient; kwargs...)

Convenience constructor for an affine `Flux` boundary condition `J(φ_b) = explicit_flux + implicit_coefficient φ_b`.
Equivalent to passing an [`IMEXFluxTimeDiscretization`](@ref) to [`FluxBoundaryCondition`](@ref):

```julia
FluxBoundaryCondition(explicit_flux; time_discretization = IMEXFluxTimeDiscretization(implicit_coefficient), kwargs...)
```
"""
IMEXFluxBoundaryCondition(Fₑ, λ; kwargs...) =
    FluxBoundaryCondition(Fₑ; time_discretization = IMEXFluxTimeDiscretization(λ), kwargs...)

@inline getbc(condition::IMEXFlux, args...) = getbc(condition.explicit_flux, args...)

"""
    implicit_flux_coefficient(bc, i, j, grid, clock, fields)

Linear coefficient `λ` of an implicit-explicit `Flux` boundary condition (see
[`FluxBoundaryCondition`](@ref)), used by the vertically-implicit solver to embed `λ φ_b` in the
boundary-cell diagonal. Returns zero for any other boundary condition.
"""
@inline implicit_flux_coefficient(bc, i, j, grid, clock, fields) = zero(grid)
@inline implicit_flux_coefficient(bc::IEFBC, i, j, grid, clock, fields) = getbc(bc.condition.implicit_coefficient, i, j, grid, clock, fields)

needs_implicit_solver(bc) = false
needs_implicit_solver(bc::IEFBC) = true

"""
    total_boundary_flux(bc, i, j, k, grid, clock, fields, ϕ)

The *realized* boundary flux of `bc` for the field `ϕ`, evaluated using the boundary-cell value
`ϕ[i, j, k]` (`k = Nz` for a top boundary, `k = 1` for a bottom boundary). A derived boundary condition
that needs the actual flux, e.g. the friction velocity `u★` in a TKE closure, must reconstruct `Fₑ + λ φ_b`
with this function. For any other boundary condition it is just `getbc`.
"""
@inline total_boundary_flux(bc, i, j, k, grid, clock, fields, ϕ) = getbc(bc, i, j, grid, clock, fields)
@inline total_boundary_flux(bc::IEFBC, i, j, k, grid, clock, fields, ϕ) = @inbounds getbc(bc, i, j, grid, clock, fields) + implicit_flux_coefficient(bc, i, j, grid, clock, fields) * ϕ[i, j, k]

function validate_implicit_explicit_flux_locations(bcs)
    for side in (bcs.west, bcs.east, bcs.south, bcs.north, bcs.immersed)
        if side isa IEFBC
            error("A Flux boundary condition with `IMEXFluxTimeDiscretization` is only supported on " *
                  "vertical (top/bottom) boundaries: its implicit part is embedded in the vertical solver. " *
                  "Found one on a horizontal or immersed boundary.")
        end
    end
    return nothing
end

Adapt.adapt_structure(to, c::IMEXFlux) = IMEXFlux(Adapt.adapt(to, c.explicit_flux), Adapt.adapt(to, c.implicit_coefficient))

on_architecture(to, c::IMEXFlux) = IMEXFlux(on_architecture(to, c.explicit_flux), on_architecture(to, c.implicit_coefficient))
