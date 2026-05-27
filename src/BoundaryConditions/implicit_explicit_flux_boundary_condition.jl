"""
    struct ImplicitExplicitFlux{E, C}

Condition for a vertical `Flux` boundary condition whose value is affine in the
boundary-cell field value `φ_b`,

```math
J(φ_b) = Fₑ + λ φ_b ,
```

where `Fₑ` is the `explicit_flux` and `λ` is the `coefficient`. The explicit part `Fₑ` is integrated through the tendency
like an ordinary flux boundary condition, while the linear part `λ φ_b` is integrated implicitly by the vertical tridiagonal
solver. This removes the `Δz`-dependent CFL limit that an explicit flux imposes, and is unconditionally stable for dissipative
fluxes (`λ φ_b` a sink), such as drag and linear restoring.
"""
struct ImplicitExplicitFlux{E, C}
    explicit_flux :: E
    coefficient   :: C
end

"""
    ImplicitExplicitFluxBoundaryCondition(explicit_flux; coefficient, kwargs...)

Return a vertical `Flux` `BoundaryCondition` representing the affine flux `J = explicit_flux + coefficient * φ_b`,
with the linear `coefficient * φ_b` part integrated implicitly by the vertical solver and `explicit_flux` integrated
explicitly.

`explicit_flux` and `coefficient` follow the same conventions as any other function boundary condition; `kwargs`
(`parameters`, `discrete_form`, `field_dependencies`) are applied to both.

!!! warning "Vertical boundaries only"
    The implicit part is embedded in the vertical tridiagonal solver, so this boundary condition (for now)
    is only meaningful on `top`/`bottom` boundaries. Setting it on a horizontal (`west`/`east`/`south`/`north`)
    or immersed boundary would result in an error.
"""
function ImplicitExplicitFluxBoundaryCondition(explicit_flux; coefficient,
                                               parameters = nothing,
                                               discrete_form = false,
                                               field_dependencies = ())

    Fₑ = materialize_condition(explicit_flux, parameters, discrete_form, field_dependencies)
    λ  = materialize_condition(coefficient,   parameters, discrete_form, field_dependencies)

    return BoundaryCondition(Flux(), ImplicitExplicitFlux(Fₑ, λ))
end

const IEFBC = BoundaryCondition{<:Flux, <:ImplicitExplicitFlux}

# Only the explicit part enters the flux-divergence tendency; the linear part is added
# to the vertical-solver diagonal via `implicit_flux_coefficient`.
@inline getbc(condition::ImplicitExplicitFlux, args...) = getbc(condition.explicit_flux, args...)

"""
    implicit_flux_coefficient(bc, i, j, grid, clock, fields)

Linear coefficient `λ` of an [`ImplicitExplicitFluxBoundaryCondition`](@ref), used by the vertically-implicit
solver to embed `λ φ_b` in the boundary-cell diagonal. Returns zero for any other boundary condition.
"""
@inline implicit_flux_coefficient(bc, i, j, grid, clock, fields) = zero(grid)
@inline implicit_flux_coefficient(bc::IEFBC, i, j, grid, clock, fields) = getbc(bc.condition.coefficient, i, j, grid, clock, fields)

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
            error("ImplicitExplicitFluxBoundaryCondition is only supported on vertical (top/bottom) boundaries: " *
                  "its implicit part is embedded in the vertical solver. Found one on a horizontal or immersed boundary.")
        end
    end
    return nothing
end

Adapt.adapt_structure(to, c::ImplicitExplicitFlux) = ImplicitExplicitFlux(Adapt.adapt(to, c.explicit_flux), Adapt.adapt(to, c.coefficient))

on_architecture(to, c::ImplicitExplicitFlux) = ImplicitExplicitFlux(on_architecture(to, c.explicit_flux), on_architecture(to, c.coefficient))
