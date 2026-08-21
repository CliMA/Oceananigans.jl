using Oceananigans.BoundaryConditions: implicit_flux_coefficient, immersed_implicit_flux_coefficient,
                                       needs_implicit_solver
using Oceananigans.Grids: Face, Center
using Oceananigans.ImmersedBoundaries: immersed_inactive_node, ImmersedBoundaryCondition
using Oceananigans.Operators: Δz
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: architecture

#####
##### Column-integrated implicit boundary-flux coefficients
#####
##### The barotropic momentum equation is the vertical integral of the baroclinic one, and the
##### vertical integral of ∂τ/∂z telescopes to the boundary values alone. A vertical boundary flux
##### that is affine in the velocity, `J = Fₑ + λ u`, therefore contributes `λ` to the depth-integrated
##### equation exactly as it contributes to the boundary cell of the baroclinic one.
#####
##### If `λ` is integrated implicitly in the vertical solver but left out of the barotropic mode, the
##### depth-mean loses that damping entirely; if it is applied explicitly in both, the barotropic mode
##### carries an explicit drag whose stability number is `λ Δt / H`. Summing the coefficients per
##### column here lets each mode damp the same `λ` implicitly, once.
#####

"""
$(TYPEDSIGNATURES)

Sum, over the column `(i, j)`, of the implicit coefficients `λ` of every vertical boundary flux acting
on `ℓx, ℓy`: the domain top and bottom, and the bottom and top faces of the immersed boundary. Returns
zero for boundary conditions that carry no implicit part, so a column with none contributes nothing.

Each `λ` is signed the way it enters the implicit diagonal, so that the depth-integrated momentum is
damped by `1 + Δτ Λ / H` exactly as the boundary cell is damped by `1 + Δt λ / Δz`. The domain top
contributes `-J/Δz` to the tendency and enters with `+λ`; the domain bottom contributes `+J/Δz` and
enters with `-λ`; both immersed facets carry fluxes along the inward-facing normal, contribute `+J/Δz`,
and enter with `-λ`. A surface stress and a bottom drag therefore both land on `Λ > 0`.
"""
@inline function column_implicit_flux_coefficient(i, j, grid, ℓx, ℓy, ℓz, clock, fields, top_bc, bottom_bc,
                                                  immersed_bc::ImmersedBoundaryCondition)
    Nz = size(grid, 3)

    Λ = domain_implicit_flux_coefficient(i, j, grid, clock, fields, top_bc, bottom_bc)

    # The immersed boundary sits at an arbitrary k, so its contribution is found by walking the column.
    for k in 1:Nz
        active = !immersed_inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

        # `immersed_inactive_node` is false outside the domain, so a column wet to `k = 1` or `k = Nz`
        # sees no immersed face there and the domain boundary condition is counted once.
        on_bottom = active & immersed_inactive_node(i, j, k-1, grid, ℓx, ℓy, ℓz)
        on_top    = active & immersed_inactive_node(i, j, k+1, grid, ℓx, ℓy, ℓz)

        λᵇ = immersed_implicit_flux_coefficient(immersed_bc.bottom, i, j, k, grid, clock, fields)
        λᵗ = immersed_implicit_flux_coefficient(immersed_bc.top,    i, j, k, grid, clock, fields)

        Λ -= ifelse(on_bottom, λᵇ, zero(grid))
        Λ -= ifelse(on_top,    λᵗ, zero(grid))
    end

    return Λ
end

# A field whose immersed condition carries no implicit part contributes only its top and bottom faces.
@inline column_implicit_flux_coefficient(i, j, grid, ℓx, ℓy, ℓz, clock, fields, top_bc, bottom_bc, immersed_bc) =
    domain_implicit_flux_coefficient(i, j, grid, clock, fields, top_bc, bottom_bc)

@inline domain_implicit_flux_coefficient(i, j, grid, clock, fields, top_bc, bottom_bc) =
    implicit_flux_coefficient(top_bc,    i, j, grid, clock, fields) -
    implicit_flux_coefficient(bottom_bc, i, j, grid, clock, fields)

# A component whose boundary fluxes are wholly explicit carries no field to write into.
@inline set_column_coefficient!(::Nothing, i, j, λ) = nothing
@inline set_column_coefficient!(field, i, j, λ) = @inbounds field[i, j, 1] = λ

@kernel function _compute_column_implicit_coefficients!(λᵁ, λⱽ, grid, clock, fields, u_bcs, v_bcs)
    i, j = @index(Global, NTuple)

    λu = column_implicit_flux_coefficient(i, j, grid, Face(), Center(), Center(), clock, fields, u_bcs.top, u_bcs.bottom, u_bcs.immersed)
    λv = column_implicit_flux_coefficient(i, j, grid, Center(), Face(), Center(), clock, fields, v_bcs.top, v_bcs.bottom, v_bcs.immersed)

    set_column_coefficient!(λᵁ, i, j, λu)
    set_column_coefficient!(λⱽ, i, j, λv)
end

compute_column_implicit_coefficients!(λᵁ, λⱽ, grid, clock, model_fields, u, v, parameters) = 
    launch!(architecture(grid), grid, parameters,
            _compute_column_implicit_coefficients!, λᵁ, λⱽ, grid, clock, model_fields,
            u.boundary_conditions, v.boundary_conditions)

# Nothing to refresh when neither component carries an implicit part.
compute_column_implicit_coefficients!(::Nothing, ::Nothing, args...) = nothing

#####
##### Allocation
#####

@inline any_implicit_boundary_flux(bcs) = needs_implicit_solver(bcs.top) |
                                          needs_implicit_solver(bcs.bottom) |
                                          immersed_needs_implicit_solver(bcs.immersed)

@inline immersed_needs_implicit_solver(immersed_bc) = false

@inline immersed_needs_implicit_solver(immersed_bc::ImmersedBoundaryCondition) =
    needs_implicit_solver(immersed_bc.top) | needs_implicit_solver(immersed_bc.bottom)

function materialize_column_implicit_coefficients(grid, u, v)
    u_needed = any_implicit_boundary_flux(u.boundary_conditions)
    v_needed = any_implicit_boundary_flux(v.boundary_conditions)
    return (U = u_needed ? Field{Face, Center, Nothing}(grid) : nothing,
            V = v_needed ? Field{Center, Face, Nothing}(grid) : nothing)
end
