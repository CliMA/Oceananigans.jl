using Oceananigans.BoundaryConditions: implicit_flux_coefficient, immersed_implicit_flux_coefficient,
                                       needs_implicit_solver
using Oceananigans.ImmersedBoundaries: immersed_inactive_node, ImmersedBoundaryCondition
using Oceananigans.Operators: Δz

#####
##### Column-integrated implicit boundary-flux coefficients
#####
##### The barotropic momentum equation is the vertical integral of the baroclinic one, and the
##### vertical integral of ∂τ/∂z telescopes to the boundary values alone. A vertical boundary flux
##### affine in the boundary-cell velocity, `J = Fₑ + λ 𝓋ᵦ`, therefore contributes `λ 𝓋ᵦ` to the
##### depth-integrated equation exactly as it contributes to the boundary cell of the baroclinic one.
##### When `λ` is integrated implicitly by the vertical solver that contribution leaves the barotropic
##### forcing `Gᵁ`, and the depth mean loses its drag.
#####
##### Restoring it explicitly would leave the barotropic mode carrying a stability number `λ Δt / H`,
##### order ten on a shallow shelf. Splitting the boundary velocity into the barotropic velocity and
##### the deviation from it, `𝓋ᵦ = U/H + (𝓋ᵦ - U/H)`, integrates the first part implicitly and leaves
##### only the deviation — which the barotropic mode does not evolve — explicit:
#####
#####     λ 𝓋ᵦ  =  λ Uⁿ⁺¹ / H  +  λ (𝓋ᵦ - Uⁿ / H) ,
#####
##### giving the substep `U ← (Uᵗ + Δτ Ω) / (1 + Δτ Λ / H)`. This is unconditionally stable in the
##### barotropic mode and carries the true stress `λ 𝓋ᵦ`, so a sheared column is damped by the stress
##### it actually has rather than by its barotropic velocity.
#####
##### The deviation is the one the boundary cell will hold at the end of the step, not the one it holds
##### now: the vertical solver relaxes that cell by `1 + Λᶠ Δt / Δzᶠ` while the free surface substeps,
##### and a stiff face relaxes to the barotropic velocity within a single step. Taking `𝓋ᵦ` before that
##### relaxation damps a stiff face as though it still carried its old velocity, which is precisely the
##### stress it has just lost. With the relaxation included the scheme is exact in both limits: it
##### reduces to the explicit stress as `Λᶠ Δt / Δzᶠ → 0`, and to no barotropic damping at all as the
##### face stiffens and its velocity is clamped to the mean.
#####
##### `Λ` is signed the way it enters the implicit diagonal: the domain top contributes `-J/Δz` to the
##### tendency and enters with `+λ`; the domain bottom contributes `+J/Δz` and enters with `-λ`; both
##### immersed facets carry fluxes along the inward-facing normal, contribute `+J/Δz`, and enter with
##### `-λ`. A surface stress and a bottom drag both land on `Λ > 0`. `Ω` is a tendency, so it carries
##### the opposite sign.
#####

"""
$(TYPEDSIGNATURES)

The pair `(Λ, Ω)` for the column `(i, j)`: the coefficient the barotropic substep damps with, and the
explicit correction carrying the part of the boundary stress held by the deviation of the boundary
velocity `𝓋ᵦ` from the barotropic velocity `𝓋̄`. Both sum over every vertical boundary flux acting on
`ℓx, ℓy` — the domain top and bottom, and the bottom and top faces of the immersed boundary — and both
vanish for boundary conditions with no implicit part. A column whose velocity is depth-uniform has
`Ω = 0`, and the damping is then purely implicit.
"""
# One face's contribution: its coefficient `Λᶠ`, and the stress it will realize once the vertical
# solver has relaxed the boundary cell over the step.
@inline function boundary_face_contribution(i, j, k, grid, ℓx, ℓy, ℓz, Δt, Λᶠ, 𝓋)
    ρ = one(grid) + Λᶠ * Δt / Δz(i, j, k, grid, ℓx, ℓy, ℓz)
    return Λᶠ, @inbounds Λᶠ * 𝓋[i, j, k] / ρ
end

@inline function column_boundary_flux_coefficients(i, j, grid, ℓx, ℓy, ℓz, Δt, clock, fields, 𝓋, 𝓋̄,
                                                   top_bc, bottom_bc, immersed_bc::ImmersedBoundaryCondition)
    Nz = size(grid, 3)
    Λ, S = domain_boundary_flux_coefficients(i, j, grid, ℓx, ℓy, ℓz, Δt, clock, fields, 𝓋, top_bc, bottom_bc)

    # The immersed boundary sits at an arbitrary k, so its contribution is found by walking the column.
    for k in 1:Nz
        active = !immersed_inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

        # `immersed_inactive_node` is false outside the domain, so a column wet to `k = 1` or `k = Nz`
        # sees no immersed face there and the domain boundary condition is counted once.
        on_bottom = active & immersed_inactive_node(i, j, k-1, grid, ℓx, ℓy, ℓz)
        on_top    = active & immersed_inactive_node(i, j, k+1, grid, ℓx, ℓy, ℓz)

        λᵇ = immersed_implicit_flux_coefficient(immersed_bc.bottom, i, j, k, grid, clock, fields)
        λᵗ = immersed_implicit_flux_coefficient(immersed_bc.top,    i, j, k, grid, clock, fields)
        λ  = ifelse(on_bottom, λᵇ, zero(grid)) + ifelse(on_top, λᵗ, zero(grid))

        Λᶠ, Sᶠ = boundary_face_contribution(i, j, k, grid, ℓx, ℓy, ℓz, Δt, -λ, 𝓋)
        Λ += Λᶠ
        S += Sᶠ
    end

    return Λ, Λ * 𝓋̄ - S
end

# A field whose immersed condition carries no implicit part contributes only its top and bottom faces.
@inline function column_boundary_flux_coefficients(i, j, grid, ℓx, ℓy, ℓz, Δt, clock, fields, 𝓋, 𝓋̄,
                                                   top_bc, bottom_bc, immersed_bc)
    Λ, S = domain_boundary_flux_coefficients(i, j, grid, ℓx, ℓy, ℓz, Δt, clock, fields, 𝓋, top_bc, bottom_bc)
    return Λ, Λ * 𝓋̄ - S
end

# `Λ` sums the coefficients, `S` the stresses they realize; the caller turns them into `Ω`.
@inline function domain_boundary_flux_coefficients(i, j, grid, ℓx, ℓy, ℓz, Δt, clock, fields, 𝓋, top_bc, bottom_bc)
    Nz = size(grid, 3)

    λᵗ = implicit_flux_coefficient(top_bc,    i, j, grid, clock, fields)
    λᵇ = implicit_flux_coefficient(bottom_bc, i, j, grid, clock, fields)

    Λᵗ, Sᵗ = boundary_face_contribution(i, j, Nz, grid, ℓx, ℓy, ℓz, Δt,  λᵗ, 𝓋)
    Λᵇ, Sᵇ = boundary_face_contribution(i, j,  1, grid, ℓx, ℓy, ℓz, Δt, -λᵇ, 𝓋)

    return Λᵗ + Λᵇ, Sᵗ + Sᵇ
end

# A component whose boundary fluxes are wholly explicit carries no fields to write into.
@inline set_column_coefficients!(::Nothing, i, j, Λ, Ω) = nothing

@inline function set_column_coefficients!(coefficients, i, j, Λ, Ω)
    @inbounds coefficients.Λ[i, j, 1] = Λ
    @inbounds coefficients.Ω[i, j, 1] = Ω
    return nothing
end

@inline barotropic_velocity(i, j, transport, H) = ifelse(H > zero(H), @inbounds(transport[i, j, 1]) / H, zero(H))

@kernel function _compute_column_implicit_coefficients!(cᵁ, cⱽ, grid, filled_halos, Δt, clock, fields, u, v, U, V, η, u_bcs, v_bcs)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz + 1

    # The same topology-aware column depth the substep divides by, so that the implicit part and the
    # correction see one barotropic velocity, free surface included.
    ū = barotropic_velocity(i, j, U, x_column_depth(i, j, k_top, grid, filled_halos, η))
    v̄ = barotropic_velocity(i, j, V, y_column_depth(i, j, k_top, grid, filled_halos, η))

    Λu, Ωu = column_boundary_flux_coefficients(i, j, grid, Face(), Center(), Center(), Δt, clock, fields, u, ū, u_bcs.top, u_bcs.bottom, u_bcs.immersed)
    Λv, Ωv = column_boundary_flux_coefficients(i, j, grid, Center(), Face(), Center(), Δt, clock, fields, v, v̄, v_bcs.top, v_bcs.bottom, v_bcs.immersed)

    set_column_coefficients!(cᵁ, i, j, Λu, Ωu)
    set_column_coefficients!(cⱽ, i, j, Λv, Ωv)
end

compute_column_implicit_coefficients!(cᵁ, cⱽ, grid, filled_halos, Δt, clock, model_fields, u, v, U, V, η) =
    launch!(architecture(grid), grid, :xy,
            _compute_column_implicit_coefficients!, cᵁ, cⱽ, grid, filled_halos, Δt, clock, model_fields,
            u, v, U, V, η, u.boundary_conditions, v.boundary_conditions)

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

# `Λ` is a coefficient and folds as a scalar at a tripolar seam, `Ω` is a tendency carrying the sign of its velocity component.
column_coefficient_fields(ℓx, ℓy, grid, bcs) = (Λ = Field{ℓx, ℓy, Nothing}(grid),
                                                Ω = Field{ℓx, ℓy, Nothing}(grid, boundary_conditions = bcs))

function materialize_column_implicit_coefficients(grid, u, v, U_bcs, V_bcs)
    u_needed = any_implicit_boundary_flux(u.boundary_conditions)
    v_needed = any_implicit_boundary_flux(v.boundary_conditions)
    return (U = u_needed ? column_coefficient_fields(Face, Center, grid, U_bcs) : nothing,
            V = v_needed ? column_coefficient_fields(Center, Face, grid, V_bcs) : nothing)
end
