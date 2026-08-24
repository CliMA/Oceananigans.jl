using Oceananigans.BoundaryConditions: implicit_flux_coefficient, immersed_implicit_flux_coefficient,
                                       needs_implicit_solver
using Oceananigans.ImmersedBoundaries: immersed_inactive_node, ImmersedBoundaryCondition

#####
##### Column-integrated boundary stress
#####
##### The vertical integral of ∂τ/∂z telescopes to the boundary values, so a flux affine in the boundary-cell
##### velocity, `J = Fₑ + λ 𝓋ᵦ`, contributes `λ 𝓋ᵦ` to the depth-integrated momentum. When `λ` is integrated
##### implicitly by the vertical solver that contribution leaves `Gᵁ`, and `Ω` carries it instead: it enters the
##### substep as a `U`-independent increment, exactly as `Gᵁ` does. The correction removes `Δt Ω` again once the
##### solver has applied the flux for real, so `Ω` sets what the free surface sees without being counted twice
##### — see `barotropic_split_explicit_corrector!`.
#####
##### `Ω` is a tendency: the domain top enters with `+λ`, the domain bottom and both immersed facets with `-λ`,
##### so a surface stress and a bottom drag both decelerate the column.
#####

@inline boundary_face_stress(i, j, k, λ, 𝓋) = @inbounds λ * 𝓋[i, j, k]

@inline function column_boundary_stress(i, j, grid, ℓx, ℓy, ℓz, clock, fields, 𝓋,
                                        top_bc, bottom_bc, immersed_bc::ImmersedBoundaryCondition)
    Ω = domain_boundary_stress(i, j, grid, ℓx, ℓy, ℓz, clock, fields, 𝓋, top_bc, bottom_bc)

    # The immersed boundary sits at an arbitrary k, so its contribution is found by walking the column.
    for k in 1:size(grid, 3)
        active = !immersed_inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

        # `immersed_inactive_node` is false outside the domain, so a column wet to `k = 1` or `k = Nz` sees no
        # immersed face there and the domain boundary condition is counted once.
        on_bottom = active & immersed_inactive_node(i, j, k-1, grid, ℓx, ℓy, ℓz)
        on_top    = active & immersed_inactive_node(i, j, k+1, grid, ℓx, ℓy, ℓz)

        λᵇ = immersed_implicit_flux_coefficient(immersed_bc.bottom, i, j, k, grid, clock, fields)
        λᵗ = immersed_implicit_flux_coefficient(immersed_bc.top,    i, j, k, grid, clock, fields)

        Ω += boundary_face_stress(i, j, k, ifelse(on_bottom, λᵇ, zero(grid)) + ifelse(on_top, λᵗ, zero(grid)), 𝓋)
    end

    return Ω
end

# A field whose immersed condition carries no implicit part contributes only its top and bottom faces.
@inline column_boundary_stress(i, j, grid, ℓx, ℓy, ℓz, clock, fields, 𝓋, top_bc, bottom_bc, immersed_bc) =
    domain_boundary_stress(i, j, grid, ℓx, ℓy, ℓz, clock, fields, 𝓋, top_bc, bottom_bc)

@inline function domain_boundary_stress(i, j, grid, ℓx, ℓy, ℓz, clock, fields, 𝓋, top_bc, bottom_bc)
    λᵗ = implicit_flux_coefficient(top_bc,    i, j, grid, clock, fields)
    λᵇ = implicit_flux_coefficient(bottom_bc, i, j, grid, clock, fields)

    return boundary_face_stress(i, j, size(grid, 3), -λᵗ, 𝓋) + boundary_face_stress(i, j, 1, λᵇ, 𝓋)
end

@inline set_column_stress!(::Nothing, i, j, Ω) = nothing
@inline set_column_stress!(Ωfield, i, j, Ω) = @inbounds Ωfield[i, j, 1] = Ω

# The boundary conditions are unpacked by the caller: a `FieldBoundaryConditions` holds the compiled halo-filling
# kernels, so it is not isbits and cannot be indexed into on a GPU.
@kernel function _compute_column_boundary_stress!(Ωᵁ, Ωⱽ, grid, clock, fields, u, v,
                                                  u_top, u_bottom, u_immersed, v_top, v_bottom, v_immersed)
    i, j = @index(Global, NTuple)

    set_column_stress!(Ωᵁ, i, j, column_boundary_stress(i, j, grid, Face(), Center(), Center(), clock, fields, u,
                                                        u_top, u_bottom, u_immersed))
    set_column_stress!(Ωⱽ, i, j, column_boundary_stress(i, j, grid, Center(), Face(), Center(), clock, fields, v,
                                                        v_top, v_bottom, v_immersed))
end

function compute_column_boundary_stress!(Ωᵁ, Ωⱽ, grid, clock, model_fields, u, v)
    u_bcs = u.boundary_conditions
    v_bcs = v.boundary_conditions

    launch!(architecture(grid), grid, :xy, _compute_column_boundary_stress!, Ωᵁ, Ωⱽ, grid, clock, model_fields,
            u, v, u_bcs.top, u_bcs.bottom, u_bcs.immersed, v_bcs.top, v_bcs.bottom, v_bcs.immersed)

    return nothing
end

compute_column_boundary_stress!(::Nothing, ::Nothing, args...) = nothing

@inline any_implicit_boundary_flux(bcs) = needs_implicit_solver(bcs.top) |
                                          needs_implicit_solver(bcs.bottom) |
                                          immersed_needs_implicit_solver(bcs.immersed)

@inline immersed_needs_implicit_solver(immersed_bc) = false

@inline immersed_needs_implicit_solver(immersed_bc::ImmersedBoundaryCondition) =
    needs_implicit_solver(immersed_bc.top) | needs_implicit_solver(immersed_bc.bottom)

# `Ω` is a tendency carrying the sign of its velocity component at a tripolar seam.
function materialize_column_implicit_coefficients(grid, u, v, U_bcs, V_bcs)
    u_needed = any_implicit_boundary_flux(u.boundary_conditions)
    v_needed = any_implicit_boundary_flux(v.boundary_conditions)
    return (U = u_needed ? Field{Face, Center, Nothing}(grid, boundary_conditions = U_bcs) : nothing,
            V = v_needed ? Field{Center, Face, Nothing}(grid, boundary_conditions = V_bcs) : nothing)
end
