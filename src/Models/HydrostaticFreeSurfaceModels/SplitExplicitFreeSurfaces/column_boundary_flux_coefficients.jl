using Oceananigans.BoundaryConditions: implicit_flux_coefficient, immersed_implicit_flux_coefficient,
                                       needs_implicit_solver
using Oceananigans.ImmersedBoundaries: immersed_inactive_node, ImmersedBoundaryCondition

#####
##### Column-integrated boundary stress
#####
##### The vertical integral of ∂τ/∂z telescopes to the boundary values, so an affine flux `J = Fₑ + λ 𝓋ᵦ`
##### contributes `λ 𝓋ᵦ` to the depth-integrated momentum. The vertical solver takes that contribution out
##### of `Gᵁ`, so `Ω` carries it into the substepping instead, as a `U`-independent tendency.
#####

@inline function column_boundary_stress(i, j, grid, ℓx, ℓy, ℓz, clock, fields, 𝓋,
                                        top_bc, bottom_bc, immersed_bc::ImmersedBoundaryCondition)
    Ω = domain_boundary_stress(i, j, grid, ℓx, ℓy, ℓz, clock, fields, 𝓋, top_bc, bottom_bc)

    for k in 1:size(grid, 3)
        active = !immersed_inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

        # `immersed_inactive_node` is false outside the domain, so a column wet to `k = 1` or `k = Nz` sees
        # no immersed face there and the domain boundary condition is counted once.
        on_bottom = active & immersed_inactive_node(i, j, k-1, grid, ℓx, ℓy, ℓz)
        on_top    = active & immersed_inactive_node(i, j, k+1, grid, ℓx, ℓy, ℓz)

        λᵇ = immersed_implicit_flux_coefficient(immersed_bc.bottom, i, j, k, grid, clock, fields)
        λᵗ = immersed_implicit_flux_coefficient(immersed_bc.top,    i, j, k, grid, clock, fields)

        λ = ifelse(on_bottom, λᵇ, zero(grid)) + ifelse(on_top, λᵗ, zero(grid))
        Ω += @inbounds λ * 𝓋[i, j, k]
    end

    return Ω
end

@inline column_boundary_stress(i, j, grid, ℓx, ℓy, ℓz, clock, fields, 𝓋, top_bc, bottom_bc, immersed_bc) =
    domain_boundary_stress(i, j, grid, ℓx, ℓy, ℓz, clock, fields, 𝓋, top_bc, bottom_bc)

# `Ω` is a tendency, so the domain top enters with `+λ` and the domain bottom with `-λ`: a surface stress
# and a bottom drag both decelerate the column.
@inline function domain_boundary_stress(i, j, grid, ℓx, ℓy, ℓz, clock, fields, 𝓋, top_bc, bottom_bc)
    Nz = size(grid, 3)
    λᵗ = implicit_flux_coefficient(top_bc,    i, j, grid, clock, fields)
    λᵇ = implicit_flux_coefficient(bottom_bc, i, j, grid, clock, fields)

    return @inbounds λᵇ * 𝓋[i, j, 1] - λᵗ * 𝓋[i, j, Nz]
end

@inline set_column_stress!(::Nothing, i, j, Ω) = nothing
@inline set_column_stress!(Ωfield, i, j, Ω) = @inbounds Ωfield[i, j, 1] = Ω

# A `FieldBoundaryConditions` holds the compiled halo-filling kernels, so it is not isbits: the sides are
# unpacked by the caller.
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

function materialize_column_implicit_coefficients(grid, u, v, U_bcs, V_bcs)
    u_needed = needs_implicit_solver(u.boundary_conditions)
    v_needed = needs_implicit_solver(v.boundary_conditions)
    return (U = u_needed ? Field{Face, Center, Nothing}(grid, boundary_conditions = U_bcs) : nothing,
            V = v_needed ? Field{Center, Face, Nothing}(grid, boundary_conditions = V_bcs) : nothing)
end
