using Oceananigans.Grids: inactive_node, topology
using Oceananigans.Operators: Δrᶜᶜᶜ

#####
##### Mass-consistent momentum advection at bottom-cell thickness jumps: where the wet thickness of neighbouring
##### bottom cells changes, momentum is advected in flux form, so that each u-cell's momentum moves with the
##### transports that carry its mass. Everywhere else the vector-invariant form is unchanged.
#####

@inline wet_thickness(i, j, k, grid) = ifelse(inactive_node(i, j, k, grid, Center(), Center(), Center()), zero(grid), Δrᶜᶜᶜ(i, j, k, grid))

# immersed cells (thickness 0) are ignored
@inline function wet_thickness_changes(a, b, c, d)
    hi = max(a, b, c, d)
    lo = min(ifelse(a == 0, hi, a), ifelse(b == 0, hi, b), ifelse(c == 0, hi, c), ifelse(d == 0, hi, d))
    return 100 * (hi - lo) > hi
end

# T-cells i-2 … i+1 enter the transports of the flux-form tendency at u-face i
@inline bottom_thickness_jump_x(i, j, k, grid) = wet_thickness_changes(wet_thickness(i-2, j, k, grid), wet_thickness(i-1, j, k, grid),
                                                                       wet_thickness(i,   j, k, grid), wet_thickness(i+1, j, k, grid))

@inline bottom_thickness_jump_y(i, j, k, grid) = wet_thickness_changes(wet_thickness(i, j-2, k, grid), wet_thickness(i, j-1, k, grid),
                                                                       wet_thickness(i, j,   k, grid), wet_thickness(i, j+1, k, grid))

# vector-invariant schemes whose kinetic-energy gradient uses a reconstruction scheme that also has momentum fluxes
const VectorInvariantWithFluxScheme = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                      <:Union{AbstractCenteredAdvectionScheme, AbstractUpwindBiasedAdvectionScheme}}

# `if`, not `ifelse`: only one of the two advection forms is computed at each face
@inline function U_dot_∇u(i, j, k, grid::PCBIBG, scheme::VectorInvariantWithFluxScheme, U)
    if topology(grid, 1) !== Flat && bottom_thickness_jump_x(i, j, k, grid)
        flux_form = scheme.kinetic_energy_gradient_scheme
        return div_𝐯u(i, j, k, grid, flux_form, U, U.u) + U_dot_∇u_hydrostatic_metric(i, j, k, grid, flux_form, U, U)
    else
        return horizontal_advection_U(i, j, k, grid, scheme, U.u, U.v) +
               vertical_advection_U(i, j, k, grid, scheme, U) +
               bernoulli_head_U(i, j, k, grid, scheme, U.u, U.v)
    end
end

@inline function U_dot_∇v(i, j, k, grid::PCBIBG, scheme::VectorInvariantWithFluxScheme, U)
    if topology(grid, 2) !== Flat && bottom_thickness_jump_y(i, j, k, grid)
        flux_form = scheme.kinetic_energy_gradient_scheme
        return div_𝐯v(i, j, k, grid, flux_form, U, U.v) + U_dot_∇v_hydrostatic_metric(i, j, k, grid, flux_form, U, U)
    else
        return horizontal_advection_V(i, j, k, grid, scheme, U.u, U.v) +
               vertical_advection_V(i, j, k, grid, scheme, U) +
               bernoulli_head_V(i, j, k, grid, scheme, U.u, U.v)
    end
end
