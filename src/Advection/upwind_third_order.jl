#####
##### Centered second-order advection scheme
#####

struct UpwindThirdOrder <: AbstractAdvectionScheme end

const U3 = UpwindThirdOrder

const centered_fourth_order = CenteredFourthOrder()

δ³xᶠᵃᵃ(i, j, k, grid, c) = δxᶠᵃᵃ(i, j, k, grid, δxᶜᵃᵃ, δxᶠᵃᵃ, c)
δ³yᵃᶠᵃ(i, j, k, grid, c) = δyᵃᶠᵃ(i, j, k, grid, δyᵃᶜᵃ, δyᵃᶠᵃ, c)
δ³zᵃᵃᶠ(i, j, k, grid, c) = δzᵃᵃᶠ(i, j, k, grid, δzᵃᵃᶜ, δzᵃᵃᶠ, c)

δ³xᶜᵃᵃ(i, j, k, grid, u) = δxᶜᵃᵃ(i, j, k, grid, δxᶠᵃᵃ, δxᶜᵃᵃ, u)
δ³yᵃᶜᵃ(i, j, k, grid, v) = δyᵃᶜᵃ(i, j, k, grid, δyᵃᶠᵃ, δyᵃᶜᵃ, v)
δ³zᵃᵃᶜ(i, j, k, grid, w) = δzᵃᵃᶜ(i, j, k, grid, δzᵃᵃᶠ, δzᵃᵃᶜ, w)

@inline momentum_flux_uu(i, j, k, grid, ::U3, u)    = momentum_flux_uu(i, j, k, grid, centered_second_order, u)
@inline momentum_flux_uv(i, j, k, grid, ::U3, u, v) = momentum_flux_uv(i, j, k, grid, centered_second_order, u, v)
@inline momentum_flux_uw(i, j, k, grid, ::U3, u, w) = momentum_flux_uw(i, j, k, grid, centered_second_order, u, w)

@inline momentum_flux_vu(i, j, k, grid, ::U3, u, v) = momentum_flux_vu(i, j, k, grid, centered_second_order, u, v)
@inline momentum_flux_vv(i, j, k, grid, ::U3, v)    = momentum_flux_vv(i, j, k, grid, centered_second_order, v)
@inline momentum_flux_vw(i, j, k, grid, ::U3, v, w) = momentum_flux_vw(i, j, k, grid, centered_second_order, v, w)

@inline momentum_flux_wu(i, j, k, grid, ::U3, u, w) = momentum_flux_wu(i, j, k, grid, centered_second_order, u, w)
@inline momentum_flux_wv(i, j, k, grid, ::U3, v, w) = momentum_flux_wv(i, j, k, grid, centered_second_order, v, w)
@inline momentum_flux_ww(i, j, k, grid, ::U3, w)    = momentum_flux_ww(i, j, k, grid, centered_second_order, w)

# Calculate the flux of a tracer quantity c through the faces of a cell.
# In this case, the fluxes are given by u*Ax*T̅ˣ, v*Ay*T̅ʸ, and w*Az*T̅ᶻ.
@inline third_order_upwind_advective_tracer_flux_x(i, j, k, grid, ::U3, u, c) = (
                 Ax_ψᵃᵃᶠ(i, j, k, grid, u)  *  ℑxᶠᵃᵃ(i, j, k, grid, ℑ³xᶜᵃᵃ, c)
    + 1/12 * abs(Ax_ψᵃᵃᶠ(i, j, k, grid, u)) * δ³xᶠᵃᵃ(i, j, k, grid, c)
)

@inline third_order_upwind_advective_tracer_flux_y(i, j, k, grid, ::U3, v, c) = (
                     Ay_ψᵃᵃᶠ(i, j, k, grid, v)  *  ℑyᵃᶠᵃ(i, j, k, grid, ℑ³yᵃᶜᵃ, c)
        + 1/12 * abs(Ay_ψᵃᵃᶠ(i, j, k, grid, v)) * δ³yᵃᶠᵃ(i, j, k, grid, c)
)

@inline third_order_upwind_advective_tracer_flux_z(i, j, k, grid, ::U3, v, c) = (
                     Az_ψᵃᵃᵃ(i, j, k, grid, w)  *  ℑzᵃᵃᶠ(i, j, k, grid, ℑ³zᵃᵃᶜ, c)
        + 1/12 * abs(Az_ψᵃᵃᵃ(i, j, k, grid, w)) * δ³zᵃᵃᶠ(i, j, k, grid, c)
)

@inline advective_tracer_flux_x(i, j, k, grid, ::U3, u, c) = third_order_upwind_advective_tracer_flux_x(i, j, k, grid, u, c)
@inline advective_tracer_flux_y(i, j, k, grid, ::U3, v, c) = third_order_upwind_advective_tracer_flux_y(i, j, k, grid, v, c)
@inline advective_tracer_flux_z(i, j, k, grid, ::U3, w, c) = third_order_upwind_advective_tracer_flux_z(i, j, k, grid, w, c)

@inline function advective_tracer_flux_x(i, j, k, grid::AbstractGrid{FT, <:Bounded}, ::U3, u, c) where FT
    if i > 1 && i < grid.Nx
        return third_order_upwind_advective_tracer_flux_x(i, j, k, grid, u, c)
    else
        return advective_tracer_flux_x(i, j, k, grid, centered_second_order, u, c)
    end
end

@inline function advective_tracer_flux_y(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, ::U3, v, c) where {FT, TX}
    if j > 1 && i < grid.Ny
        return third_order_upwind_advective_tracer_flux_x(i, j, k, grid, v, c)
    else
        return advective_tracer_flux_y(i, j, k, grid, centered_second_order, v, c)
    end
end

@inline function advective_tracer_flux_z(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, ::U3, w, c) where {FT, TX, TY}
    if k > 1 && i < grid.Nz
        return third_order_upwind_advective_tracer_flux_x(i, j, k, grid, w, c)
    else
        return advective_tracer_flux_z(i, j, k, grid, centered_second_order, w, c)
    end
end
