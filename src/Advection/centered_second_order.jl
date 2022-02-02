#####
##### Centered second-order advection scheme
#####

"""
    struct CenteredSecondOrder <: AbstractAdvectionScheme{0}

Centered second-order advection scheme.
"""
struct CenteredSecondOrder <: AbstractAdvectionScheme{0} end

boundary_buffer(::CenteredSecondOrder) = 0

const C2 = CenteredSecondOrder

@inline advective_momentum_flux_Uu(i, j, k, grid, ::C2, U, u) = ℑxᶜᵃᵃ(i, j, k, grid, Ax_uᶠᶜᶜ, U) * ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline advective_momentum_flux_Vu(i, j, k, grid, ::C2, V, u) = ℑxᶠᵃᵃ(i, j, k, grid, Ay_vᶜᶠᶜ, V) * ℑyᵃᶠᵃ(i, j, k, grid, u)
@inline advective_momentum_flux_Wu(i, j, k, grid, ::C2, W, u) = ℑxᶠᵃᵃ(i, j, k, grid, Az_wᶜᶜᵃ, W) * ℑzᵃᵃᶠ(i, j, k, grid, u)

@inline advective_momentum_flux_Uv(i, j, k, grid, ::C2, U, v) = ℑyᵃᶠᵃ(i, j, k, grid, Ax_uᶠᶜᶜ, U) * ℑxᶠᵃᵃ(i, j, k, grid, v)
@inline advective_momentum_flux_Vv(i, j, k, grid, ::C2, V, v) = ℑyᵃᶜᵃ(i, j, k, grid, Ay_vᶜᶠᶜ, V) * ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline advective_momentum_flux_Wv(i, j, k, grid, ::C2, W, v) = ℑyᵃᶠᵃ(i, j, k, grid, Az_wᶜᶜᵃ, W) * ℑzᵃᵃᶠ(i, j, k, grid, v)

@inline advective_momentum_flux_Uw(i, j, k, grid, ::C2, U, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ax_uᶠᶜᶜ, U) * ℑxᶠᵃᵃ(i, j, k, grid, w)
@inline advective_momentum_flux_Vw(i, j, k, grid, ::C2, V, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ay_vᶜᶠᶜ, V) * ℑyᵃᶠᵃ(i, j, k, grid, w)
@inline advective_momentum_flux_Ww(i, j, k, grid, ::C2, W, w) = ℑzᵃᵃᶜ(i, j, k, grid, Az_wᶜᶜᵃ, W) * ℑzᵃᵃᶜ(i, j, k, grid, w)

# Calculate the flux of a tracer quantity c through the faces of a cell.
# In this case, the fluxes are given by u*Ax*c̄ˣ, v*Ay*c̄ʸ, and w*Az*c̄ᶻ.
@inline advective_tracer_flux_x(i, j, k, grid, ::C2, U, c) = Ax_uᶠᶜᶜ(i, j, k, grid, U) * ℑxᶠᵃᵃ(i, j, k, grid, c)
@inline advective_tracer_flux_y(i, j, k, grid, ::C2, V, c) = Ay_vᶜᶠᶜ(i, j, k, grid, V) * ℑyᵃᶠᵃ(i, j, k, grid, c)
@inline advective_tracer_flux_z(i, j, k, grid, ::C2, W, c) = Az_wᶜᶜᶠ(i, j, k, grid, W) * ℑzᵃᵃᶠ(i, j, k, grid, c)
