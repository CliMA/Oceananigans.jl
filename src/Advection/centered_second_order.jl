#####
##### Centered second-order advection scheme
#####

struct CenteredSecondOrder <: AbstractAdvectionScheme end

const C2 = CenteredSecondOrder

@inline momentum_flux_uu(i, j, k, grid, ::C2, u)    = ℑxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline momentum_flux_uv(i, j, k, grid, ::C2, u, v) = ℑxᶠᵃᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * ℑyᵃᶠᵃ(i, j, k, grid, u)
@inline momentum_flux_uw(i, j, k, grid, ::C2, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * ℑzᵃᵃᶠ(i, j, k, grid, u)

@inline momentum_flux_vu(i, j, k, grid, ::C2, u, v) = ℑyᵃᶠᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * ℑxᶠᵃᵃ(i, j, k, grid, v)
@inline momentum_flux_vv(i, j, k, grid, ::C2, v)    = ℑyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline momentum_flux_vw(i, j, k, grid, ::C2, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * ℑzᵃᵃᶠ(i, j, k, grid, v)

@inline momentum_flux_wu(i, j, k, grid, ::C2, u, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * ℑxᶠᵃᵃ(i, j, k, grid, w)
@inline momentum_flux_wv(i, j, k, grid, ::C2, v, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * ℑyᵃᶠᵃ(i, j, k, grid, w)
@inline momentum_flux_ww(i, j, k, grid, ::C2, w)    = ℑzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * ℑzᵃᵃᶜ(i, j, k, grid, w)

# Calculate the flux of a tracer quantity c through the faces of a cell.
# In this case, the fluxes are given by u*Ax*T̅ˣ, v*Ay*T̅ʸ, and w*Az*T̅ᶻ.
@inline advective_tracer_flux_x(i, j, k, grid, ::C2, u, c) = Ax_ψᵃᵃᶠ(i, j, k, grid, u) * ℑxᶠᵃᵃ(i, j, k, grid, c)
@inline advective_tracer_flux_y(i, j, k, grid, ::C2, v, c) = Ay_ψᵃᵃᶠ(i, j, k, grid, v) * ℑyᵃᶠᵃ(i, j, k, grid, c)
@inline advective_tracer_flux_z(i, j, k, grid, ::C2, w, c) = Az_ψᵃᵃᵃ(i, j, k, grid, w) * ℑzᵃᵃᶠ(i, j, k, grid, c)
