#####
##### Centered second-order advection scheme
#####

struct CenteredSecondOrder <: AbstractAdvectionScheme end

boundary_buffer(::CenteredSecondOrder) = 0

const C2 = CenteredSecondOrder

@inline momentum_flux_uu(i, j, k, grid, ::C2, U, u) = ℑxᶜᵃᵃ(i, j, k, grid, Ax_uᶠᶜᶜ, U) * ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline momentum_flux_uv(i, j, k, grid, ::C2, V, u) = ℑxᶠᵃᵃ(i, j, k, grid, Ay_vᶜᶠᶜ, V) * ℑyᵃᶠᵃ(i, j, k, grid, u)
@inline momentum_flux_uw(i, j, k, grid, ::C2, W, u) = ℑxᶠᵃᵃ(i, j, k, grid, Az_wᶜᶜᵃ, W) * ℑzᵃᵃᶠ(i, j, k, grid, u)

@inline momentum_flux_vu(i, j, k, grid, ::C2, U, v) = ℑyᵃᶠᵃ(i, j, k, grid, Ax_uᶠᶜᶜ, U) * ℑxᶠᵃᵃ(i, j, k, grid, v)
@inline momentum_flux_vv(i, j, k, grid, ::C2, V, v) = ℑyᵃᶜᵃ(i, j, k, grid, Ay_vᶜᶠᶜ, V) * ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline momentum_flux_vw(i, j, k, grid, ::C2, W, v) = ℑyᵃᶠᵃ(i, j, k, grid, Az_wᶜᶜᵃ, W) * ℑzᵃᵃᶠ(i, j, k, grid, v)

@inline momentum_flux_wu(i, j, k, grid, ::C2, U, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ax_uᶠᶜᶜ, U) * ℑxᶠᵃᵃ(i, j, k, grid, w)
@inline momentum_flux_wv(i, j, k, grid, ::C2, V, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ay_vᶜᶠᶜ, V) * ℑyᵃᶠᵃ(i, j, k, grid, w)
@inline momentum_flux_ww(i, j, k, grid, ::C2, W, w) = ℑzᵃᵃᶜ(i, j, k, grid, Az_wᶜᶜᵃ, W) * ℑzᵃᵃᶜ(i, j, k, grid, w)

# Calculate the flux of a tracer quantity c through the faces of a cell.
# In this case, the fluxes are given by u*Ax*c̄ˣ, v*Ay*c̄ʸ, and w*Az*c̄ᶻ.
@inline advective_tracer_flux_x(i, j, k, grid, ::C2, U, c) = Ax_uᶠᶜᶜ(i, j, k, grid, U) * ℑxᶠᵃᵃ(i, j, k, grid, c)
@inline advective_tracer_flux_y(i, j, k, grid, ::C2, V, c) = Ay_vᶜᶠᶜ(i, j, k, grid, V) * ℑyᵃᶠᵃ(i, j, k, grid, c)
@inline advective_tracer_flux_z(i, j, k, grid, ::C2, W, c) = Az_wᶜᶜᵃ(i, j, k, grid, W) * ℑzᵃᵃᶠ(i, j, k, grid, c)
