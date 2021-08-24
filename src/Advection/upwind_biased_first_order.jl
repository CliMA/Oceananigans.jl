#####
##### Upwind-biased 1rd-order advection scheme
#####

struct UpwindBiasedFirstOrder <: AbstractUpwindBiasedAdvectionScheme{1} end

const U1 = UpwindBiasedFirstOrder

@inline boundary_buffer(::U1) = 1

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U1, c) = ℑxᶠᵃᵃ(i, j, k, grid, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U1, c) = ℑyᵃᶠᵃ(i, j, k, grid, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U1, c) = ℑzᵃᵃᶠ(i, j, k, grid, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::U1, u) = ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::U1, v) = ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::U1, w) = ℑzᵃᵃᶜ(i, j, k, grid, w)

@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U1, c) = @inbounds c[i-1, j, k] 
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U1, c) = @inbounds c[i, j-1, k]
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U1, c) = @inbounds c[i, j, k-1]

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::U1, u) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::U1, v) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::U1, w) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U1, c) = @inbounds c[i, j, k] 
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U1, c) = @inbounds c[i, j, k]
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U1, c) = @inbounds c[i, j, k]

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::U1, u) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::U1, v) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::U1, w) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)
