using Oceananigans.Grids

#####
##### Centered fourth-order advection scheme
#####

struct UpwindBiasedThirdOrder <: AbstractUpwindBiasedAdvectionScheme end

@inline boundary_buffer(::UpwindBiasedThirdOrder) = 1

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = ℑxᶠᵃᵃ(i, j, k, grid, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = ℑyᵃᶠᵃ(i, j, k, grid, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = ℑzᵃᵃᶠ(i, j, k, grid, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, u) = ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, v) = ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::UpwindBiasedThirdOrder, w) = ℑzᵃᵃᶜ(i, j, k, grid, w)

@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = @inbounds (6 * ℑxᶠᵃᵃ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑxᶠᵃᵃ(i-1, j, k, grid, c)) / 6 
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = @inbounds (6 * ℑyᵃᶠᵃ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑyᵃᶠᵃ(i, j-1, k, grid, c)) / 6 
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = @inbounds (6 * ℑzᵃᵃᶠ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑzᵃᵃᶠ(i, j, k-1, grid, c)) / 6 

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::UpwindBiasedThirdOrder, u) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::UpwindBiasedThirdOrder, v) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::UpwindBiasedThirdOrder, w) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = @inbounds (2 * ℑxᶠᵃᵃ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑxᶠᵃᵃ(i+1, j, k, grid, c)) / 6 
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = @inbounds (2 * ℑyᵃᶠᵃ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑyᵃᶠᵃ(i, j+1, k, grid, c)) / 6
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = @inbounds (2 * ℑzᵃᵃᶠ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑzᵃᵃᶠ(i, j, k+1, grid, c)) / 6

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::UpwindBiasedThirdOrder, u) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::UpwindBiasedThirdOrder, v) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::UpwindBiasedThirdOrder, w) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)
