using Oceananigans.Grids

#####
##### Centered fourth-order advection scheme
#####

struct UpwindBiasedThirdOrder <: AbstractUpwindBiasedAdvectionScheme end

@inline halo_buffer(::UpwindBiasedThirdOrder) = 1

symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = ℑxᶠᵃᵃ(i, j, k, grid, c)
symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = ℑyᵃᶠᵃ(i, j, k, grid, c)
symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = ℑzᵃᵃᶠ(i, j, k, grid, c)

symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, u) = ℑxᶜᵃᵃ(i, j, k, grid, u)
symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, v) = ℑyᵃᶜᵃ(i, j, k, grid, v)
symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::UpwindBiasedThirdOrder, w) = ℑzᵃᵃᶜ(i, j, k, grid, w)

left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = @inbounds (6 * ℑxᶠᵃᵃ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑxᶠᵃᵃ(i-1, j, k, grid, c)) / 6 
left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = @inbounds (6 * ℑyᵃᶠᵃ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑyᵃᶠᵃ(i, j-1, k, grid, c)) / 6 
left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = @inbounds (6 * ℑzᵃᵃᶠ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑzᵃᵃᶠ(i, j, k-1, grid, c)) / 6 

left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::UpwindBiasedThirdOrder, u) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::UpwindBiasedThirdOrder, v) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::UpwindBiasedThirdOrder, w) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)

right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = @inbounds (2 * ℑxᶠᵃᵃ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑxᶠᵃᵃ(i+1, j, k, grid, c)) / 6 
right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = @inbounds (2 * ℑyᵃᶠᵃ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑyᵃᶠᵃ(i, j+1, k, grid, c)) / 6
right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::UpwindBiasedThirdOrder, c) = @inbounds (2 * ℑzᵃᵃᶠ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑzᵃᵃᶠ(i, j, k+1, grid, c)) / 6

right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::UpwindBiasedThirdOrder, u) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::UpwindBiasedThirdOrder, v) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::UpwindBiasedThirdOrder, w) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)
