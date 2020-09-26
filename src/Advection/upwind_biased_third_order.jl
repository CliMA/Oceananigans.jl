#####
##### Upwind-biased 3rd-order advection scheme
#####

struct UpwindBiasedThirdOrder <: AbstractUpwindBiasedAdvectionScheme
    function UpwindBiasedThirdOrder()
        @warn "UpwindBiasedThirdOrder is currently an experimental scheme and may blow up in your face!"
        return new{}()
    end
end

const U3 = UpwindBiasedThirdOrder

@inline boundary_buffer(::U3) = 1

@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U3, c) = ℑxᶠᵃᵃ(i, j, k, grid, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U3, c) = ℑyᵃᶠᵃ(i, j, k, grid, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U3, c) = ℑzᵃᵃᶠ(i, j, k, grid, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::U3, u) = ℑxᶜᵃᵃ(i, j, k, grid, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::U3, v) = ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::U3, w) = ℑzᵃᵃᶜ(i, j, k, grid, w)

@inline left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U3, c) = @inbounds (6 * ℑxᶠᵃᵃ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑxᶠᵃᵃ(i-1, j, k, grid, c)) / 6 
@inline left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U3, c) = @inbounds (6 * ℑyᵃᶠᵃ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑyᵃᶠᵃ(i, j-1, k, grid, c)) / 6 
@inline left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U3, c) = @inbounds (6 * ℑzᵃᵃᶠ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑzᵃᵃᶠ(i, j, k-1, grid, c)) / 6 

@inline left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::U3, u) = left_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
@inline left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::U3, v) = left_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
@inline left_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::U3, w) = left_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)

@inline right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::U3, c) = @inbounds (2 * ℑxᶠᵃᵃ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑxᶠᵃᵃ(i+1, j, k, grid, c)) / 6 
@inline right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::U3, c) = @inbounds (2 * ℑyᵃᶠᵃ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑyᵃᶠᵃ(i, j+1, k, grid, c)) / 6
@inline right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::U3, c) = @inbounds (2 * ℑzᵃᵃᶠ(i, j, k, grid, c) + 4 * c[i, j, k] - ℑzᵃᵃᶠ(i, j, k+1, grid, c)) / 6

@inline right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::U3, u) = right_biased_interpolate_xᶠᵃᵃ(i+1, j, k, grid, scheme, u)
@inline right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::U3, v) = right_biased_interpolate_yᵃᶠᵃ(i, j+1, k, grid, scheme, v)
@inline right_biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::U3, w) = right_biased_interpolate_zᵃᵃᶠ(i, j, k+1, grid, scheme, w)
