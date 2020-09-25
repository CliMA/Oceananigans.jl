using Oceananigans.Grids

#####
##### Centered fourth-order advection scheme
#####

struct CenteredFourthOrder <: AbstractAdvectionScheme end

@inline halo_buffer(::CenteredFourthOrder) = 1

@inline ℑ³xᶠᵃᵃ(i, j, k, grid, u) = @inbounds u[i, j, k] - δxᶠᵃᵃ(i, j, k, grid, δxᶜᵃᵃ, u) / 6
@inline ℑ³xᶜᵃᵃ(i, j, k, grid, c) = @inbounds c[i, j, k] - δxᶜᵃᵃ(i, j, k, grid, δxᶠᵃᵃ, c) / 6

@inline ℑ³yᵃᶠᵃ(i, j, k, grid, v) = @inbounds v[i, j, k] - δyᵃᶠᵃ(i, j, k, grid, δyᵃᶜᵃ, v) / 6
@inline ℑ³yᵃᶜᵃ(i, j, k, grid, c) = @inbounds c[i, j, k] - δyᵃᶜᵃ(i, j, k, grid, δyᵃᶠᵃ, c) / 6

@inline ℑ³zᵃᵃᶠ(i, j, k, grid, w) = @inbounds w[i, j, k] - δzᵃᵃᶠ(i, j, k, grid, δzᵃᵃᶜ, w) / 6
@inline ℑ³zᵃᵃᶜ(i, j, k, grid, c) = @inbounds c[i, j, k] - δzᵃᵃᶜ(i, j, k, grid, δzᵃᵃᶠ, c) / 6

symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::CenteredFourthOrder, u) = ℑxᶜᵃᵃ(i, j, k, grid, ℑ³xᶠᵃᵃ, u)
symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::CenteredFourthOrder, c) = ℑxᶠᵃᵃ(i, j, k, grid, ℑ³xᶜᵃᵃ, c)

symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::CenteredFourthOrder, v) = ℑyᵃᶜᵃ(i, j, k, grid, ℑ³yᵃᶠᵃ, v)
symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::CenteredFourthOrder, c) = ℑyᵃᶠᵃ(i, j, k, grid, ℑ³yᵃᶜᵃ, c)

symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::CenteredFourthOrder, w) = ℑzᵃᵃᶜ(i, j, k, grid, ℑ³zᵃᵃᶠ, w)
symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::CenteredFourthOrder, c) = ℑzᵃᵃᶠ(i, j, k, grid, ℑ³zᵃᵃᶜ, c)
