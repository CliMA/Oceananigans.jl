#####
##### Centered fourth-order advection scheme
#####

"""
    struct CenteredFourthOrder <: AbstractCenteredAdvectionScheme{1}

Centered fourth-order advection scheme.
"""
struct CenteredFourthOrder <: AbstractCenteredAdvectionScheme{1} end

const C4 = CenteredFourthOrder
const centered_fourth_order = C4()

@inline boundary_buffer(::C4) = 1

@inline ℑ³xᶠᵃᵃ(i, j, k, grid, u) = @inbounds u[i, j, k] - δxᶠᵃᵃ(i, j, k, grid, δxᶜᵃᵃ, u) / 6
@inline ℑ³xᶜᵃᵃ(i, j, k, grid, c) = @inbounds c[i, j, k] - δxᶜᵃᵃ(i, j, k, grid, δxᶠᵃᵃ, c) / 6

@inline ℑ³yᵃᶠᵃ(i, j, k, grid, v) = @inbounds v[i, j, k] - δyᵃᶠᵃ(i, j, k, grid, δyᵃᶜᵃ, v) / 6
@inline ℑ³yᵃᶜᵃ(i, j, k, grid, c) = @inbounds c[i, j, k] - δyᵃᶜᵃ(i, j, k, grid, δyᵃᶠᵃ, c) / 6

@inline ℑ³zᵃᵃᶠ(i, j, k, grid, w) = @inbounds w[i, j, k] - δzᵃᵃᶠ(i, j, k, grid, δzᵃᵃᶜ, w) / 6
@inline ℑ³zᵃᵃᶜ(i, j, k, grid, c) = @inbounds c[i, j, k] - δzᵃᵃᶜ(i, j, k, grid, δzᵃᵃᶠ, c) / 6

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::C4, u) = ℑxᶜᵃᵃ(i, j, k, grid, ℑ³xᶠᵃᵃ, u)
@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::C4, c) = ℑxᶠᵃᵃ(i, j, k, grid, ℑ³xᶜᵃᵃ, c)

@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::C4, v) = ℑyᵃᶜᵃ(i, j, k, grid, ℑ³yᵃᶠᵃ, v)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::C4, c) = ℑyᵃᶠᵃ(i, j, k, grid, ℑ³yᵃᶜᵃ, c)

@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::C4, w) = ℑzᵃᵃᶜ(i, j, k, grid, ℑ³zᵃᵃᶠ, w)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::C4, c) = ℑzᵃᵃᶠ(i, j, k, grid, ℑ³zᵃᵃᶜ, c)
