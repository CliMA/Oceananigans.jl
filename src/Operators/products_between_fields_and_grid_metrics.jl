#####
##### Products between fields and grid lengths
#####

@inline Δx_uᶠᶜᵃ(i, j, k, grid, u) = @inbounds Δxᶠᶜᵃ(i, j, k, grid) * u[i, j, k]
@inline Δx_vᶜᶠᵃ(i, j, k, grid, v) = @inbounds Δxᶜᶠᵃ(i, j, k, grid) * v[i, j, k]
@inline Δx_cᶜᶜᵃ(i, j, k, grid, c) = @inbounds Δxᶜᶜᵃ(i, j, k, grid) * c[i, j, k]

@inline Δy_uᶠᶜᵃ(i, j, k, grid, u) = @inbounds Δyᶠᶜᵃ(i, j, k, grid) * u[i, j, k]
@inline Δy_vᶜᶠᵃ(i, j, k, grid, v) = @inbounds Δyᶜᶠᵃ(i, j, k, grid) * v[i, j, k]
@inline Δy_cᶜᶜᵃ(i, j, k, grid, c) = @inbounds Δyᶜᶜᵃ(i, j, k, grid) * c[i, j, k]

@inline Δy_uᶠᶜᵃ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Δyᶠᶜᵃ(i, j, k, grid) * f(i, j, k, grid, args...)
@inline Δy_vᶜᶠᵃ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Δyᶜᶠᵃ(i, j, k, grid) * f(i, j, k, grid, args...)
@inline Δy_cᶜᶜᵃ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Δyᶜᶜᵃ(i, j, k, grid) * f(i, j, k, grid, args...)

@inline Δx_uᶠᶜᵃ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Δxᶠᶜᵃ(i, j, k, grid) * f(i, j, k, grid, args...)
@inline Δx_vᶜᶠᵃ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Δxᶜᶠᵃ(i, j, k, grid) * f(i, j, k, grid, args...)
@inline Δx_cᶜᶜᵃ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Δxᶜᶜᵃ(i, j, k, grid) * f(i, j, k, grid, args...)

#####
##### Products between fields and grid areas
#####

@inline Ax_ψᵃᵃᶠ(i, j, k, grid, u) = @inbounds Axᵃᵃᶠ(i, j, k, grid) * u[i, j, k]
@inline Ax_ψᵃᵃᶜ(i, j, k, grid, c) = @inbounds Axᵃᵃᶜ(i, j, k, grid) * c[i, j, k]
@inline Ay_ψᵃᵃᶠ(i, j, k, grid, v) = @inbounds Ayᵃᵃᶠ(i, j, k, grid) * v[i, j, k]
@inline Ay_ψᵃᵃᶜ(i, j, k, grid, c) = @inbounds Ayᵃᵃᶜ(i, j, k, grid) * c[i, j, k]
@inline Az_ψᵃᵃᵃ(i, j, k, grid, c) = @inbounds Azᵃᵃᵃ(i, j, k, grid) * c[i, j, k]

@inline Ax_uᶠᶜᶜ(i, j, k, grid, u) = @inbounds Axᶠᶜᶜ(i, j, k, grid) * u[i, j, k]
@inline Ay_vᶜᶠᶜ(i, j, k, grid, v) = @inbounds Ayᶜᶠᶜ(i, j, k, grid) * v[i, j, k]
@inline Az_wᶜᶜᵃ(i, j, k, grid, w) = @inbounds Azᶜᶜᵃ(i, j, k, grid) * w[i, j, k]

@inline Ax_uᶠᶜᶜ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Axᶠᶜᶜ(i, j, k, grid) * f(i, j, k, grid, args...)
@inline Ay_vᶜᶠᶜ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Ayᶜᶠᶜ(i, j, k, grid) * f(i, j, k, grid, args...)
@inline Az_wᶜᶜᵃ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Azᶜᶜᵃ(i, j, k, grid) * f(i, j, k, grid, args...)

@inline Ax_cᶜᶜᶜ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Axᶜᶜᶜ(i, j, k, grid) * f(i, j, k, grid, args...)
@inline Ay_cᶜᶜᶜ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Ayᶜᶜᶜ(i, j, k, grid) * f(i, j, k, grid, args...)
@inline Az_cᶜᶜᵃ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Azᶜᶜᵃ(i, j, k, grid) * f(i, j, k, grid, args...)

@inline Ax_ζᶠᶠᶜ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Axᶠᶠᶜ(i, j, k, grid) * f(i, j, k, grid, args...)
@inline Ay_ζᶠᶠᶜ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Ayᶠᶠᶜ(i, j, k, grid) * f(i, j, k, grid, args...)

@inline Ax_ηᶠᶜᶠ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Axᶠᶜᶠ(i, j, k, grid) * f(i, j, k, grid, args...)
@inline Az_ηᶠᶜᵃ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Azᶠᶜᵃ(i, j, k, grid) * f(i, j, k, grid, args...)

@inline Ay_ξᶜᶠᶠ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Ayᶜᶠᶠ(i, j, k, grid) * f(i, j, k, grid, args...)
@inline Az_ξᶜᶠᵃ(i, j, k, grid, f::F, args...) where F <: Function = @inbounds Azᶜᶠᵃ(i, j, k, grid) * f(i, j, k, grid, args...)
