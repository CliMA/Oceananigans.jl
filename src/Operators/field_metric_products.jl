#####
##### Products between fields and grid lengths
#####

@inline Δx_uᶠᶜᵃ(i, j, k, grid, u) = @inbounds Δxᶠᶜᵃ(i, j, k, grid) * u[i, j, k]
@inline Δx_vᶜᶠᵃ(i, j, k, grid, v) = @inbounds Δxᶜᶠᵃ(i, j, k, grid) * v[i, j, k]

@inline Δy_uᶠᶜᵃ(i, j, k, grid, u) = @inbounds Δyᶠᶜᵃ(i, j, k, grid) * u[i, j, k]
@inline Δy_vᶜᶠᵃ(i, j, k, grid, v) = @inbounds Δyᶜᶠᵃ(i, j, k, grid) * v[i, j, k]

#####
##### Products between fields and grid areas
#####

@inline Ax_ψᵃᵃᶠ(i, j, k, grid, u) = @inbounds Axᵃᵃᶠ(i, j, k, grid) * u[i, j, k]
@inline Ax_ψᵃᵃᶜ(i, j, k, grid, c) = @inbounds Axᵃᵃᶜ(i, j, k, grid) * c[i, j, k]
@inline Ay_ψᵃᵃᶠ(i, j, k, grid, v) = @inbounds Ayᵃᵃᶠ(i, j, k, grid) * v[i, j, k]
@inline Ay_ψᵃᵃᶜ(i, j, k, grid, c) = @inbounds Ayᵃᵃᶜ(i, j, k, grid) * c[i, j, k]
@inline Az_ψᵃᵃᵃ(i, j, k, grid, c) = @inbounds Azᵃᵃᵃ(i, j, k, grid) * c[i, j, k]

@inline Az_wᶜᶜᵃ(i, j, k, grid, w) = @inbounds Azᶜᶜᵃ(i, j, k, grid) * w[i, j, k]
