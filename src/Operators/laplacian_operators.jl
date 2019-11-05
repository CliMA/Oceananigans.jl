####
#### Horizontal Laplacians
####

@inline function ∇²hᶜᶜᵃ(i, j, k, grid, c)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᵃᵃ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᵃᶠᵃ, c))
end

@inline function ∇²hᶠᶜᵃ(i, j, k, grid, u)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_∂xᶜᵃᵃ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᵃᶠᵃ, u))
end

@inline function ∇²hᶜᶠᵃ(i, j, k, grid, u)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᵃᵃ, u) +
                                    δyᵃᶠᵃ(i, j, k, grid, Ay_∂yᵃᶜᵃ, u))
end

####
#### 3D Laplacian
####

"""
    ∇²(i, j, k, grid, c)

Calculates the Laplacian of c via

    1/V * [δxᶜᵃᵃ(Ax * ∂xᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * ∂yᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * ∂zᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function ∇²(i, j, k, grid, c)
    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᵃᵃ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᵃᶠᵃ, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_∂zᵃᵃᶠ, c))
end

####
#### Horizontal biharmonic operators
####

@inline ∇h⁴_cca(i, j, k, grid, c::AbstractArray) = ∇h²_cca(i, j, k, grid, ∇h²_cca, c)
@inline ∇h⁴_fca(i, j, k, grid, c::AbstractArray) = ∇h²_fca(i, j, k, grid, ∇h²_fca, c)
@inline ∇h⁴_cfa(i, j, k, grid, c::AbstractArray) = ∇h²_cfa(i, j, k, grid, ∇h²_cfa, c)
