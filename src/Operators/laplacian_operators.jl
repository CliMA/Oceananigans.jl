#####
##### Horizontal Laplacians
#####

@inline function ∇²hᶜᶜᶜ(i, j, k, grid, c)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᶜᶜ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᶜᶠᶜ, c))
end

@inline function ∇²hᶠᶜᶜ(i, j, k, grid, u)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_∂xᶜᶜᶜ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᶠᶠᶜ, u))
end

@inline function ∇²hᶜᶠᶜ(i, j, k, grid, v)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᶠᶜ, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, Ay_∂yᶜᶜᶜ, v))
end

@inline ∇²hᶜᶜᵃ(i, j, k, grid, F::FU, args...) where FU <: Function = ∂²xᶜᵃᵃ(i, j, k, grid, F, args...) + ∂²yᵃᶜᵃ(i, j, k, grid, F, args...)
@inline ∇²hᶠᶜᵃ(i, j, k, grid, F::FU, args...) where FU <: Function = ∂²xᶠᵃᵃ(i, j, k, grid, F, args...) + ∂²yᵃᶜᵃ(i, j, k, grid, F, args...)
@inline ∇²hᶜᶠᵃ(i, j, k, grid, F::FU, args...) where FU <: Function = ∂²xᶜᵃᵃ(i, j, k, grid, F, args...) + ∂²yᵃᶠᵃ(i, j, k, grid, F, args...)

#####
##### 3D Laplacian
#####

"""
    ∇²(i, j, k, grid, c)

Calculates the Laplacian of c via

    1/V * [δxᶜᵃᵃ(Ax * ∂xᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * ∂yᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * ∂zᵃᵃᶠ(c))]

which will end up at the location `ccc`.
"""
@inline function ∇²ᶜᶜᶜ(i, j, k, grid, c)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᶜᶜ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᶜᶠᶜ, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_∂zᶜᶜᶠ, c))
end
