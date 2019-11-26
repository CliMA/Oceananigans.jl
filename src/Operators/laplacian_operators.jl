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

@inline ∇²hᶜᶜᵃ(i, j, k, grid, F::FU, args...) where FU <: Function = ∂²xᶜᵃᵃ(i, j, k, grid, F, args...) + ∂²yᵃᶜᵃ(i, j, k, grid, F, args...)
@inline ∇²hᶠᶜᵃ(i, j, k, grid, F::FU, args...) where FU <: Function = ∂²xᶠᵃᵃ(i, j, k, grid, F, args...) + ∂²yᵃᶜᵃ(i, j, k, grid, F, args...)
@inline ∇²hᶜᶠᵃ(i, j, k, grid, F::FU, args...) where FU <: Function = ∂²xᶜᵃᵃ(i, j, k, grid, F, args...) + ∂²yᵃᶠᵃ(i, j, k, grid, F, args...)

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

@inline ∇⁴hᶜᶜᵃ(i, j, k, grid, c::AbstractArray) = ∇²hᶜᶜᵃ(i, j, k, grid, ∇²hᶜᶜᵃ, c)
@inline ∇⁴hᶠᶜᵃ(i, j, k, grid, c::AbstractArray) = ∇²hᶠᶜᵃ(i, j, k, grid, ∇²hᶠᶜᵃ, c)
@inline ∇⁴hᶜᶠᵃ(i, j, k, grid, c::AbstractArray) = ∇²hᶜᶠᵃ(i, j, k, grid, ∇²hᶜᶠᵃ, c)
