#####
##### Horizontal Laplacians
#####

@inline function ∇²hᶜᶜᶜ(i, j, k, grid, c)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᶜᶜ(i, j, k, grid, Ax_∂xᶠᶜᶜ, c) +
                                    δyᶜᶜᶜ(i, j, k, grid, Ay_∂yᶜᶠᶜ, c))
end

@inline function ∇²hᶠᶜᶜ(i, j, k, grid, u)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᶜᶜ(i, j, k, grid, Ax_∂xᶜᶜᶜ, u) +
                                    δyᶠᶜᶜ(i, j, k, grid, Ay_∂yᶠᶠᶜ, u))
end

@inline function ∇²hᶜᶠᶜ(i, j, k, grid, v)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᶠᶜ(i, j, k, grid, Ax_∂xᶠᶠᶜ, v) +
                                    δyᶜᶠᶜ(i, j, k, grid, Ay_∂yᶜᶜᶜ, v))
end

@inline function ∇²hᶜᶜᶠ(i, j, k, grid, w)
    return 1/Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᶜᶠ(i, j, k, grid, Ax_∂xᶠᶜᶠ, w) +
                                    δyᶜᶜᶠ(i, j, k, grid, Ay_∂yᶜᶠᶠ, w))
end

"""
    ∇²ᶜᶜᶜ(i, j, k, grid, c)

Calculates the Laplacian of `c`.
"""
@inline function ∇²ᶜᶜᶜ(i, j, k, grid, c)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᶜᶜ(i, j, k, grid, Ax_∂xᶠᶜᶜ, c) +
                                    δyᶜᶜᶜ(i, j, k, grid, Ay_∂yᶜᶠᶜ, c) +
                                    δzᶜᶜᶜ(i, j, k, grid, Az_∂zᶜᶜᶠ, c))
end

@inline function ∇²ᶠᶜᶜ(i, j, k, grid, u)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᶜᶜ(i, j, k, grid, Ax_∂xᶜᶜᶜ, u) +
                                    δyᶠᶜᶜ(i, j, k, grid, Ay_∂yᶠᶠᶜ, u) +
                                    δzᶠᶜᶜ(i, j, k, grid, Az_∂zᶠᶜᶠ, u))
end

@inline function ∇²ᶜᶠᶜ(i, j, k, grid, v)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᶠᶜ(i, j, k, grid, Ax_∂xᶠᶠᶜ, v) +
                                    δyᶜᶠᶜ(i, j, k, grid, Ay_∂yᶜᶜᶜ, v) +
                                    δzᶜᶠᶜ(i, j, k, grid, Az_∂zᶜᶠᶠ, v))
end

@inline function ∇²ᶜᶜᶠ(i, j, k, grid, w)
    return 1/Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᶜᶠ(i, j, k, grid, Ax_∂xᶠᶜᶠ, w) +
                                    δyᶜᶜᶠ(i, j, k, grid, Ay_∂yᶜᶠᶠ, w) +
                                    δzᶜᶜᶠ(i, j, k, grid, Az_∂zᶜᶜᶜ, w))
end
