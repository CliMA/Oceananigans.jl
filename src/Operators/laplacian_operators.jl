using Oceananigans.Grids: solid_node

#####
##### Horizontal Laplacians
#####

@inline function ∇²hᶜᶜᶜ(i, j, k, grid, c)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᶜᶜ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᶜᶠᶜ, c))
end

# we cancel Laplacians which are evaluated on boundary faces (these are used only in the biharmonic operator)
∇²hᶠᶜᶜ(i, j, k, grid, args...) = ifelse(solid_node(i, j, k, grid) | solid_node(i-1, j, k, grid), zero(eltype(grid)), _∇²hᶠᶜᶜ(i, j, k, grid, args...))
∇²hᶜᶠᶜ(i, j, k, grid, args...) = ifelse(solid_node(i, j, k, grid) | solid_node(i, j-1, k, grid), zero(eltype(grid)), _∇²hᶜᶠᶜ(i, j, k, grid, args...))
∇²hᶜᶜᶠ(i, j, k, grid, args...) = ifelse(solid_node(i, j, k, grid) | solid_node(i, j, k-1, grid), zero(eltype(grid)), _∇²hᶜᶜᶠ(i, j, k, grid, args...))

@inline function _∇²hᶠᶜᶜ(i, j, k, grid, u)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_∂xᶜᶜᶜ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᶠᶠᶜ, u))
end

@inline function _∇²hᶜᶠᶜ(i, j, k, grid, v)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᶠᶜ, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, Ay_∂yᶜᶜᶜ, v))
end

@inline function _∇²hᶜᶜᶠ(i, j, k, grid, w)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᶠᶜ, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, Ay_∂yᶜᶜᶜ, v))
end

"""
    ∇²ᶜᶜᶜ(i, j, k, grid, c)

Calculates the Laplacian of `c` via

```
1/V * [δxᶜᵃᵃ(Ax * ∂xᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * ∂yᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * ∂zᵃᵃᶠ(c))]
```

which will end up at the location `ccc`.
"""
@inline function ∇²ᶜᶜᶜ(i, j, k, grid, c)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᶜᶜ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᶜᶠᶜ, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_∂zᶜᶜᶠ, c))
end

# we cancel Laplacians which are evaluated on boundary faces (these are used only in the biharmonic operator)
∇²ᶠᶜᶜ(i, j, k, grid, args...) = ifelse(solid_node(i, j, k, grid) | solid_node(i-1, j, k, grid), zero(eltype(grid)), _∇²ᶠᶜᶜ(i, j, k, grid, args...))
∇²ᶜᶠᶜ(i, j, k, grid, args...) = ifelse(solid_node(i, j, k, grid) | solid_node(i, j-1, k, grid), zero(eltype(grid)), _∇²ᶜᶠᶜ(i, j, k, grid, args...))
∇²ᶜᶜᶠ(i, j, k, grid, args...) = ifelse(solid_node(i, j, k, grid) | solid_node(i, j, k-1, grid), zero(eltype(grid)), _∇²ᶜᶜᶠ(i, j, k, grid, args...))

@inline function _∇²ᶠᶜᶜ(i, j, k, grid, u)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_∂xᶜᶜᶜ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᶠᶠᶜ, u) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_∂zᶠᶜᶠ, u))
end

@inline function _∇²ᶜᶠᶜ(i, j, k, grid, v)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᶠᶜ, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, Ay_∂yᶜᶜᶜ, v) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_∂zᶜᶠᶠ, v))
end

@inline function _∇²ᶜᶜᶠ(i, j, k, grid, w)
    return 1/Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᶜᶠ, w) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᶜᶠᶠ, w) +
                                    δzᵃᵃᶠ(i, j, k, grid, Az_∂zᶜᶜᶜ, w))
end