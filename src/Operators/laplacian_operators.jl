#####
##### Horizontal Laplacians
#####

@inline function ∇²hᶜᶜᶜ(i, j, k, grid, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, ∂xᵣᶠᶜᶜ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, ∂yᵣᶜᶠᶜ, c))
end

@inline function ∇²hᶠᶜᶜ(i, j, k, grid, u)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_qᶜᶜᶜ, ∂xᵣᶜᶜᶜ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_qᶠᶠᶜ, ∂yᵣᶠᶠᶜ, u))
end

@inline function ∇²hᶜᶠᶜ(i, j, k, grid, v)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶠᶜ, ∂xᵣᶠᶠᶜ, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, Ay_qᶜᶜᶜ, ∂yᵣᶜᶜᶜ, v))
end

@inline function ∇²hᶜᶜᶠ(i, j, k, grid, w)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶠᶜ, ∂xᵣᶠᶠᶜ, w) +
                                    δyᵃᶠᵃ(i, j, k, grid, Ay_qᶜᶜᶜ, ∂yᵣᶜᶜᶜ, w))
end

"""
    ∇²ᶜᶜᶜ(i, j, k, grid, c)

Calculate the Laplacian of ``c`` via

```julia
1/V * [δxᶜᵃᵃ(Ax * ∂xᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * ∂yᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * ∂zᵃᵃᶠ(c))]
```

which ends up at the location `ccc`.
"""
@inline function ∇²ᶜᶜᶜ(i, j, k, grid, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, ∂xᵣᶠᶜᶜ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, ∂yᵣᶜᶠᶜ, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ,  ∂zᶜᶜᶠ, c))
end

@inline function ∇²ᶠᶜᶜ(i, j, k, grid, u)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_qᶜᶜᶜ, ∂xᵣᶜᶜᶜ, u) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_qᶠᶠᶜ, ∂yᵣᶠᶠᶜ, u) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_qᶠᶜᶠ,  ∂zᶠᶜᶠ, u))
end

@inline function ∇²ᶜᶠᶜ(i, j, k, grid, v)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶠᶜ, ∂xᵣᶠᶠᶜ, v) +
                                    δyᵃᶠᵃ(i, j, k, grid, Ay_qᶜᶜᶜ, ∂yᵣᶜᶜᶜ, v) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶠᶠ,  ∂zᶜᶠᶠ, v))
end

@inline function ∇²ᶜᶜᶠ(i, j, k, grid, w)
    return V⁻¹ᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶠ, ∂xᵣᶠᶜᶠ, w) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶠ, ∂yᵣᶜᶠᶠ, w) +
                                    δzᵃᵃᶠ(i, j, k, grid, Az_qᶜᶜᶜ,  ∂zᶜᶜᶜ, w))
end
