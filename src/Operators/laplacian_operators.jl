#####
##### Horizontal Laplacians
#####

@inline function ∇²hᶜᶜᶜ(i, j, k, grid, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, ∂xᵣᶠᶜᶜ, c) +
                                    δyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, ∂yᵣᶜᶠᶜ, c))
end

@inline function ∇²hᶠᶜᶜ(i, j, k, grid, u)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δxᶠᶜᶜ(i, j, k, grid, Ax_qᶜᶜᶜ, ∂xᵣᶜᶜᶜ, u) +
                                    δyᶠᶜᶜ(i, j, k, grid, Ay_qᶠᶠᶜ, ∂yᵣᶠᶠᶜ, u))
end

@inline function ∇²hᶜᶠᶜ(i, j, k, grid, v)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᶠᶜ(i, j, k, grid, Ax_qᶠᶠᶜ, ∂xᵣᶠᶠᶜ, v) +
                                    δyᶜᶠᶜ(i, j, k, grid, Ay_qᶜᶜᶜ, ∂yᵣᶜᶜᶜ, v))
end

"""
$(TYPEDSIGNATURES)

Calculate the Laplacian of ``c`` via

```julia
1/V * [δxᶜᵃᵃ(Ax * ∂xᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * ∂yᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * ∂zᵃᵃᶠ(c))]
```

which ends up at the location `ccc`.
"""
@inline function ∇²ᶜᶜᶜ(i, j, k, grid, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᶜᶜ(i, j, k, grid, Ax_qᶠᶜᶜ, ∂xᵣᶠᶜᶜ, c) +
                                    δyᶜᶜᶜ(i, j, k, grid, Ay_qᶜᶠᶜ, ∂yᵣᶜᶠᶜ, c) +
                                    δzᶜᶜᶜ(i, j, k, grid, Az_qᶜᶜᶠ,  ∂zᶜᶜᶠ, c))
end

@inline function ∇²ᶠᶜᶜ(i, j, k, grid, u)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (δxᶠᶜᶜ(i, j, k, grid, Ax_qᶜᶜᶜ, ∂xᵣᶜᶜᶜ, u) +
                                    δyᶠᶜᶜ(i, j, k, grid, Ay_qᶠᶠᶜ, ∂yᵣᶠᶠᶜ, u) +
                                    δzᶠᶜᶜ(i, j, k, grid, Az_qᶠᶜᶠ,  ∂zᶠᶜᶠ, u))
end

@inline function ∇²ᶜᶠᶜ(i, j, k, grid, v)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (δxᶜᶠᶜ(i, j, k, grid, Ax_qᶠᶠᶜ, ∂xᵣᶠᶠᶜ, v) +
                                    δyᶜᶠᶜ(i, j, k, grid, Ay_qᶜᶜᶜ, ∂yᵣᶜᶜᶜ, v) +
                                    δzᶜᶠᶜ(i, j, k, grid, Az_qᶜᶠᶠ,  ∂zᶜᶠᶠ, v))
end

@inline function ∇²ᶜᶜᶠ(i, j, k, grid, w)
    return V⁻¹ᶜᶜᶠ(i, j, k, grid) * (δxᶜᶜᶠ(i, j, k, grid, Ax_qᶠᶜᶠ, ∂xᵣᶠᶜᶠ, w) +
                                    δyᶜᶜᶠ(i, j, k, grid, Ay_qᶜᶠᶠ, ∂yᵣᶜᶠᶠ, w) +
                                    δzᶜᶜᶠ(i, j, k, grid, Az_qᶜᶜᶜ,  ∂zᶜᶜᶜ, w))
end
