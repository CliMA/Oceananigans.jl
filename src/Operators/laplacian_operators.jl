#####
##### Horizontal Laplacians
#####

@inline function ∇²hᶜᶜᶜ(i, j, k, grid, c)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᶜᵃ(i, j, k, grid, Ax_∂xᶠᶜᶜ, c) +
                                    δyᶜᶜᵃ(i, j, k, grid, Ay_∂yᶜᶠᶜ, c))
end

@inline function ∇²hᶠᶜᶜ(i, j, k, grid, u)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᶜᵃ(i, j, k, grid, Ax_∂xᶜᶜᶜ, u) +
                                    δyᶠᶜᵃ(i, j, k, grid, Ay_∂yᶠᶠᶜ, u))
end

@inline function ∇²hᶜᶠᶜ(i, j, k, grid, v)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᶠᵃ(i, j, k, grid, Ax_∂xᶠᶠᶜ, v) +
                                    δyᶜᶠᵃ(i, j, k, grid, Ay_∂yᶜᶜᶜ, v))
end
