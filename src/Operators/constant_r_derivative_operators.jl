#####
##### Derivative operators at constant r (computational coordinate)
#####
##### These operators compute simple finite differences divided by grid spacing,
##### without any chain-rule corrections for mutable vertical discretizations.
##### They are appropriate for quantities that live on the free surface (like η)
##### where the "derivative at constant r" is the physically meaningful derivative.
#####
##### For fields with vertical structure, use the standard ∂x, ∂y operators which
##### include chain-rule corrections to compute derivatives at constant z.
#####

# Horizontal derivatives at constant r
for ℓ1 in (:ᶜ, :ᶠ), ℓ2 in (:ᶜ, :ᶠ, :ᵃ), ℓ3 in (:ᶜ, :ᶠ, :ᵃ)
    ∂xᵣ    = Symbol(:∂xᵣ, ℓ1, ℓ2, ℓ3)
    δx     = Symbol(:δx, ℓ1, ℓ2, ℓ3)
    rcp_Δx = Symbol(:Δx⁻¹, ℓ1, ℓ2, ℓ3)

    ∂yᵣ    = Symbol(:∂yᵣ, ℓ2, ℓ1, ℓ3)
    δy     = Symbol(:δy, ℓ2, ℓ1, ℓ3)
    rcp_Δy = Symbol(:Δy⁻¹, ℓ2, ℓ1, ℓ3)

    @eval begin
        @inline $∂xᵣ(i, j, k, grid, c) = $δx(i, j, k, grid, c) * $rcp_Δx(i, j, k, grid)
        @inline $∂yᵣ(i, j, k, grid, c) = $δy(i, j, k, grid, c) * $rcp_Δy(i, j, k, grid)

        @inline $∂xᵣ(i, j, k, grid, f::Function, args...) = $δx(i, j, k, grid, f, args...) * $rcp_Δx(i, j, k, grid)
        @inline $∂yᵣ(i, j, k, grid, f::Function, args...) = $δy(i, j, k, grid, f, args...) * $rcp_Δy(i, j, k, grid)

        @inline $∂xᵣ(i, j, k, grid, c::Number) = zero(grid)
        @inline $∂yᵣ(i, j, k, grid, c::Number) = zero(grid)

        export $∂xᵣ, $∂yᵣ
    end
end

#####
##### Topology-aware derivatives at constant r for free surface calculations
#####
##### These use the topology-aware difference operators (δxT, δyT) that correctly
##### handle boundary conditions, combined with simple spacing division.
#####

# ∂x_rTᶠᶜᶠ: topology-aware x-derivative at constant r (for free surface η)
@inline ∂xᵣTᶠᶜᶠ(i, j, k, grid, ϕ, args...) = conditional_δ(Val(1), Face(), Center(), Face(), i, j, k, grid, δxTᶠᵃᵃ, ϕ, args...) * Δx⁻¹ᶠᶜᶠ(i, j, k, grid)

# ∂y_rTᶜᶠᶠ: topology-aware y-derivative at constant r (for free surface η)
@inline ∂yᵣTᶜᶠᶠ(i, j, k, grid, ϕ, args...) = conditional_δ(Val(2), Center(), Face(), Face(), i, j, k, grid, δyTᵃᶠᵃ, ϕ, args...) * Δy⁻¹ᶜᶠᶠ(i, j, k, grid)

# The topology-aware derivatives at constant r are the topology-aware derivatives of the barotropic mode
const ∂xTᶠᶜᶠ = ∂xᵣTᶠᶜᶠ
const ∂yTᶜᶠᶠ = ∂yᵣTᶜᶠᶠ

export ∂xᵣTᶠᶜᶠ, ∂yᵣTᶜᶠᶠ
