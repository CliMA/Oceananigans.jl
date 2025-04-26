# Reciprocal metrics operators; equal to the inverse of the metric operators
for L1 in (:ᶜ, :ᶠ, :ᵃ), L2 in (:ᶜ, :ᶠ, :ᵃ), L3 in (:ᶜ, :ᶠ, :ᵃ)
    for dir in (:x, :y)
        rcp_spacing = Symbol(:Δ, dir, :⁻¹, L1, L2, L3)
        spacing = Symbol(:Δ, dir, L1, L2, L3)
        @eval @inline $rcp_spacing(i, j, k, grid) = 1 / $spacing(i, j, k, grid)
    end
    
    for dir in (:x, :y, :z)
        rcp_area = Symbol(:A, dir, :⁻¹, L1, L2, L3)
        area = Symbol(:A, dir, L1, L2, L3)
        @eval @inline $rcp_area(i, j, k, grid) = 1 / $area(i, j, k, grid)
    end

    rcp_vol = Symbol(:V⁻¹, L1, L2, L3)
    vol = Symbol(:V, L1, L2, L3)
    @eval @inline $rcp_vol(i, j, k, grid) = 1 / $vol(i, j, k, grid)
end

@inline Δr⁻¹ᵃᵃᶜ(i, j, k, grid) = getspacing(k, grid.z.Δ⁻¹ᵃᵃᶜ)
@inline Δr⁻¹ᵃᵃᶠ(i, j, k, grid) = getspacing(k, grid.z.Δ⁻¹ᵃᵃᶠ)

@inline Δz⁻¹ᵃᵃᶜ(i, j, k, grid) = getspacing(k, grid.z.Δ⁻¹ᵃᵃᶜ)
@inline Δz⁻¹ᵃᵃᶠ(i, j, k, grid) = getspacing(k, grid.z.Δ⁻¹ᵃᵃᶠ)

for L1 in (:ᶜ, :ᶠ), L2 in (:ᶜ, :ᶠ)
    Δz⁻¹ᵃᵃˡ = Symbol(:Δz⁻¹, :ᵃ, :ᵃ, L1)
    Δr⁻¹ᵃᵃˡ = Symbol(:Δr⁻¹, :ᵃ, :ᵃ, L1)
    Δz⁻¹ˡᵃˡ = Symbol(:Δz⁻¹, L2, :ᵃ, L1)
    Δr⁻¹ˡᵃˡ = Symbol(:Δr⁻¹, L2, :ᵃ, L1)
    Δz⁻¹ᵃˡˡ = Symbol(:Δz⁻¹, :ᵃ, L2, L1)
    Δr⁻¹ᵃˡˡ = Symbol(:Δr⁻¹, :ᵃ, L2, L1)

    @eval @inline $Δr⁻¹ᵃᵃˡ(i, j, k, grid) = $Δz⁻¹ᵃᵃˡ(i, j, k, grid)
    @eval @inline $Δz⁻¹ˡᵃˡ(i, j, k, grid) = $Δz⁻¹ᵃᵃˡ(i, j, k, grid)
    @eval @inline $Δr⁻¹ˡᵃˡ(i, j, k, grid) = $Δr⁻¹ᵃᵃˡ(i, j, k, grid)
    @eval @inline $Δr⁻¹ᵃˡˡ(i, j, k, grid) = $Δr⁻¹ᵃᵃˡ(i, j, k, grid)

    for L3 in (:ᶜ, :ᶠ)
        Δz⁻¹ˡˡˡ = Symbol(:Δz⁻¹, L2, L3, L1)
        Δr⁻¹ˡˡˡ = Symbol(:Δr⁻¹, L2, L3, L1)

        @eval @inline $Δz⁻¹ˡˡˡ(i, j, k, grid) = $Δz⁻¹ˡᵃˡ(i, j, k, grid)
        @eval @inline $Δr⁻¹ˡˡˡ(i, j, k, grid) = $Δr⁻¹ˡᵃˡ(i, j, k, grid)
    end
end