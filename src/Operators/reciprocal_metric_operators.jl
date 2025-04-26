# Reciprocal metrics operators; equal to the inverse of the metric operators
for L1 in (:ᶜ, :ᶠ, :ᵃ), L2 in (:ᶜ, :ᶠ, :ᵃ), L3 in (:ᶜ, :ᶠ, :ᵃ)
    for dir in (:x, :y)
        rcp_spacing = Symbol(:Δ, dir, :⁻¹, L1, L2, L3)
        spacing = Symbol(:Δ, dir, L1, L2, L3)
        @eval @inline $rcp_metric(i, j, k, grid) = 1 / $metric(i, j, k, grid)
    end
    
    for dir in (:x, :y, :z)
        rcp_area = Symbol(:A, dir, :⁻¹, L1, L2, L3)
        area = Symbol(:A, dir, L1, L2, L3)
        @eval @inline $rcp_metric(i, j, k, grid) = 1 / $metric(i, j, k, grid)
    end

    rcp_volume = Symbol(:V⁻¹, L1, L2, L3)
    volume = Symbol(:V, L1, L2, L3)
    @eval @inline $rcp_volume(i, j, k, grid) = 1 / $volume(i, j, k, grid)
end

@inline Δr⁻¹ᵃᵃᶜ(i, j, k, grid) = getspacing(k, grid.z.Δ⁻¹ᵃᵃᶜ)
@inline Δr⁻¹ᵃᵃᶠ(i, j, k, grid) = getspacing(k, grid.z.Δ⁻¹ᵃᵃᶠ)

@inline Δz⁻¹ᵃᵃᶜ(i, j, k, grid) = getspacing(k, grid.z.Δ⁻¹ᵃᵃᶜ)
@inline Δz⁻¹ᵃᵃᶠ(i, j, k, grid) = getspacing(k, grid.z.Δ⁻¹ᵃᵃᶠ)

for L1 in (:ᶜ, :ᶠ), L2 in (:ᶜ, :ᶠ)
    Δzᵃᵃˡ = Symbol(:Δz, :ᵃ, :ᵃ, L1)
    Δrᵃᵃˡ = Symbol(:Δr, :ᵃ, :ᵃ, L1)
    Δzˡᵃˡ = Symbol(:Δz, L2, :ᵃ, L1)
    Δrˡᵃˡ = Symbol(:Δr, L2, :ᵃ, L1)
    Δzᵃˡˡ = Symbol(:Δz, :ᵃ, L2, L1)
    Δrᵃˡˡ = Symbol(:Δr, :ᵃ, L2, L1)

    @eval @inline $Δzˡᵃˡ(i, j, k, grid) = $Δzᵃᵃˡ(i, j, k, grid)
    @eval @inline $Δzᵃˡˡ(i, j, k, grid) = $Δzᵃᵃˡ(i, j, k, grid)
    @eval @inline $Δrˡᵃˡ(i, j, k, grid) = $Δrᵃᵃˡ(i, j, k, grid)
    @eval @inline $Δrᵃˡˡ(i, j, k, grid) = $Δrᵃᵃˡ(i, j, k, grid)

    for L3 in (:ᶜ, :ᶠ)
        Δzˡˡˡ = Symbol(:Δz, L2, L3, L1)
        Δrˡˡˡ = Symbol(:Δr, L2, L3, L1)

        @eval @inline $Δzˡˡˡ(i, j, k, grid) = $Δzˡᵃˡ(i, j, k, grid)
        @eval @inline $Δrˡˡˡ(i, j, k, grid) = $Δrˡᵃˡ(i, j, k, grid)
    end
end