# Reciprocal metrics operators; equal to the inverse of the metric operators
for L1 in (:ᶜ, :ᶠ, :ᵃ), L2 in (:ᶜ, :ᶠ, :ᵃ), L3 in (:ᶜ, :ᶠ, :ᵃ)
    for func in (:Δ, :A)
        for dir in (:x, :y, :λ, :φ, :z, :r)
            rcp_metric = Symbol(func, dir, :⁻¹, L1, L2, L3)
            metric = Symbol(func, dir, L1, L2, L3)
            @eval @inline $rcp_metric(i, j, k, grid) = 1 / $metric(i, j, k, grid)
        end
    end

    rcp_volume = Symbol(:V⁻¹, L1, L2, L3)
    volume = Symbol(:V, L1, L2, L3)
    @eval @inline $rcp_volume(i, j, k, grid) = 1 / $volume(i, j, k, grid)
end

