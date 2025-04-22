# Reciprocal metrics operators, at the moment equal to the inverse of the metric operators
for L1 in (:ᶜ, :ᶠ), L2 in (:ᶜ, :ᶠ), L3 in (:ᶜ, :ᶠ)
    for func in (:Δ, :A)
        for dir in (:x, :y, :z)
            rcp_metric = Symbol(func, dir, :⁻¹, L1, L2, L3)
            metric = Symbol(func, dir, L1, L2, L3)
            @eval @inline $rcp_metric(i, j, k, grid) = 1 / $metric(i, j, k, grid)
        end
    end

    rcp_volume = Symbol(:V⁻¹, L1, L2, L3)
    volume = Symbol(:V, L1, L2, L3)
    @eval @inline $rcp_volume(i, j, k, grid) = 1 / $volume(i, j, k, grid)
end

