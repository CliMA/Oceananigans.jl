# Reciprocal metrics operators, at the moment equal to the inverse of the metric operators
for L1 in (:ᶜ, :ᶠ), L2 in (:ᶜ, :ᶠ), L3 in (:ᶜ, :ᶠ)
    for metric in (:Δ, :A)
        for dir in (:x, :y, :z)
            rcp_metric = Symbol(metric, dir, L1, L2, L3, :⁻¹)
            metric = Symbol(metric, dir, L1, L2, L3)
            @eval @inline $rcp_metric(i, j, k, grid) = 1 / $metric(i, j, k, grid)
        end
    end

    rcp_volume = Symbol(:V, L1, L2, L3, :⁻¹)
    volume = Symbol(:V, L1, L2, L3)
    @eval @inline $rcp_volume(i, j, k, grid) = 1 / $volume(i, j, k, grid)
end

