# Reciprocal metrics operators; equal to the inverse of the metric operators
for L1 in (:ᶜ, :ᶠ), L2 in (:ᶜ, :ᶠ, :ᵃ), L3 in (:ᶜ, :ᶠ, :ᵃ)
    for func in (:Δ, :A)
        rcp_x = Symbol(func, :x⁻¹, L1, L2, L3)
        rcp_y = Symbol(func, :y⁻¹, L2, L1, L3)
        rcp_z = Symbol(func, :z⁻¹, L2, L3, L1)

        metric_x = Symbol(func, :x, L1, L2, L3)
        metric_y = Symbol(func, :y, L2, L1, L3)
        metric_z = Symbol(func, :z, L2, L3, L1)

        @eval begin
            @inline $rcp_x(i, j, k, grid) = 1 / $metric_x(i, j, k, grid)
            @inline $rcp_y(i, j, k, grid) = 1 / $metric_y(i, j, k, grid)
            @inline $rcp_z(i, j, k, grid) = 1 / $metric_z(i, j, k, grid)

            export $rcp_x, $rcp_y, $rcp_z
        end
    end
end

for L1 in (:ᶜ, :ᶠ), L2 in (:ᶜ, :ᶠ), L3 in (:ᶜ, :ᶠ)
    rcp_volume = Symbol(:V⁻¹, L1, L2, L3)
    volume     = Symbol(:V,   L1, L2, L3)
    @eval begin
        @inline $rcp_volume(i, j, k, grid) = 1 / $volume(i, j, k, grid)

        export $rcp_volume
    end
end
