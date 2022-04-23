struct VariableBiharmonicDiffusionCoefficient
    time_scale :: Float64
end

@inline function (vbd::VariableBiharmonicDiffusionCoefficient)(i, j, k, grid, lx, ly, lz)
    Δh⁻² = 1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2
    Δh⁴ = 1 / Δh⁻²^2 
    return Δh⁴ / vbd.time_scale
end

