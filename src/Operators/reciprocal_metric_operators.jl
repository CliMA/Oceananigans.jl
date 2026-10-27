# Reciprocal metric operators, equal to the inverse of the metric operators and reduced to lower-dimensional
# reciprocals exactly as the metrics are.

#####
##### One-dimensional reciprocal spacings
#####

@inline Δr⁻¹ᵃᵃᶜ(i, j, k, grid) = getspacing(k, grid.z.Δ⁻¹ᵃᵃᶜ)
@inline Δr⁻¹ᵃᵃᶠ(i, j, k, grid) = getspacing(k, grid.z.Δ⁻¹ᵃᵃᶠ)
@inline Δz⁻¹ᵃᵃᶜ(i, j, k, grid) = getspacing(k, grid.z.Δ⁻¹ᵃᵃᶜ)
@inline Δz⁻¹ᵃᵃᶠ(i, j, k, grid) = getspacing(k, grid.z.Δ⁻¹ᵃᵃᶠ)

export Δr⁻¹ᵃᵃᶜ, Δr⁻¹ᵃᵃᶠ, Δz⁻¹ᵃᵃᶜ, Δz⁻¹ᵃᵃᶠ

for L in (:ᶜ, :ᶠ)
    Δx⁻¹ˡᵃᵃ = Symbol(:Δx⁻¹, L, :ᵃ, :ᵃ)
    Δy⁻¹ᵃˡᵃ = Symbol(:Δy⁻¹, :ᵃ, L, :ᵃ)
    Δxˡᵃᵃ   = Symbol(:Δx, L, :ᵃ, :ᵃ)
    Δyᵃˡᵃ   = Symbol(:Δy, :ᵃ, L, :ᵃ)

    @eval begin
        @inline $Δx⁻¹ˡᵃᵃ(i, j, k, grid) = 1 / $Δxˡᵃᵃ(i, j, k, grid)
        @inline $Δy⁻¹ᵃˡᵃ(i, j, k, grid) = 1 / $Δyᵃˡᵃ(i, j, k, grid)

        @inline $Δx⁻¹ˡᵃᵃ(i, j, k, grid::RG)  = @inbounds grid.$Δx⁻¹ˡᵃᵃ[i]
        @inline $Δy⁻¹ᵃˡᵃ(i, j, k, grid::RG)  = @inbounds grid.$Δy⁻¹ᵃˡᵃ[j]
        @inline $Δx⁻¹ˡᵃᵃ(i, j, k, grid::RGX) = grid.$Δx⁻¹ˡᵃᵃ
        @inline $Δy⁻¹ᵃˡᵃ(i, j, k, grid::RGY) = grid.$Δy⁻¹ᵃˡᵃ

        export $Δx⁻¹ˡᵃᵃ, $Δy⁻¹ᵃˡᵃ
    end
end

@inline Δy⁻¹ᵃᶜᵃ(i, j, k, grid::LLG)   = @inbounds grid.Δy⁻¹ᶠᶜᵃ[j]
@inline Δy⁻¹ᵃᶠᵃ(i, j, k, grid::LLG)   = @inbounds grid.Δy⁻¹ᶜᶠᵃ[j]
@inline Δy⁻¹ᵃᶜᵃ(i, j, k, grid::LLGY)  = grid.Δy⁻¹ᶠᶜᵃ
@inline Δy⁻¹ᵃᶠᵃ(i, j, k, grid::LLGY)  = grid.Δy⁻¹ᶜᶠᵃ
@inline Δy⁻¹ᵃᶜᵃ(i, j, k, grid::LLGF)  = 1 / Δyᵃᶜᵃ(i, j, k, grid)
@inline Δy⁻¹ᵃᶠᵃ(i, j, k, grid::LLGF)  = 1 / Δyᵃᶠᵃ(i, j, k, grid)
@inline Δy⁻¹ᵃᶜᵃ(i, j, k, grid::LLGFY) = 1 / Δyᵃᶜᵃ(i, j, k, grid)
@inline Δy⁻¹ᵃᶠᵃ(i, j, k, grid::LLGFY) = 1 / Δyᵃᶠᵃ(i, j, k, grid)

#####
##### Two-dimensional reciprocal spacings with an unspecified horizontal location
#####

for L1 in (:ᶜ, :ᶠ), L2 in (:ᶜ, :ᶠ)
    Δx⁻¹ˡᵃˡ = Symbol(:Δx⁻¹, L1, :ᵃ, L2)
    Δy⁻¹ᵃˡˡ = Symbol(:Δy⁻¹, :ᵃ, L1, L2)
    Δx⁻¹ˡᵃᵃ = Symbol(:Δx⁻¹, L1, :ᵃ, :ᵃ)
    Δy⁻¹ᵃˡᵃ = Symbol(:Δy⁻¹, :ᵃ, L1, :ᵃ)

    @eval begin
        @inline $Δx⁻¹ˡᵃˡ(i, j, k, grid) = $Δx⁻¹ˡᵃᵃ(i, j, k, grid)
        @inline $Δy⁻¹ᵃˡˡ(i, j, k, grid) = $Δy⁻¹ᵃˡᵃ(i, j, k, grid)

        export $Δx⁻¹ˡᵃˡ, $Δy⁻¹ᵃˡˡ
    end

    for func in (:Δz⁻¹, :Δr⁻¹)
        rcp_xz = Symbol(func, L2, :ᵃ, L1)
        rcp_yz = Symbol(func, :ᵃ, L2, L1)
        rcp_1D = Symbol(func, :ᵃ, :ᵃ, L1)

        @eval begin
            @inline $rcp_xz(i, j, k, grid) = $rcp_1D(i, j, k, grid)
            @inline $rcp_yz(i, j, k, grid) = $rcp_1D(i, j, k, grid)

            export $rcp_xz, $rcp_yz
        end
    end
end

#####
##### Horizontal reciprocals, and their three-dimensional extensions
#####

for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ)
    Δx⁻¹ˡᵃᵃ = Symbol(:Δx⁻¹, LX, :ᵃ, :ᵃ)
    Δy⁻¹ᵃˡᵃ = Symbol(:Δy⁻¹, :ᵃ, LY, :ᵃ)
    Δx⁻¹ˡˡᵃ = Symbol(:Δx⁻¹, LX, LY, :ᵃ)
    Δy⁻¹ˡˡᵃ = Symbol(:Δy⁻¹, LX, LY, :ᵃ)
    Az⁻¹ˡˡᵃ = Symbol(:Az⁻¹, LX, LY, :ᵃ)
    Δxˡˡᵃ   = Symbol(:Δx, LX, LY, :ᵃ)
    Azˡˡᵃ   = Symbol(:Az, LX, LY, :ᵃ)

    @eval begin
        @inline $Δx⁻¹ˡˡᵃ(i, j, k, grid) = $Δx⁻¹ˡᵃᵃ(i, j, k, grid)
        @inline $Δy⁻¹ˡˡᵃ(i, j, k, grid) = $Δy⁻¹ᵃˡᵃ(i, j, k, grid)
        @inline $Az⁻¹ˡˡᵃ(i, j, k, grid) = $Δx⁻¹ˡˡᵃ(i, j, k, grid) * $Δy⁻¹ˡˡᵃ(i, j, k, grid)

        @inline $Δx⁻¹ˡˡᵃ(i, j, k, grid::OSSG) = @inbounds grid.$Δx⁻¹ˡˡᵃ[i, j]
        @inline $Δy⁻¹ˡˡᵃ(i, j, k, grid::OSSG) = @inbounds grid.$Δy⁻¹ˡˡᵃ[i, j]
        @inline $Az⁻¹ˡˡᵃ(i, j, k, grid::OSSG) = @inbounds grid.$Az⁻¹ˡˡᵃ[i, j]

        @inline $Δx⁻¹ˡˡᵃ(i, j, k, grid::LLG)   = @inbounds grid.$Δx⁻¹ˡˡᵃ[i, j]
        @inline $Δx⁻¹ˡˡᵃ(i, j, k, grid::LLGX)  = @inbounds grid.$Δx⁻¹ˡˡᵃ[j]
        @inline $Δx⁻¹ˡˡᵃ(i, j, k, grid::LLGF)  = 1 / $Δxˡˡᵃ(i, j, k, grid)
        @inline $Δx⁻¹ˡˡᵃ(i, j, k, grid::LLGFX) = 1 / $Δxˡˡᵃ(i, j, k, grid)

        @inline $Az⁻¹ˡˡᵃ(i, j, k, grid::LLG)   = @inbounds grid.$Az⁻¹ˡˡᵃ[i, j]
        @inline $Az⁻¹ˡˡᵃ(i, j, k, grid::LLGX)  = @inbounds grid.$Az⁻¹ˡˡᵃ[j]
        @inline $Az⁻¹ˡˡᵃ(i, j, k, grid::LLGF)  = 1 / $Azˡˡᵃ(i, j, k, grid)
        @inline $Az⁻¹ˡˡᵃ(i, j, k, grid::LLGFX) = 1 / $Azˡˡᵃ(i, j, k, grid)

        export $Δx⁻¹ˡˡᵃ, $Δy⁻¹ˡˡᵃ, $Az⁻¹ˡˡᵃ
    end

    for LZ in (:ᶜ, :ᶠ)
        Δx⁻¹ˡˡˡ = Symbol(:Δx⁻¹, LX, LY, LZ)
        Δy⁻¹ˡˡˡ = Symbol(:Δy⁻¹, LX, LY, LZ)
        Δz⁻¹ˡˡˡ = Symbol(:Δz⁻¹, LX, LY, LZ)
        Δr⁻¹ˡˡˡ = Symbol(:Δr⁻¹, LX, LY, LZ)
        Ax⁻¹ˡˡˡ = Symbol(:Ax⁻¹, LX, LY, LZ)
        Ay⁻¹ˡˡˡ = Symbol(:Ay⁻¹, LX, LY, LZ)
        Az⁻¹ˡˡˡ = Symbol(:Az⁻¹, LX, LY, LZ)
        V⁻¹ˡˡˡ  = Symbol(:V⁻¹,  LX, LY, LZ)

        Δz⁻¹ˡᵃˡ = Symbol(:Δz⁻¹, LX, :ᵃ, LZ)
        Δr⁻¹ˡᵃˡ = Symbol(:Δr⁻¹, LX, :ᵃ, LZ)

        @eval begin
            @inline $Δx⁻¹ˡˡˡ(i, j, k, grid) = $Δx⁻¹ˡˡᵃ(i, j, k, grid)
            @inline $Δy⁻¹ˡˡˡ(i, j, k, grid) = $Δy⁻¹ˡˡᵃ(i, j, k, grid)
            @inline $Δz⁻¹ˡˡˡ(i, j, k, grid) = $Δz⁻¹ˡᵃˡ(i, j, k, grid)
            @inline $Δr⁻¹ˡˡˡ(i, j, k, grid) = $Δr⁻¹ˡᵃˡ(i, j, k, grid)
            @inline $Az⁻¹ˡˡˡ(i, j, k, grid) = $Az⁻¹ˡˡᵃ(i, j, k, grid)

            @inline $Ax⁻¹ˡˡˡ(i, j, k, grid) = $Δy⁻¹ˡˡˡ(i, j, k, grid) * $Δz⁻¹ˡˡˡ(i, j, k, grid)
            @inline $Ay⁻¹ˡˡˡ(i, j, k, grid) = $Δx⁻¹ˡˡˡ(i, j, k, grid) * $Δz⁻¹ˡˡˡ(i, j, k, grid)
            @inline $V⁻¹ˡˡˡ(i, j, k, grid)  = $Az⁻¹ˡˡˡ(i, j, k, grid) * $Δz⁻¹ˡˡˡ(i, j, k, grid)

            export $Δx⁻¹ˡˡˡ, $Δy⁻¹ˡˡˡ, $Δz⁻¹ˡˡˡ, $Δr⁻¹ˡˡˡ, $Ax⁻¹ˡˡˡ, $Ay⁻¹ˡˡˡ, $Az⁻¹ˡˡˡ, $V⁻¹ˡˡˡ
        end
    end
end

#####
##### Reciprocals at a location given by its instances, e.g. V⁻¹(i, j, k, grid, Center(), Center(), Face()) = V⁻¹ᶜᶜᶠ(i, j, k, grid)
#####

for ℓ1 in (:ᶜ, :ᶠ), ℓ2 in (:ᶜ, :ᶠ, :ᵃ), ℓ3 in (:ᶜ, :ᶠ, :ᵃ)
    L1 = location_from_superscript(ℓ1)
    L2 = location_from_superscript(ℓ2)
    L3 = location_from_superscript(ℓ3)

    rcp_spacing_x = Symbol(:Δx⁻¹, ℓ1, ℓ2, ℓ3)
    rcp_spacing_y = Symbol(:Δy⁻¹, ℓ2, ℓ1, ℓ3)
    rcp_spacing_z = Symbol(:Δz⁻¹, ℓ2, ℓ3, ℓ1)

    @eval begin
        @inline Δx⁻¹(i, j, k, grid, ::$L1, ::$L2, ::$L3) = $rcp_spacing_x(i, j, k, grid)
        @inline Δy⁻¹(i, j, k, grid, ::$L2, ::$L1, ::$L3) = $rcp_spacing_y(i, j, k, grid)
        @inline Δz⁻¹(i, j, k, grid, ::$L2, ::$L3, ::$L1) = $rcp_spacing_z(i, j, k, grid)
    end
end

for ℓ1 in (:ᶜ, :ᶠ), ℓ2 in (:ᶜ, :ᶠ), ℓ3 in (:ᶜ, :ᶠ)
    L1 = location_from_superscript(ℓ1)
    L2 = location_from_superscript(ℓ2)
    L3 = location_from_superscript(ℓ3)

    rcp_volume = Symbol(:V⁻¹, ℓ1, ℓ2, ℓ3)

    @eval @inline V⁻¹(i, j, k, grid, ::$L1, ::$L2, ::$L3) = $rcp_volume(i, j, k, grid)
end
