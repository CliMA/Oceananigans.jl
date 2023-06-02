import Oceananigans.Operators: 
    δxᶠᶜᶜ, δxᶠᶜᶠ, δxᶠᶠᶜ, δxᶠᶠᶠ,
    δxᶜᶜᶜ, δxᶜᶜᶠ, δxᶜᶠᶜ, δxᶜᶠᶠ,
    δyᶜᶠᶜ, δyᶜᶠᶠ, δyᶠᶠᶜ, δyᶠᶠᶠ,
    δyᶜᶜᶜ, δyᶜᶜᶠ, δyᶠᶜᶜ, δyᶠᶜᶠ,
    δzᶜᶜᶠ, δzᶜᶠᶠ, δzᶠᶜᶠ, δzᶠᶠᶠ,
    δzᶜᶜᶜ, δzᶜᶠᶜ, δzᶠᶜᶜ, δzᶠᶠᶜ

import Oceananigans.Operators: 
    ℑxᶠᶜᶜ, ℑxᶠᶜᶠ, ℑxᶠᶠᶜ, ℑxᶠᶠᶠ,
    ℑxᶜᶜᶜ, ℑxᶜᶜᶠ, ℑxᶜᶠᶜ, ℑxᶜᶠᶠ,
    ℑyᶜᶠᶜ, ℑyᶜᶠᶠ, ℑyᶠᶠᶜ, ℑyᶠᶠᶠ,
    ℑyᶜᶜᶜ, ℑyᶜᶜᶠ, ℑyᶠᶜᶜ, ℑyᶠᶜᶠ,
    ℑzᶜᶜᶠ, ℑzᶜᶠᶠ, ℑzᶠᶜᶠ, ℑzᶠᶠᶠ,
    ℑzᶜᶜᶜ, ℑzᶜᶠᶜ, ℑzᶠᶜᶜ, ℑzᶠᶠᶜ
    
# Defining Difference operators for the immersed boundaries
@inline conditional_δx_f(LY, LZ, i, j, k, ibg::IBG{FT}, δx, args...) where FT = ifelse(inactive_node(i, j, k, ibg, c, LY, LZ) | inactive_node(i-1, j, k, ibg, c, LY, LZ), zero(FT), δx(i, j, k, ibg.underlying_grid, args...))
@inline conditional_δx_c(LY, LZ, i, j, k, ibg::IBG{FT}, δx, args...) where FT = ifelse(inactive_node(i, j, k, ibg, f, LY, LZ) | inactive_node(i+1, j, k, ibg, f, LY, LZ), zero(FT), δx(i, j, k, ibg.underlying_grid, args...))
@inline conditional_δy_f(LX, LZ, i, j, k, ibg::IBG{FT}, δy, args...) where FT = ifelse(inactive_node(i, j, k, ibg, LX, c, LZ) | inactive_node(i, j-1, k, ibg, LX, c, LZ), zero(FT), δy(i, j, k, ibg.underlying_grid, args...))
@inline conditional_δy_c(LX, LZ, i, j, k, ibg::IBG{FT}, δy, args...) where FT = ifelse(inactive_node(i, j, k, ibg, LX, f, LZ) | inactive_node(i, j+1, k, ibg, LX, f, LZ), zero(FT), δy(i, j, k, ibg.underlying_grid, args...))
@inline conditional_δz_f(LX, LY, i, j, k, ibg::IBG{FT}, δz, args...) where FT = ifelse(inactive_node(i, j, k, ibg, LX, LY, c) | inactive_node(i, j, k-1, ibg, LX, LY, c), zero(FT), δz(i, j, k, ibg.underlying_grid, args...))
@inline conditional_δz_c(LX, LY, i, j, k, ibg::IBG{FT}, δz, args...) where FT = ifelse(inactive_node(i, j, k, ibg, LX, LY, f) | inactive_node(i, j, k+1, ibg, LX, LY, f), zero(FT), δz(i, j, k, ibg.underlying_grid, args...))

@inline ı(i, j, k, grid, f::Function, args...) = f(i, j, k, grid, args...)
@inline ı(i, j, k, grid, ϕ)                    = ϕ[i, j, k]

# Defining Interpolation operators for the immersed boundaries
@inline conditional_ℑx_f(LY, LZ, i, j, k, ibg::IBG{FT}, ℑx, args...) where FT = ifelse(inactive_node(i, j, k, ibg, c, LY, LZ), ı(i-1, j, k, ibg, args...), ifelse(inactive_node(i-1, j, k, ibg, c, LY, LZ), ı(i, j, k, ibg, args...), ℑx(i, j, k, ibg.underlying_grid, args...)))
@inline conditional_ℑx_c(LY, LZ, i, j, k, ibg::IBG{FT}, ℑx, args...) where FT = ifelse(inactive_node(i, j, k, ibg, f, LY, LZ), ı(i+1, j, k, ibg, args...), ifelse(inactive_node(i+1, j, k, ibg, f, LY, LZ), ı(i, j, k, ibg, args...), ℑx(i, j, k, ibg.underlying_grid, args...)))
@inline conditional_ℑy_f(LX, LZ, i, j, k, ibg::IBG{FT}, ℑy, args...) where FT = ifelse(inactive_node(i, j, k, ibg, LX, c, LZ), ı(i, j-1, k, ibg, args...), ifelse(inactive_node(i, j-1, k, ibg, LX, c, LZ), ı(i, j, k, ibg, args...), ℑy(i, j, k, ibg.underlying_grid, args...)))
@inline conditional_ℑy_c(LX, LZ, i, j, k, ibg::IBG{FT}, ℑy, args...) where FT = ifelse(inactive_node(i, j, k, ibg, LX, f, LZ), ı(i, j+1, k, ibg, args...), ifelse(inactive_node(i, j+1, k, ibg, LX, f, LZ), ı(i, j, k, ibg, args...), ℑy(i, j, k, ibg.underlying_grid, args...)))
@inline conditional_ℑz_f(LX, LY, i, j, k, ibg::IBG{FT}, ℑz, args...) where FT = ifelse(inactive_node(i, j, k, ibg, LX, LY, c), ı(i, j, k-1, ibg, args...), ifelse(inactive_node(i, j, k-1, ibg, LX, LY, c), ı(i, j, k, ibg, args...), ℑz(i, j, k, ibg.underlying_grid, args...)))
@inline conditional_ℑz_c(LX, LY, i, j, k, ibg::IBG{FT}, ℑz, args...) where FT = ifelse(inactive_node(i, j, k, ibg, LX, LY, f), ı(i, j, k+1, ibg, args...), ifelse(inactive_node(i, j, k+1, ibg, LX, LY, f), ı(i, j, k, ibg, args...), ℑz(i, j, k, ibg.underlying_grid, args...)))

@inline translate_loc(a) = a == :ᶠ ? :f : :c

for (d, ξ) in enumerate((:x, :y, :z))
    for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ), LZ in (:ᶠ, :ᶜ)

        δξ             = Symbol(:δ, ξ, LX, LY, LZ)
        ℑξ             = Symbol(:ℑ, ξ, LX, LY, LZ)
        loc            = translate_loc.((LX, LY, LZ))
        conditional_δξ = Symbol(:conditional_δ, ξ, :_, loc[d])
        conditional_ℑξ = Symbol(:conditional_ℑ, ξ, :_, loc[d])

        # `other_locs` contains locations in the two "other" directions not being differenced
        other_locs = []
        for l in 1:3 
            if l != d
                push!(other_locs, loc[l])
            end
        end
        
        @eval begin
            @inline $δξ(i, j, k, ibg::IBG, args...)              = $conditional_δξ($(other_locs[1]), $(other_locs[2]), i, j, k, ibg, $δξ, args...)
            @inline $δξ(i, j, k, ibg::IBG, f::Function, args...) = $conditional_δξ($(other_locs[1]), $(other_locs[2]), i, j, k, ibg, $δξ, f::Function, args...)
            @inline $ℑξ(i, j, k, ibg::IBG, args...)              = $conditional_δξ($(other_locs[1]), $(other_locs[2]), i, j, k, ibg, $ℑξ, args...)
            @inline $ℑξ(i, j, k, ibg::IBG, f::Function, args...) = $conditional_δξ($(other_locs[1]), $(other_locs[2]), i, j, k, ibg, $ℑξ, f::Function, args...)
        end
    end
end
