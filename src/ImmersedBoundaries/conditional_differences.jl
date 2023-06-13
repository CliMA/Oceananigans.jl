import Oceananigans.Operators: 
    δxᶠᶜᶜ, δxᶠᶜᶠ, δxᶠᶠᶜ, δxᶠᶠᶠ,
    δxᶜᶜᶜ, δxᶜᶜᶠ, δxᶜᶠᶜ, δxᶜᶠᶠ,
    δyᶜᶠᶜ, δyᶜᶠᶠ, δyᶠᶠᶜ, δyᶠᶠᶠ,
    δyᶜᶜᶜ, δyᶜᶜᶠ, δyᶠᶜᶜ, δyᶠᶜᶠ,
    δzᶜᶜᶠ, δzᶜᶠᶠ, δzᶠᶜᶠ, δzᶠᶠᶠ,
    δzᶜᶜᶜ, δzᶜᶠᶜ, δzᶠᶜᶜ, δzᶠᶠᶜ

# Defining Difference operators for the immersed boundaries
@inline conditional_δx_f(LY, LZ, i, j, k, ibg::IBG{FT}, δx, args...) where FT = ifelse(inactive_node(i, j, k, ibg, c, LY, LZ) | inactive_node(i-1, j, k, ibg, c, LY, LZ), zero(FT), δx(i, j, k, ibg.underlying_grid, args...))
@inline conditional_δx_c(LY, LZ, i, j, k, ibg::IBG{FT}, δx, args...) where FT = ifelse(inactive_node(i, j, k, ibg, f, LY, LZ) | inactive_node(i+1, j, k, ibg, f, LY, LZ), zero(FT), δx(i, j, k, ibg.underlying_grid, args...))
@inline conditional_δy_f(LX, LZ, i, j, k, ibg::IBG{FT}, δy, args...) where FT = ifelse(inactive_node(i, j, k, ibg, LX, c, LZ) | inactive_node(i, j-1, k, ibg, LX, c, LZ), zero(FT), δy(i, j, k, ibg.underlying_grid, args...))
@inline conditional_δy_c(LX, LZ, i, j, k, ibg::IBG{FT}, δy, args...) where FT = ifelse(inactive_node(i, j, k, ibg, LX, f, LZ) | inactive_node(i, j+1, k, ibg, LX, f, LZ), zero(FT), δy(i, j, k, ibg.underlying_grid, args...))
@inline conditional_δz_f(LX, LY, i, j, k, ibg::IBG{FT}, δz, args...) where FT = ifelse(inactive_node(i, j, k, ibg, LX, LY, c) | inactive_node(i, j, k-1, ibg, LX, LY, c), zero(FT), δz(i, j, k, ibg.underlying_grid, args...))
@inline conditional_δz_c(LX, LY, i, j, k, ibg::IBG{FT}, δz, args...) where FT = ifelse(inactive_node(i, j, k, ibg, LX, LY, f) | inactive_node(i, j, k+1, ibg, LX, LY, f), zero(FT), δz(i, j, k, ibg.underlying_grid, args...))

@inline translate_loc(a) = a == :ᶠ ? :f : :c

for (d, ξ) in enumerate((:x, :y, :z))
    for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ), LZ in (:ᶠ, :ᶜ)

        δξ             = Symbol(:δ, ξ, LX, LY, LZ)
        loc            = translate_loc.((LX, LY, LZ))
        conditional_δξ = Symbol(:conditional_δ, ξ, :_, loc[d])

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
       end
    end
end
