@inline ı(i, j, k, grid, f::Function, args...) = f(i, j, k, grid, args...)
@inline ı(i, j, k, grid, ϕ)                    = ϕ[i, j, k]

# Defining Interpolation operators that return only valid cells across a boundary
@inline conditional_ℑx_f(LY, LZ, i, j, k, grid::AbstractGrid, ℑx, args...) where FT = ifelse(inactive_node(i, j, k, grid, c, LY, LZ), ı(i-1, j, k, grid, args...), ifelse(inactive_node(i-1, j, k, grid, c, LY, LZ), ı(i, j, k, grid, args...), ℑx(i, j, k, grid.underlying_grid, args...)))
@inline conditional_ℑx_c(LY, LZ, i, j, k, grid::AbstractGrid, ℑx, args...) where FT = ifelse(inactive_node(i, j, k, grid, f, LY, LZ), ı(i+1, j, k, grid, args...), ifelse(inactive_node(i+1, j, k, grid, f, LY, LZ), ı(i, j, k, grid, args...), ℑx(i, j, k, grid.underlying_grid, args...)))
@inline conditional_ℑy_f(LX, LZ, i, j, k, grid::AbstractGrid, ℑy, args...) where FT = ifelse(inactive_node(i, j, k, grid, LX, c, LZ), ı(i, j-1, k, grid, args...), ifelse(inactive_node(i, j-1, k, grid, LX, c, LZ), ı(i, j, k, grid, args...), ℑy(i, j, k, grid.underlying_grid, args...)))
@inline conditional_ℑy_c(LX, LZ, i, j, k, grid::AbstractGrid, ℑy, args...) where FT = ifelse(inactive_node(i, j, k, grid, LX, f, LZ), ı(i, j+1, k, grid, args...), ifelse(inactive_node(i, j+1, k, grid, LX, f, LZ), ı(i, j, k, grid, args...), ℑy(i, j, k, grid.underlying_grid, args...)))
@inline conditional_ℑz_f(LX, LY, i, j, k, grid::AbstractGrid, ℑz, args...) where FT = ifelse(inactive_node(i, j, k, grid, LX, LY, c), ı(i, j, k-1, grid, args...), ifelse(inactive_node(i, j, k-1, grid, LX, LY, c), ı(i, j, k, grid, args...), ℑz(i, j, k, grid.underlying_grid, args...)))
@inline conditional_ℑz_c(LX, LY, i, j, k, grid::AbstractGrid, ℑz, args...) where FT = ifelse(inactive_node(i, j, k, grid, LX, LY, f), ı(i, j, k+1, grid, args...), ifelse(inactive_node(i, j, k+1, grid, LX, LY, f), ı(i, j, k, grid, args...), ℑz(i, j, k, grid.underlying_grid, args...)))


@inline ℑxᴮᶜᶜᶜ(i, j, k, ibg, args...) = conditional_ℑx_c(c, c, i, j, k, ibg, ℑxᶜᵃᵃ, args...)
@inline ℑxᴮᶠᶜᶜ(i, j, k, ibg, args...) = conditional_ℑx_f(c, c, i, j, k, ibg, ℑxᶠᵃᵃ, args...)
@inline ℑyᴮᶜᶜᶜ(i, j, k, ibg, args...) = conditional_ℑy_c(c, c, i, j, k, ibg, ℑyᵃᶜᵃ, args...)
@inline ℑyᴮᶜᶠᶜ(i, j, k, ibg, args...) = conditional_ℑy_f(c, c, i, j, k, ibg, ℑyᵃᶠᵃ, args...)
@inline ℑzᴮᶜᶜᶜ(i, j, k, ibg, args...) = conditional_ℑz_c(c, c, i, j, k, ibg, ℑzᵃᵃᶜ, args...)
@inline ℑzᴮᶜᶜᶠ(i, j, k, ibg, args...) = conditional_ℑz_f(c, c, i, j, k, ibg, ℑzᵃᵃᶠ, args...)

@inline ℑxᴮᶜᶠᶜ(i, j, k, ibg, args...) = conditional_ℑx_c(f, c, i, j, k, ibg, ℑxᶜᵃᵃ, args...)
@inline ℑxᴮᶠᶠᶜ(i, j, k, ibg, args...) = conditional_ℑx_f(f, c, i, j, k, ibg, ℑxᶠᵃᵃ, args...)
@inline ℑyᴮᶠᶜᶜ(i, j, k, ibg, args...) = conditional_ℑy_c(f, c, i, j, k, ibg, ℑyᵃᶜᵃ, args...)
@inline ℑyᴮᶠᶠᶜ(i, j, k, ibg, args...) = conditional_ℑy_f(f, c, i, j, k, ibg, ℑyᵃᶠᵃ, args...)
@inline ℑzᴮᶠᶜᶜ(i, j, k, ibg, args...) = conditional_ℑz_c(f, c, i, j, k, ibg, ℑzᵃᵃᶜ, args...)
@inline ℑzᴮᶠᶜᶠ(i, j, k, ibg, args...) = conditional_ℑz_f(f, c, i, j, k, ibg, ℑzᵃᵃᶠ, args...)

@inline ℑxᴮᶜᶜᶠ(i, j, k, ibg, args...) = conditional_ℑx_c(c, f, i, j, k, ibg, ℑxᶜᵃᵃ, args...)
@inline ℑxᴮᶠᶜᶠ(i, j, k, ibg, args...) = conditional_ℑx_f(c, f, i, j, k, ibg, ℑxᶠᵃᵃ, args...)
@inline ℑyᴮᶜᶜᶠ(i, j, k, ibg, args...) = conditional_ℑy_c(c, f, i, j, k, ibg, ℑyᵃᶜᵃ, args...)
@inline ℑyᴮᶜᶠᶠ(i, j, k, ibg, args...) = conditional_ℑy_f(c, f, i, j, k, ibg, ℑyᵃᶠᵃ, args...)
@inline ℑzᴮᶜᶠᶜ(i, j, k, ibg, args...) = conditional_ℑz_c(c, f, i, j, k, ibg, ℑzᵃᵃᶜ, args...)
@inline ℑzᴮᶜᶠᶠ(i, j, k, ibg, args...) = conditional_ℑz_f(c, f, i, j, k, ibg, ℑzᵃᵃᶠ, args...)

@inline ℑxᴮᶜᶠᶠ(i, j, k, ibg, args...) = conditional_ℑx_c(f, f, i, j, k, ibg, ℑxᶜᵃᵃ, args...)
@inline ℑxᴮᶠᶠᶠ(i, j, k, ibg, args...) = conditional_ℑx_f(f, f, i, j, k, ibg, ℑxᶠᵃᵃ, args...)
@inline ℑyᴮᶠᶜᶠ(i, j, k, ibg, args...) = conditional_ℑy_c(f, f, i, j, k, ibg, ℑyᵃᶜᵃ, args...)
@inline ℑyᴮᶠᶠᶠ(i, j, k, ibg, args...) = conditional_ℑy_f(f, f, i, j, k, ibg, ℑyᵃᶠᵃ, args...)
@inline ℑzᴮᶠᶠᶜ(i, j, k, ibg, args...) = conditional_ℑz_c(f, f, i, j, k, ibg, ℑzᵃᵃᶜ, args...)
@inline ℑzᴮᶠᶠᶠ(i, j, k, ibg, args...) = conditional_ℑz_f(f, f, i, j, k, ibg, ℑzᵃᵃᶠ, args...)

@inline ℑxyᴮᶜᶜᶜ(i, j, k, grid, args...) = ℑyᴮᶜᶜᶜ(i, j, k, grid, ℑxᴮᶜᶜᶜ, args...)
@inline ℑxyᴮᶠᶜᶜ(i, j, k, grid, args...) = ℑyᴮᶠᶜᶜ(i, j, k, grid, ℑxᴮᶠᶜᶜ, args...)
@inline ℑxyᴮᶜᶠᶜ(i, j, k, grid, args...) = ℑyᴮᶜᶠᶜ(i, j, k, grid, ℑxᴮᶜᶠᶜ, args...)
@inline ℑxyᴮᶠᶠᶜ(i, j, k, grid, args...) = ℑyᴮᶠᶠᶜ(i, j, k, grid, ℑxᴮᶠᶠᶜ, args...)
@inline ℑxyᴮᶜᶠᶠ(i, j, k, grid, args...) = ℑyᴮᶜᶠᶠ(i, j, k, grid, ℑxᴮᶜᶠᶠ, args...)
@inline ℑxyᴮᶠᶠᶠ(i, j, k, grid, args...) = ℑyᴮᶠᶠᶠ(i, j, k, grid, ℑxᴮᶠᶠᶠ, args...)

@inline ℑxzᴮᶜᶜᶜ(i, j, k, grid, args...) = ℑzᴮᶜᶜᶜ(i, j, k, grid, ℑxᴮᶜᶜᶜ, args...)
@inline ℑxzᴮᶠᶜᶜ(i, j, k, grid, args...) = ℑzᴮᶠᶜᶜ(i, j, k, grid, ℑxᴮᶠᶜᶜ, args...)
@inline ℑxzᴮᶜᶠᶜ(i, j, k, grid, args...) = ℑzᴮᶜᶠᶜ(i, j, k, grid, ℑxᴮᶜᶠᶜ, args...)
@inline ℑxzᴮᶠᶠᶜ(i, j, k, grid, args...) = ℑzᴮᶠᶠᶜ(i, j, k, grid, ℑxᴮᶠᶠᶜ, args...)
@inline ℑxzᴮᶜᶠᶠ(i, j, k, grid, args...) = ℑzᴮᶜᶠᶠ(i, j, k, grid, ℑxᴮᶜᶠᶠ, args...)
@inline ℑxzᴮᶠᶠᶠ(i, j, k, grid, args...) = ℑzᴮᶠᶠᶠ(i, j, k, grid, ℑxᴮᶠᶠᶠ, args...)

@inline ℑyzᴮᶜᶜᶜ(i, j, k, grid, args...) = ℑzᴮᶜᶜᶜ(i, j, k, grid, ℑyᴮᶜᶜᶜ, args...)
@inline ℑyzᴮᶠᶜᶜ(i, j, k, grid, args...) = ℑzᴮᶠᶜᶜ(i, j, k, grid, ℑyᴮᶠᶜᶜ, args...)
@inline ℑyzᴮᶜᶠᶜ(i, j, k, grid, args...) = ℑzᴮᶜᶠᶜ(i, j, k, grid, ℑyᴮᶜᶠᶜ, args...)
@inline ℑyzᴮᶠᶠᶜ(i, j, k, grid, args...) = ℑzᴮᶠᶠᶜ(i, j, k, grid, ℑyᴮᶠᶠᶜ, args...)
@inline ℑyzᴮᶜᶠᶠ(i, j, k, grid, args...) = ℑzᴮᶜᶠᶠ(i, j, k, grid, ℑyᴮᶜᶠᶠ, args...)
@inline ℑyzᴮᶠᶠᶠ(i, j, k, grid, args...) = ℑzᴮᶠᶠᶠ(i, j, k, grid, ℑyᴮᶠᶠᶠ, args...)

@inline ℑxyzᴮᶜᶜᶜ(i, j, k, grid, args...) = ℑzᴮᶜᶜᶜ(i, j, k, grid, ℑxyᴮᶜᶜᶜ, args...)
@inline ℑxyzᴮᶠᶜᶜ(i, j, k, grid, args...) = ℑzᴮᶠᶜᶜ(i, j, k, grid, ℑxyᴮᶠᶜᶜ, args...)
@inline ℑxyzᴮᶜᶠᶜ(i, j, k, grid, args...) = ℑzᴮᶜᶠᶜ(i, j, k, grid, ℑxyᴮᶜᶠᶜ, args...)
@inline ℑxyzᴮᶠᶠᶜ(i, j, k, grid, args...) = ℑzᴮᶠᶠᶜ(i, j, k, grid, ℑxyᴮᶠᶠᶜ, args...)
@inline ℑxyzᴮᶜᶠᶠ(i, j, k, grid, args...) = ℑzᴮᶜᶠᶠ(i, j, k, grid, ℑxyᴮᶜᶠᶠ, args...)
@inline ℑxyzᴮᶠᶠᶠ(i, j, k, grid, args...) = ℑzᴮᶠᶠᶠ(i, j, k, grid, ℑxyᴮᶠᶠᶠ, args...)