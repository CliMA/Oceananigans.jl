
# Defining all the First order derivatives for the immersed boundaries

@inline conditional_x_derivative_f(LY, LZ, i, j, k, ibg::IBG{FT}, deriv, args...) where FT = ifelse(solid_node(c, LY, LZ, i, j, k, ibg) || solid_node(c, LY, LZ, i-1, j, k, ibg), zero(FT), deriv(i, j, k, ibg.grid, args...))
@inline conditional_x_derivative_c(LY, LZ, i, j, k, ibg::IBG{FT}, deriv, args...) where FT = ifelse(solid_node(f, LY, LZ, i, j, k, ibg) || solid_node(f, LY, LZ, i+1, j, k, ibg), zero(FT), deriv(i, j, k, ibg.grid, args...))
@inline conditional_y_derivative_f(LX, LZ, i, j, k, ibg::IBG{FT}, deriv, args...) where FT = ifelse(solid_node(LX, c, LZ, i, j, k, ibg) || solid_node(LX, c, LZ, i, j-1, k, ibg), zero(FT), deriv(i, j, k, ibg.grid, args...))
@inline conditional_y_derivative_c(LX, LZ, i, j, k, ibg::IBG{FT}, deriv, args...) where FT = ifelse(solid_node(LX, f, LZ, i, j, k, ibg) || solid_node(LX, f, LZ, i, j+1, k, ibg), zero(FT), deriv(i, j, k, ibg.grid, args...))
@inline conditional_z_derivative_f(LX, LY, i, j, k, ibg::IBG{FT}, deriv, args...) where FT = ifelse(solid_node(LX, LY, c, i, j, k, ibg) || solid_node(LX, LY, c, i, j, k-1, ibg), zero(FT), deriv(i, j, k, ibg.grid, args...))
@inline conditional_z_derivative_c(LX, LY, i, j, k, ibg::IBG{FT}, deriv, args...) where FT = ifelse(solid_node(LX, LY, f, i, j, k, ibg) || solid_node(LX, LY, f, i, j, k+1, ibg), zero(FT), deriv(i, j, k, ibg.grid, args...))

∂xᶠᶜᶜ(i, j, k, ibg::IBG, args...) = conditional_x_derivative_f(c, c, i, j, k, ibg, ∂xᶠᶜᶜ, args...)
∂xᶠᶜᶠ(i, j, k, ibg::IBG, args...) = conditional_x_derivative_f(c, f, i, j, k, ibg, ∂xᶠᶜᶠ, args...)
∂xᶠᶠᶜ(i, j, k, ibg::IBG, args...) = conditional_x_derivative_f(f, c, i, j, k, ibg, ∂xᶠᶠᶜ, args...)
∂xᶠᶠᶠ(i, j, k, ibg::IBG, args...) = conditional_x_derivative_f(f, f, i, j, k, ibg, ∂xᶠᶠᶠ, args...)

∂xᶜᶜᶜ(i, j, k, ibg::IBG, args...) = conditional_x_derivative_c(c, c, i, j, k, ibg, ∂xᶜᶜᶜ, args...)
∂xᶜᶜᶠ(i, j, k, ibg::IBG, args...) = conditional_x_derivative_c(c, f, i, j, k, ibg, ∂xᶜᶜᶠ, args...)
∂xᶜᶠᶜ(i, j, k, ibg::IBG, args...) = conditional_x_derivative_c(f, c, i, j, k, ibg, ∂xᶜᶠᶜ, args...)
∂xᶜᶠᶠ(i, j, k, ibg::IBG, args...) = conditional_x_derivative_c(f, f, i, j, k, ibg, ∂xᶜᶠᶠ, args...)

∂yᶜᶠᶜ(i, j, k, ibg::IBG, args...) = conditional_y_derivative_f(c, c, i, j, k, ibg, ∂yᶜᶠᶜ, args...)
∂yᶜᶠᶠ(i, j, k, ibg::IBG, args...) = conditional_y_derivative_f(c, f, i, j, k, ibg, ∂yᶜᶠᶠ, args...)
∂yᶠᶠᶜ(i, j, k, ibg::IBG, args...) = conditional_y_derivative_f(f, c, i, j, k, ibg, ∂yᶠᶠᶜ, args...)
∂yᶠᶠᶠ(i, j, k, ibg::IBG, args...) = conditional_y_derivative_f(f, f, i, j, k, ibg, ∂yᶠᶠᶠ, args...)

∂yᶜᶜᶜ(i, j, k, ibg::IBG, args...) = conditional_y_derivative_c(c, c, i, j, k, ibg, ∂yᶜᶜᶜ, args...)
∂yᶜᶜᶠ(i, j, k, ibg::IBG, args...) = conditional_y_derivative_c(c, f, i, j, k, ibg, ∂yᶜᶜᶠ, args...)
∂yᶠᶜᶜ(i, j, k, ibg::IBG, args...) = conditional_y_derivative_c(f, c, i, j, k, ibg, ∂yᶠᶜᶜ, args...)
∂yᶠᶜᶠ(i, j, k, ibg::IBG, args...) = conditional_y_derivative_c(f, f, i, j, k, ibg, ∂yᶠᶜᶠ, args...)

∂zᶜᶜᶠ(i, j, k, ibg::IBG, args...) = conditional_z_derivative_f(c, c, i, j, k, ibg, ∂zᶜᶜᶠ, args...)
∂zᶠᶜᶠ(i, j, k, ibg::IBG, args...) = conditional_z_derivative_f(c, f, i, j, k, ibg, ∂zᶠᶜᶠ, args...)
∂zᶜᶠᶠ(i, j, k, ibg::IBG, args...) = conditional_z_derivative_f(f, c, i, j, k, ibg, ∂zᶠᶜᶠ, args...)
∂zᶠᶠᶠ(i, j, k, ibg::IBG, args...) = conditional_z_derivative_f(f, f, i, j, k, ibg, ∂zᶠᶠᶠ, args...)

∂zᶜᶜᶜ(i, j, k, ibg::IBG, args...) = conditional_z_derivative_c(c, c, i, j, k, ibg, ∂zᶜᶜᶜ, args...)
∂zᶜᶠᶜ(i, j, k, ibg::IBG, args...) = conditional_z_derivative_c(c, f, i, j, k, ibg, ∂zᶜᶠᶜ, args...)
∂zᶠᶜᶜ(i, j, k, ibg::IBG, args...) = conditional_z_derivative_c(f, c, i, j, k, ibg, ∂zᶠᶜᶜ, args...)
∂zᶠᶠᶜ(i, j, k, ibg::IBG, args...) = conditional_z_derivative_c(f, f, i, j, k, ibg, ∂zᶠᶠᶜ, args...)
