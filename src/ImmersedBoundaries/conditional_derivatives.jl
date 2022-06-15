import Oceananigans.Operators: 
            ∂xᶠᶜᶜ, ∂xᶠᶜᶠ, ∂xᶠᶠᶜ, ∂xᶠᶠᶠ,
            ∂xᶜᶜᶜ, ∂xᶜᶜᶠ, ∂xᶜᶠᶜ, ∂xᶜᶠᶠ,
            ∂yᶜᶠᶜ, ∂yᶜᶠᶠ, ∂yᶠᶠᶜ, ∂yᶠᶠᶠ,
            ∂yᶜᶜᶜ, ∂yᶜᶜᶠ, ∂yᶠᶜᶜ, ∂yᶠᶜᶠ,
            ∂zᶜᶜᶠ, ∂zᶜᶠᶠ, ∂zᶠᶜᶠ, ∂zᶠᶠᶠ,
            ∂zᶜᶜᶜ, ∂zᶜᶠᶜ, ∂zᶠᶜᶜ, ∂zᶠᶠᶜ

# Defining all the First order derivatives for the immersed boundaries

@inline conditional_x_derivative_f(LY, LZ, i, j, k, ibg::IBG{FT}, deriv, args...) where FT = ifelse(inactive_node(c, LY, LZ, i, j, k, ibg) | inactive_node(c, LY, LZ, i-1, j, k, ibg), zero(FT), deriv(i, j, k, ibg.underlying_grid, args...))
@inline conditional_x_derivative_c(LY, LZ, i, j, k, ibg::IBG{FT}, deriv, args...) where FT = ifelse(inactive_node(f, LY, LZ, i, j, k, ibg) | inactive_node(f, LY, LZ, i+1, j, k, ibg), zero(FT), deriv(i, j, k, ibg.underlying_grid, args...))
@inline conditional_y_derivative_f(LX, LZ, i, j, k, ibg::IBG{FT}, deriv, args...) where FT = ifelse(inactive_node(LX, c, LZ, i, j, k, ibg) | inactive_node(LX, c, LZ, i, j-1, k, ibg), zero(FT), deriv(i, j, k, ibg.underlying_grid, args...))
@inline conditional_y_derivative_c(LX, LZ, i, j, k, ibg::IBG{FT}, deriv, args...) where FT = ifelse(inactive_node(LX, f, LZ, i, j, k, ibg) | inactive_node(LX, f, LZ, i, j+1, k, ibg), zero(FT), deriv(i, j, k, ibg.underlying_grid, args...))
@inline conditional_z_derivative_f(LX, LY, i, j, k, ibg::IBG{FT}, deriv, args...) where FT = ifelse(inactive_node(LX, LY, c, i, j, k, ibg) | inactive_node(LX, LY, c, i, j, k-1, ibg), zero(FT), deriv(i, j, k, ibg.underlying_grid, args...))
@inline conditional_z_derivative_c(LX, LY, i, j, k, ibg::IBG{FT}, deriv, args...) where FT = ifelse(inactive_node(LX, LY, f, i, j, k, ibg) | inactive_node(LX, LY, f, i, j, k+1, ibg), zero(FT), deriv(i, j, k, ibg.underlying_grid, args...))

@inline translate_loc(a) = a == :ᶠ ? :f : :c

for (d, ξ) in enumerate((:x, :y, :z))
    for locx in (:ᶠ, :ᶜ), locy in (:ᶠ, :ᶜ), locz in (:ᶠ, :ᶜ)

        der             = Symbol(:∂, ξ, locx, locy, locz)
        location        = translate_loc.((locx, locy, locz))
        conditional_der = Symbol(:conditional_, ξ, :_derivative_, location[d])
        loc = []
        for l in 1:3 
            if l != d
                push!(loc, location[l])
            end
        end
        
        @eval begin
            $der(i, j, k, ibg::IBG, args...)              = $conditional_der($(loc[1]), $(loc[2]), i, j, k, ibg, $der, args...)
            $der(i, j, k, ibg::IBG, f::Function, args...) = $conditional_der($(loc[1]), $(loc[2]), i, j, k, ibg, $der, f::Function, args...)
        end
    end
end

using  Oceananigans.Operators
import Oceananigans.Operators: Γᶠᶠᶜ

# Circulation equal to zero on a solid nodes
@inline Γᶠᶠᶜ(i, j, k, ibg::IBG, u, v) =  
    conditional_x_derivative_f(f, c, i, j, k, ibg, δxᶠᵃᵃ, Δy_qᶜᶠᶜ, v) - conditional_y_derivative_f(f, c, i, j, k, ibg, δyᵃᶠᵃ, Δx_qᶠᶜᶜ, u)


