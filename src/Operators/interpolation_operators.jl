using Oceananigans.Grids: Flat, node_has_data
using Oceananigans.Grids: peripheral_node

#####
##### Base interpolation operators
#####

@inline ℑxᶜᵃᵃ(i, j, k, grid::AG{FT}, u) where FT = @inbounds FT(0.5) * (u[i,   j, k] + u[i+1, j, k])
@inline ℑxᶠᵃᵃ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i-1, j, k] + c[i,   j, k])

@inline ℑyᵃᶜᵃ(i, j, k, grid::AG{FT}, v) where FT = @inbounds FT(0.5) * (v[i, j,   k] + v[i,  j+1, k])
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i, j-1, k] + c[i,  j,   k])

@inline ℑzᵃᵃᶜ(i, j, k, grid::AG{FT}, w) where FT = @inbounds FT(0.5) * (w[i, j,   k] + w[i, j, k+1])
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG{FT}, c) where FT = @inbounds FT(0.5) * (c[i, j, k-1] + c[i, j,   k])

#####
##### Interpolation operators acting on functions
#####

@inline ℑxᶜᵃᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i,   j, k, grid, args...) + f(i+1, j, k, grid, args...))
@inline ℑxᶠᵃᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i-1, j, k, grid, args...) + f(i,   j, k, grid, args...))

@inline ℑyᵃᶜᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j,   k, grid, args...) + f(i, j+1, k, grid, args...))
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j-1, k, grid, args...) + f(i, j,   k, grid, args...))

@inline ℑzᵃᵃᶜ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j, k,   grid, args...) + f(i, j, k+1, grid, args...))
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * (f(i, j, k-1, grid, args...) + f(i, j, k,   grid, args...))

#####
##### Convenience operators for "interpolating constants"
#####

@inline ℑxᶠᵃᵃ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑxᶜᵃᵃ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑyᵃᶜᵃ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG, f::Number, args...) = f
@inline ℑzᵃᵃᶜ(i, j, k, grid::AG, f::Number, args...) = f

#####
##### Double interpolation
#####

@inline ℑxyᶜᶜᵃ(i, j, k, grid, f, args...) = ℑyᵃᶜᵃ(i, j, k, grid, ℑxᶜᵃᵃ, f, args...)
@inline ℑxyᶠᶜᵃ(i, j, k, grid, f, args...) = ℑyᵃᶜᵃ(i, j, k, grid, ℑxᶠᵃᵃ, f, args...)
@inline ℑxyᶠᶠᵃ(i, j, k, grid, f, args...) = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶠᵃᵃ, f, args...)
@inline ℑxyᶜᶠᵃ(i, j, k, grid, f, args...) = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, f, args...)
@inline ℑxzᶜᵃᶜ(i, j, k, grid, f, args...) = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶜᵃᵃ, f, args...)
@inline ℑxzᶠᵃᶜ(i, j, k, grid, f, args...) = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, f, args...)
@inline ℑxzᶠᵃᶠ(i, j, k, grid, f, args...) = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶠᵃᵃ, f, args...)
@inline ℑxzᶜᵃᶠ(i, j, k, grid, f, args...) = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, f, args...)
@inline ℑyzᵃᶜᶜ(i, j, k, grid, f, args...) = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶜᵃ, f, args...)
@inline ℑyzᵃᶠᶜ(i, j, k, grid, f, args...) = ℑzᵃᵃᶜ(i, j, k, grid, ℑyᵃᶠᵃ, f, args...)
@inline ℑyzᵃᶠᶠ(i, j, k, grid, f, args...) = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶠᵃ, f, args...)
@inline ℑyzᵃᶜᶠ(i, j, k, grid, f, args...) = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, f, args...)

#####
##### Leftover
#####

@inline ℑxᶠᵃᵃ(i, j, k, grid::AG{FT}, c, args...) where FT = @inbounds FT(0.5) * (c[i-1, j, k] + c[i, j, k])
@inline ℑyᵃᶠᵃ(i, j, k, grid::AG{FT}, c, args...) where FT = @inbounds FT(0.5) * (c[i, j-1, k] + c[i, j, k])
@inline ℑzᵃᵃᶠ(i, j, k, grid::AG{FT}, c, args...) where FT = @inbounds FT(0.5) * (c[i, j, k-1] + c[i, j, k])

#####
##### Support for Flat Earths
#####

using Oceananigans.Grids: XFlatGrid, YFlatGrid, ZFlatGrid

@inline ℑxᶜᵃᵃ(i, j, k, grid::XFlatGrid, u) = @inbounds u[i, j, k]
@inline ℑxᶠᵃᵃ(i, j, k, grid::XFlatGrid, c) = @inbounds c[i, j, k]

@inline ℑyᵃᶜᵃ(i, j, k, grid::YFlatGrid, v) = @inbounds v[i, j, k]
@inline ℑyᵃᶠᵃ(i, j, k, grid::YFlatGrid, c) = @inbounds c[i, j, k]

@inline ℑzᵃᵃᶜ(i, j, k, grid::ZFlatGrid, w) = @inbounds w[i, j, k]
@inline ℑzᵃᵃᶠ(i, j, k, grid::ZFlatGrid, c) = @inbounds c[i, j, k]

@inline ℑxᶜᵃᵃ(i, j, k, grid::XFlatGrid, u::Number) = u
@inline ℑxᶠᵃᵃ(i, j, k, grid::XFlatGrid, c::Number) = c

@inline ℑyᵃᶜᵃ(i, j, k, grid::YFlatGrid, v::Number) = v
@inline ℑyᵃᶠᵃ(i, j, k, grid::YFlatGrid, c::Number) = c

@inline ℑzᵃᵃᶜ(i, j, k, grid::ZFlatGrid, w::Number) = w
@inline ℑzᵃᵃᶠ(i, j, k, grid::ZFlatGrid, c::Number) = c

@inline ℑxᶜᵃᵃ(i, j, k, grid::XFlatGrid, f::F, args...) where {F<:Function} = f(i, j, k, grid, args...)
@inline ℑxᶠᵃᵃ(i, j, k, grid::XFlatGrid, f::F, args...) where {F<:Function} = f(i, j, k, grid, args...)

@inline ℑyᵃᶜᵃ(i, j, k, grid::YFlatGrid, f::F, args...) where {F<:Function} = f(i, j, k, grid, args...)
@inline ℑyᵃᶠᵃ(i, j, k, grid::YFlatGrid, f::F, args...) where {F<:Function} = f(i, j, k, grid, args...)

@inline ℑzᵃᵃᶜ(i, j, k, grid::ZFlatGrid, f::F, args...) where {F<:Function} = f(i, j, k, grid, args...)
@inline ℑzᵃᵃᶠ(i, j, k, grid::ZFlatGrid, f::F, args...) where {F<:Function} = f(i, j, k, grid, args...)


#####
##### Location-explicit interpolations average the operands with data, take the one with data next to a boundary, and
##### vanish where neither has data: beyond the sides of the interpolated direction, or inside an immersed boundary
#####

@inline function conditional_ℑ(i, j, k, grid, ::Val{d}, ℓx, ℓy, ℓz, ψ, args...) where d
    n⁻, n⁺, ℓᵒ = operands(i, j, k, Val(d), ℓx, ℓy, ℓz)
    d⁻ = node_has_data(n⁻..., grid, Val(d), ℓᵒ...)
    d⁺ = node_has_data(n⁺..., grid, Val(d), ℓᵒ...)
    ψ⁻ = getvalue(ψ, n⁻..., grid, args...)
    ψ⁺ = getvalue(ψ, n⁺..., grid, args...)
    return ifelse(d⁻ & d⁺, interior_ℑ(i, j, k, grid, Val(d), ℓx, ℓy, ℓz, ψ⁻, ψ⁺, ψ, args...), ifelse(d⁻, ψ⁻, ifelse(d⁺, ψ⁺, zero(ψ⁺))))
end

# The interpolation between two operands with data
@inline interior_ℑ(i, j, k, grid, d, ℓx, ℓy, ℓz, ψ⁻, ψ⁺, ψ, args...) = (ψ⁻ + ψ⁺) / 2

@inline conditional_ℑ(i, j, k, grid::XFlatGrid, ::Val{1}, ℓx, ℓy, ℓz, ψ, args...) = getvalue(ψ, i, j, k, grid, args...)
@inline conditional_ℑ(i, j, k, grid::YFlatGrid, ::Val{2}, ℓx, ℓy, ℓz, ψ, args...) = getvalue(ψ, i, j, k, grid, args...)
@inline conditional_ℑ(i, j, k, grid::ZFlatGrid, ::Val{3}, ℓx, ℓy, ℓz, ψ, args...) = getvalue(ψ, i, j, k, grid, args...)

# One-direction interpolations, named by the location they interpolate to
for ℓ1 in (:ᶜ, :ᶠ), ℓ2 in (:ᶜ, :ᶠ), ℓ3 in (:ᶜ, :ᶠ)
    ℑx = Symbol(:ℑx, ℓ1, ℓ2, ℓ3)
    ℑy = Symbol(:ℑy, ℓ2, ℓ1, ℓ3)
    ℑz = Symbol(:ℑz, ℓ2, ℓ3, ℓ1)

    ℓ  = ℓ1 == :ᶜ ? :(Center()) : :(Face())
    ℓa = ℓ2 == :ᶜ ? :(Center()) : :(Face())
    ℓb = ℓ3 == :ᶜ ? :(Center()) : :(Face())

    @eval begin
        @inline $ℑx(i, j, k, grid, ψ, args...) = conditional_ℑ(i, j, k, grid, Val(1), $ℓ, $ℓa, $ℓb, ψ, args...)
        @inline $ℑy(i, j, k, grid, ψ, args...) = conditional_ℑ(i, j, k, grid, Val(2), $ℓa, $ℓ, $ℓb, ψ, args...)
        @inline $ℑz(i, j, k, grid, ψ, args...) = conditional_ℑ(i, j, k, grid, Val(3), $ℓa, $ℓb, $ℓ, ψ, args...)

        @inline $ℑx(i, j, k, grid, ψ::Number, args...) = ψ
        @inline $ℑy(i, j, k, grid, ψ::Number, args...) = ψ
        @inline $ℑz(i, j, k, grid, ψ::Number, args...) = ψ

        export $ℑx, $ℑy, $ℑz
    end
end

# Interpolations in two and three directions, composed of one-direction ones: x before y before z, except xyz, which interpolates z first
for ℓx in (:ᶜ, :ᶠ), ℓy in (:ᶜ, :ᶠ), ℓz in (:ᶜ, :ᶠ)
    ℓx⁰ = ℓx == :ᶜ ? :ᶠ : :ᶜ
    ℓy⁰ = ℓy == :ᶜ ? :ᶠ : :ᶜ
    ℓz⁰ = ℓz == :ᶜ ? :ᶠ : :ᶜ

    ℑxy  = Symbol(:ℑxy,  ℓx, ℓy, ℓz)
    ℑxz  = Symbol(:ℑxz,  ℓx, ℓy, ℓz)
    ℑyz  = Symbol(:ℑyz,  ℓx, ℓy, ℓz)
    ℑxyz = Symbol(:ℑxyz, ℓx, ℓy, ℓz)

    x_then_y = Symbol(:ℑx, ℓx, ℓy⁰, ℓz)
    x_then_z = Symbol(:ℑx, ℓx, ℓy, ℓz⁰)
    y_then_z = Symbol(:ℑy, ℓx, ℓy, ℓz⁰)
    last_x   = Symbol(:ℑx, ℓx, ℓy, ℓz)
    last_y   = Symbol(:ℑy, ℓx, ℓy, ℓz)
    last_z   = Symbol(:ℑz, ℓx, ℓy, ℓz)
    middle_y = Symbol(:ℑy, ℓx⁰, ℓy, ℓz)
    first_z  = Symbol(:ℑz, ℓx⁰, ℓy⁰, ℓz)

    @eval begin
        @inline $ℑxy(i, j, k, grid, ψ, args...)  = $last_y(i, j, k, grid, $x_then_y, ψ, args...)
        @inline $ℑxz(i, j, k, grid, ψ, args...)  = $last_z(i, j, k, grid, $x_then_z, ψ, args...)
        @inline $ℑyz(i, j, k, grid, ψ, args...)  = $last_z(i, j, k, grid, $y_then_z, ψ, args...)
        @inline $ℑxyz(i, j, k, grid, ψ, args...) = $last_x(i, j, k, grid, $middle_y, $first_z, ψ, args...)

        export $ℑxy, $ℑxz, $ℑyz, $ℑxyz
    end
end

#####
##### Active-weighted interpolation
#####

@inline not_peripheral_node(args...) = !peripheral_node(args...)

@inline function active_weighted_ℑxyᶜᶠᶜ(i, j, k, grid, q, args...)
    active_nodes = ℑxyᶜᶠᵃ(i, j, k, grid, not_peripheral_node, f, c, c)
    mask = active_nodes == 0
    return ifelse(mask, zero(grid), ℑxyᶜᶠᵃ(i, j, k, grid, q, args...) / active_nodes)
end

@inline function active_weighted_ℑxyᶠᶜᶜ(i, j, k, grid, q, args...)
    active_nodes = ℑxyᶠᶜᵃ(i, j, k, grid, not_peripheral_node, c, f, c)
    mask = active_nodes == 0
    return ifelse(mask, zero(grid), ℑxyᶠᶜᵃ(i, j, k, grid, q, args...) / active_nodes)
end

@inline function active_weighted_ℑxyᶠᶠᶜ(i, j, k, grid, q, args...)
    active_nodes = ℑxyᶠᶠᵃ(i, j, k, grid, not_peripheral_node, c, c, c)
    mask = active_nodes == 0
    return ifelse(mask, zero(grid), ℑxyᶠᶠᵃ(i, j, k, grid, q, args...) / active_nodes)
end

@inline function active_weighted_ℑxyᶜᶜᶜ(i, j, k, grid, q, args...)
    active_nodes = ℑxyᶜᶜᵃ(i, j, k, grid, not_peripheral_node, f, f, c)
    mask = active_nodes == 0
    return ifelse(mask, zero(grid), ℑxyᶜᶜᵃ(i, j, k, grid, q, args...) / active_nodes)
end

@inline function active_weighted_ℑxzᶜᶜᶜ(i, j, k, grid, q, args...)
    active_nodes = ℑxzᶜᵃᶜ(i, j, k, grid, not_peripheral_node, f, c, f)
    mask = active_nodes == 0
    return ifelse(mask, zero(grid), ℑxzᶜᵃᶜ(i, j, k, grid, q, args...) / active_nodes)
end

@inline function active_weighted_ℑyzᶜᶜᶜ(i, j, k, grid, q, args...)
    active_nodes = ℑyzᵃᶜᶜ(i, j, k, grid, not_peripheral_node, c, f, f)
    mask = active_nodes == 0
    return ifelse(mask, zero(grid), ℑyzᵃᶜᶜ(i, j, k, grid, q, args...) / active_nodes)
end
