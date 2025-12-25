using Oceananigans.ImmersedBoundaries
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, immersed_inactive_node
using Oceananigans.Fields: ZeroField

const IBG = ImmersedBoundaryGrid

const c = Center()
const f = Face()

"""
    conditional_flux(i, j, k, ibg::IBG, ℓx, ℓy, ℓz, qᴮ, qᴵ, nc)

Return either

    i) The boundary flux `qᴮ` if the node condition `nc` is true (default: `nc = immersed_peripheral_node`), or
    ii) The interior flux `qᴵ` otherwise.

This can be used either to condition intrinsic flux functions, or immersed boundary flux functions.
"""
@inline function conditional_flux(i, j, k, ibg, ℓx, ℓy, ℓz, q_boundary, q_interior)
    on_immersed_periphery = immersed_peripheral_node(i, j, k, ibg, ℓx, ℓy, ℓz)
    return ifelse(on_immersed_periphery, q_boundary, q_interior)
end

# Conveniences
@inline conditional_flux_ccc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, c, c, qᴮ, qᴵ)
@inline conditional_flux_ffc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, f, f, c, qᴮ, qᴵ)
@inline conditional_flux_fcf(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, f, c, f, qᴮ, qᴵ)
@inline conditional_flux_cff(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, f, f, qᴮ, qᴵ)

@inline conditional_flux_fcc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, f, c, c, qᴮ, qᴵ)
@inline conditional_flux_cfc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, f, c, qᴮ, qᴵ)
@inline conditional_flux_ccf(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, c, f, qᴮ, qᴵ)

#####
##### Immersed Advective fluxes
#####

# Fallback for `nothing` advection
@inline _advective_tracer_flux_x(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_tracer_flux_y(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_tracer_flux_z(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)

# Disambiguation for `FluxForm` momentum fluxes....
@inline _advective_momentum_flux_Uu(i, j, k, ibg::IBG, scheme::FluxFormAdvection, U, u) = _advective_momentum_flux_Uu(i, j, k, ibg, scheme.x, U, u)
@inline _advective_momentum_flux_Vu(i, j, k, ibg::IBG, scheme::FluxFormAdvection, V, u) = _advective_momentum_flux_Vu(i, j, k, ibg, scheme.y, V, u)
@inline _advective_momentum_flux_Wu(i, j, k, ibg::IBG, scheme::FluxFormAdvection, W, u) = _advective_momentum_flux_Wu(i, j, k, ibg, scheme.z, W, u)

@inline _advective_momentum_flux_Uv(i, j, k, ibg::IBG, scheme::FluxFormAdvection, U, v) = _advective_momentum_flux_Uv(i, j, k, ibg, scheme.x, U, v)
@inline _advective_momentum_flux_Vv(i, j, k, ibg::IBG, scheme::FluxFormAdvection, V, v) = _advective_momentum_flux_Vv(i, j, k, ibg, scheme.y, V, v)
@inline _advective_momentum_flux_Wv(i, j, k, ibg::IBG, scheme::FluxFormAdvection, W, v) = _advective_momentum_flux_Wv(i, j, k, ibg, scheme.z, W, v)

@inline _advective_momentum_flux_Uw(i, j, k, ibg::IBG, scheme::FluxFormAdvection, U, w) = _advective_momentum_flux_Uw(i, j, k, ibg, scheme.x, U, w)
@inline _advective_momentum_flux_Vw(i, j, k, ibg::IBG, scheme::FluxFormAdvection, V, w) = _advective_momentum_flux_Vw(i, j, k, ibg, scheme.y, V, w)
@inline _advective_momentum_flux_Ww(i, j, k, ibg::IBG, scheme::FluxFormAdvection, W, w) = _advective_momentum_flux_Ww(i, j, k, ibg, scheme.z, W, w)

# Disambiguation for `FluxForm` tracer fluxes....
@inline _advective_tracer_flux_x(i, j, k, ibg::IBG, scheme::FluxFormAdvection, U, c) =
        _advective_tracer_flux_x(i, j, k, ibg, scheme.x, U, c)

@inline _advective_tracer_flux_y(i, j, k, ibg::IBG, scheme::FluxFormAdvection, V, c) =
        _advective_tracer_flux_y(i, j, k, ibg, scheme.y, V, c)

@inline _advective_tracer_flux_z(i, j, k, ibg::IBG, scheme::FluxFormAdvection, W, c) =
        _advective_tracer_flux_z(i, j, k, ibg, scheme.z, W, c)

#####
##### "Boundary-aware" reconstruct
#####
##### Don't reconstruct with immersed cells!
#####

"""
    inside_immersed_boundary(buffer, shift, dir, side;
                             xside = :ᶠ, yside = :ᶠ, zside = :ᶠ)

Check if the stencil required for reconstruction contains immersed nodes

Example
=======

```
julia> inside_immersed_boundary(2, :z, :ᶜ)
4-element Vector{Any}:
 :(inactive_node(i, j, k + -1, ibg, c, c, f))
 :(inactive_node(i, j, k + 0,  ibg, c, c, f))
 :(inactive_node(i, j, k + 1,  ibg, c, c, f))
 :(inactive_node(i, j, k + 2,  ibg, c, c, f))

julia> inside_immersed_boundary(3, :x, :ᶠ)
5-element Vector{Any}:
 :(inactive_node(i + -3, j, k, ibg, c, c, c))
 :(inactive_node(i + -2, j, k, ibg, c, c, c))
 :(inactive_node(i + -1, j, k, ibg, c, c, c))
 :(inactive_node(i + 0,  j, k, ibg, c, c, c))
 :(inactive_node(i + 1,  j, k, ibg, c, c, c))
```
"""
@inline function inside_immersed_boundary(buffer, dir, side)

    N = buffer * 2
    inactive_cells  = Vector(undef, N)

    xside = dir == :x ? side : Symbol("f")
    yside = dir == :y ? side : Symbol("f")
    zside = dir == :z ? side : Symbol("f")

    for (idx, n) in enumerate(1:N)
        c = side == :f ? n - buffer - 1 : n - buffer 
        xflipside = flip(xside)
        yflipside = flip(yside)
        zflipside = flip(zside)
        inactive_cells[idx] =  dir == :x ? 
                               :(immersed_inactive_node(i + $c, j, k, ibg, $xflipside, $yflipside, $zflipside)) :
                               dir == :y ?
                               :(immersed_inactive_node(i, j + $c, k, ibg, $xflipside, $yflipside, $zflipside)) :
                               :(immersed_inactive_node(i, j, k + $c, ibg, $xflipside, $yflipside, $zflipside))
    end

    return :($(inactive_cells...),)
end

flip(l) = ifelse(l == :f, :c, :f)

# For an immersed boundary grid, we compute the reduced order based on the inactive cells around the
# reconstruction interface (either face or center). 
#
# Below an example for an 10th (or 9th for upwind schemes) order reconstruction performed on interface `X`.
# Note that the buffer size is 5, and we represent reconstructions based on the buffer size, not the formal order.
# The check follows the following logic (for a symmetric stencil represented by a bias == NoBias):
#
# - if at least one between 1 or 10 are inactive, reduce from 5 to 4.
# - if at least one between 2 or  9 are inactive, reduce from 4 to 3.
# - if at least one between 3 or  8 are inactive, reduce from 3 to 2.
# - if at least one between 4 or  7 are inactive, reduce from 2 to 1.
#     
#      1     2     3     4     5  X  6     7     8     9    10   
#   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
#   |     |     |     |     └── 1st ────|     |     |     |     |
#   |     |     |     └── 2nd ─────────────── |     |     |     |
#   |     |     └── 3rd ────────────────────────────|     |     |
#   |     └── 4th ────────────────────────────────────────|     |
#   └── 5th ────────────────────────────────────────────────────|
#
# The same logic applies to biased stencils, with the only difference that we a biased stencil.
# For example, for a RightBias stencil, we have:
#
#      1     2     3     4     5  X  6     7     8     9    10   
#   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
#         |     |     |     └── 1st ────|     |     |     |     |
#         |     |     |     └── 2nd ──────────|     |     |     |
#         |     |     └── 3rd ──────────────────────|     |     |
#         |     └── 4th ──────────────────────────────────|     |
#         └── 5th ──────────────────────────────────────────────|
#   
for (Loc, loc) in zip((:face, :center), (:f, :c)), dir in (:x, :y, :z)
    compute_reduced_order = Symbol(:compute_, Loc,:_reduced_order_, dir)
    compute_immersed_reduced_order = Symbol(:compute_, Loc, :_immersed_reduced_order_, dir)

    @eval begin
        @inline function $compute_reduced_order(i, j, k, grid::IBG, a, bias)
            ior = $compute_immersed_reduced_order(i, j, k, grid, a, bias)
            bor = $compute_reduced_order(i, j, k, grid.underlying_grid, a, bias)
            return min(ior, bor)
        end
    end

    @eval begin 
        # Faces symmetric
        @inline $compute_immersed_reduced_order(i, j, k, ibg::IBG, ::A{1}, bias) = 1

        @inline function $compute_immersed_reduced_order(i, j, k, ibg::IBG, a::A{2}, bias) 
            I = $(inside_immersed_boundary(2, dir, loc))
            to1 = first_order_bounds_check(I, bias)
            return ifelse(to1, 1, 2) 
        end

        @inline function $compute_immersed_reduced_order(i, j, k, ibg::IBG, a::A{3}, bias) 
            I = $(inside_immersed_boundary(3, dir, loc))
            to2 = second_order_bounds_check(I, bias)
            to1 =  first_order_bounds_check(I, bias)
            return ifelse(to1, 1, 
                   ifelse(to2, 2, 3))
        end

        @inline function $compute_immersed_reduced_order(i, j, k, ibg::IBG, a::A{4}, bias) 
            I = $(inside_immersed_boundary(4, dir, loc))
            to3 =  third_order_bounds_check(I, bias)
            to2 = second_order_bounds_check(I, bias)
            to1 =  first_order_bounds_check(I, bias)
            return ifelse(to1, 1, 
                   ifelse(to2, 2, 
                   ifelse(to3, 3, 4)))
        end

        @inline function $compute_immersed_reduced_order(i, j, k, ibg::IBG, a::A{5}, bias) 
            I = $(inside_immersed_boundary(5, dir, loc))
            to4 = fourth_order_bounds_check(I, bias)
            to3 =  third_order_bounds_check(I, bias)
            to2 = second_order_bounds_check(I, bias)
            to1 =  first_order_bounds_check(I, bias)
            return ifelse(to1, 1, 
                   ifelse(to2, 2, 
                   ifelse(to3, 3, 
                   ifelse(to4, 4, 5))))
        end

        @inline function $compute_immersed_reduced_order(i, j, k, ibg::IBG, a::A{6}, bias) 
            I = $(inside_immersed_boundary(6, dir, loc))
            to5 =  fifth_order_bounds_check(I, bias)
            to4 = fourth_order_bounds_check(I, bias)
            to3 =  third_order_bounds_check(I, bias)
            to2 = second_order_bounds_check(I, bias)
            to1 =  first_order_bounds_check(I, bias)
            return ifelse(to1, 1, 
                   ifelse(to2, 2, 
                   ifelse(to3, 3, 
                   ifelse(to4, 4, 
                   ifelse(to5, 5, 6)))))
        end
    end
end

# NoBias immersed bounds checks
@inline first_order_bounds_check(I::NTuple{4}, ::NoBias) = @inbounds (I[1] | I[4])

@inline  first_order_bounds_check(I::NTuple{6}, ::NoBias) = @inbounds (I[2] | I[5])
@inline second_order_bounds_check(I::NTuple{6}, ::NoBias) = @inbounds (I[1] | I[6])

@inline  first_order_bounds_check(I::NTuple{8}, ::NoBias) = @inbounds (I[3] | I[6])
@inline second_order_bounds_check(I::NTuple{8}, ::NoBias) = @inbounds (I[2] | I[7])
@inline  third_order_bounds_check(I::NTuple{8}, ::NoBias) = @inbounds (I[1] | I[8])

@inline  first_order_bounds_check(I::NTuple{10}, ::NoBias) = @inbounds (I[4] | I[7])
@inline second_order_bounds_check(I::NTuple{10}, ::NoBias) = @inbounds (I[3] | I[8])
@inline  third_order_bounds_check(I::NTuple{10}, ::NoBias) = @inbounds (I[2] | I[9])
@inline fourth_order_bounds_check(I::NTuple{10}, ::NoBias) = @inbounds (I[1] | I[10])

@inline  first_order_bounds_check(I::NTuple{12}, ::NoBias) = @inbounds (I[5] | I[8])
@inline second_order_bounds_check(I::NTuple{12}, ::NoBias) = @inbounds (I[4] | I[9])
@inline  third_order_bounds_check(I::NTuple{12}, ::NoBias) = @inbounds (I[3] | I[10])
@inline fourth_order_bounds_check(I::NTuple{12}, ::NoBias) = @inbounds (I[2] | I[11])
@inline  fifth_order_bounds_check(I::NTuple{12}, ::NoBias) = @inbounds (I[1] | I[12])

# LeftBias immersed bounds checks
@inline first_order_bounds_check(I::NTuple{4}, ::LeftBias) = @inbounds (I[1] | I[4])

@inline  first_order_bounds_check(I::NTuple{6}, ::LeftBias) = @inbounds (I[2] | I[5])
@inline second_order_bounds_check(I::NTuple{6}, ::LeftBias) = @inbounds (I[1] | I[5])

@inline  first_order_bounds_check(I::NTuple{8}, ::LeftBias) = @inbounds (I[3] | I[6])
@inline second_order_bounds_check(I::NTuple{8}, ::LeftBias) = @inbounds (I[2] | I[6])
@inline  third_order_bounds_check(I::NTuple{8}, ::LeftBias) = @inbounds (I[1] | I[7])

@inline  first_order_bounds_check(I::NTuple{10}, ::LeftBias) = @inbounds (I[4] | I[7])
@inline second_order_bounds_check(I::NTuple{10}, ::LeftBias) = @inbounds (I[3] | I[7])
@inline  third_order_bounds_check(I::NTuple{10}, ::LeftBias) = @inbounds (I[2] | I[8])
@inline fourth_order_bounds_check(I::NTuple{10}, ::LeftBias) = @inbounds (I[1] | I[9])

@inline  first_order_bounds_check(I::NTuple{12}, ::LeftBias) = @inbounds (I[5] | I[8])
@inline second_order_bounds_check(I::NTuple{12}, ::LeftBias) = @inbounds (I[4] | I[8])
@inline  third_order_bounds_check(I::NTuple{12}, ::LeftBias) = @inbounds (I[3] | I[9])
@inline fourth_order_bounds_check(I::NTuple{12}, ::LeftBias) = @inbounds (I[2] | I[10])
@inline  fifth_order_bounds_check(I::NTuple{12}, ::LeftBias) = @inbounds (I[1] | I[11])

# RightBias immersed bounds checks
@inline first_order_bounds_check(I::NTuple{4}, ::RightBias) = @inbounds (I[1] | I[4])

@inline  first_order_bounds_check(I::NTuple{6}, ::RightBias) = @inbounds (I[2] | I[5])
@inline second_order_bounds_check(I::NTuple{6}, ::RightBias) = @inbounds (I[2] | I[6])

@inline  first_order_bounds_check(I::NTuple{8}, ::RightBias) = @inbounds (I[3] | I[6])
@inline second_order_bounds_check(I::NTuple{8}, ::RightBias) = @inbounds (I[3] | I[7])
@inline  third_order_bounds_check(I::NTuple{8}, ::RightBias) = @inbounds (I[2] | I[8])

@inline  first_order_bounds_check(I::NTuple{10}, ::RightBias) = @inbounds (I[4] | I[7])
@inline second_order_bounds_check(I::NTuple{10}, ::RightBias) = @inbounds (I[4] | I[8])
@inline  third_order_bounds_check(I::NTuple{10}, ::RightBias) = @inbounds (I[3] | I[9])
@inline fourth_order_bounds_check(I::NTuple{10}, ::RightBias) = @inbounds (I[2] | I[10])

@inline  first_order_bounds_check(I::NTuple{12}, ::RightBias) = @inbounds (I[5] | I[8])
@inline second_order_bounds_check(I::NTuple{12}, ::RightBias) = @inbounds (I[5] | I[9])
@inline  third_order_bounds_check(I::NTuple{12}, ::RightBias) = @inbounds (I[4] | I[10])
@inline fourth_order_bounds_check(I::NTuple{12}, ::RightBias) = @inbounds (I[3] | I[11])
@inline  fifth_order_bounds_check(I::NTuple{12}, ::RightBias) = @inbounds (I[2] | I[12])
