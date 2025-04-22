using Oceananigans.ImmersedBoundaries
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, inactive_node
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

# dx(uu), dy(vu), dz(wu)
# ccc,    ffc,    fcf
@inline _advective_momentum_flux_Uu(i, j, k, ibg::IBG, args...) = conditional_flux_ccc(i, j, k, ibg, zero(ibg), advective_momentum_flux_Uu(i, j, k, ibg, args...))
@inline _advective_momentum_flux_Vu(i, j, k, ibg::IBG, args...) = conditional_flux_ffc(i, j, k, ibg, zero(ibg), advective_momentum_flux_Vu(i, j, k, ibg, args...))
@inline _advective_momentum_flux_Wu(i, j, k, ibg::IBG, args...) = conditional_flux_fcf(i, j, k, ibg, zero(ibg), advective_momentum_flux_Wu(i, j, k, ibg, args...))

# dx(uv), dy(vv), dz(wv)
# ffc,    ccc,    cff
@inline _advective_momentum_flux_Uv(i, j, k, ibg::IBG, args...) = conditional_flux_ffc(i, j, k, ibg, zero(ibg), advective_momentum_flux_Uv(i, j, k, ibg, args...))
@inline _advective_momentum_flux_Vv(i, j, k, ibg::IBG, args...) = conditional_flux_ccc(i, j, k, ibg, zero(ibg), advective_momentum_flux_Vv(i, j, k, ibg, args...))
@inline _advective_momentum_flux_Wv(i, j, k, ibg::IBG, args...) = conditional_flux_cff(i, j, k, ibg, zero(ibg), advective_momentum_flux_Wv(i, j, k, ibg, args...))

# dx(uw), dy(vw), dz(ww)
# fcf,    cff,    ccc
@inline _advective_momentum_flux_Uw(i, j, k, ibg::IBG, args...) = conditional_flux_fcf(i, j, k, ibg, zero(ibg), advective_momentum_flux_Uw(i, j, k, ibg, args...))
@inline _advective_momentum_flux_Vw(i, j, k, ibg::IBG, args...) = conditional_flux_cff(i, j, k, ibg, zero(ibg), advective_momentum_flux_Vw(i, j, k, ibg, args...))
@inline _advective_momentum_flux_Ww(i, j, k, ibg::IBG, args...) = conditional_flux_ccc(i, j, k, ibg, zero(ibg), advective_momentum_flux_Ww(i, j, k, ibg, args...))

@inline _advective_tracer_flux_x(i, j, k, ibg::IBG, args...) = conditional_flux_fcc(i, j, k, ibg, zero(ibg), advective_tracer_flux_x(i, j, k, ibg, args...))
@inline _advective_tracer_flux_y(i, j, k, ibg::IBG, args...) = conditional_flux_cfc(i, j, k, ibg, zero(ibg), advective_tracer_flux_y(i, j, k, ibg, args...))
@inline _advective_tracer_flux_z(i, j, k, ibg::IBG, args...) = conditional_flux_ccf(i, j, k, ibg, zero(ibg), advective_tracer_flux_z(i, j, k, ibg, args...))

# Fallback for `nothing` advection
@inline _advective_tracer_flux_x(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_tracer_flux_y(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)
@inline _advective_tracer_flux_z(i, j, k, ibg::IBG, ::Nothing, args...) = zero(ibg)

# Disambiguation for `FluxForm` momentum fluxes....
@inline _advective_momentum_flux_Uu(i, j, k, ibg::IBG, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Uu(i, j, k, ibg, advection.x, args...) 
@inline _advective_momentum_flux_Vu(i, j, k, ibg::IBG, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Vu(i, j, k, ibg, advection.y, args...) 
@inline _advective_momentum_flux_Wu(i, j, k, ibg::IBG, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Wu(i, j, k, ibg, advection.z, args...) 

@inline _advective_momentum_flux_Uv(i, j, k, ibg::IBG, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Uv(i, j, k, ibg, advection.x, args...)
@inline _advective_momentum_flux_Vv(i, j, k, ibg::IBG, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Vv(i, j, k, ibg, advection.y, args...)
@inline _advective_momentum_flux_Wv(i, j, k, ibg::IBG, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Wv(i, j, k, ibg, advection.z, args...)

@inline _advective_momentum_flux_Uw(i, j, k, ibg::IBG, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Uw(i, j, k, ibg, advection.x, args...)
@inline _advective_momentum_flux_Vw(i, j, k, ibg::IBG, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Vw(i, j, k, ibg, advection.y, args...)
@inline _advective_momentum_flux_Ww(i, j, k, ibg::IBG, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Ww(i, j, k, ibg, advection.z, args...)

# Disambiguation for `FluxForm` tracer fluxes....
@inline _advective_tracer_flux_x(i, j, k, ibg::IBG, advection::FluxFormAdvection, args...) =
        _advective_tracer_flux_x(i, j, k, ibg, advection.x, args...)

@inline _advective_tracer_flux_y(i, j, k, ibg::IBG, advection::FluxFormAdvection, args...) =
        _advective_tracer_flux_y(i, j, k, ibg, advection.y, args...)

@inline _advective_tracer_flux_z(i, j, k, ibg::IBG, advection::FluxFormAdvection, args...) =
        _advective_tracer_flux_z(i, j, k, ibg, advection.z, args...)

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
julia> inside_immersed_boundary(2, :none, :z, :ᶜ)
4-element Vector{Any}:
 :(inactive_node(i, j, k + -1, ibg, c, c, f))
 :(inactive_node(i, j, k + 0,  ibg, c, c, f))
 :(inactive_node(i, j, k + 1,  ibg, c, c, f))
 :(inactive_node(i, j, k + 2,  ibg, c, c, f))

julia> inside_immersed_boundary(3, :left, :x, :ᶠ)
5-element Vector{Any}:
 :(inactive_node(i + -3, j, k, ibg, c, c, c))
 :(inactive_node(i + -2, j, k, ibg, c, c, c))
 :(inactive_node(i + -1, j, k, ibg, c, c, c))
 :(inactive_node(i + 0,  j, k, ibg, c, c, c))
 :(inactive_node(i + 1,  j, k, ibg, c, c, c))
```
"""
@inline function inside_immersed_boundary(buffer, dir, side;
                                          xside = :f, yside = :f, zside = :f)

    N = buffer * 2
    rng = 1:N+1

    inactive_cells  = Vector(undef, length(rng))

    for (idx, n) in enumerate(rng)
        c = side == :f ? n - buffer - 1 : n - buffer 
        xflipside = xside == :f ? :c : :f
        yflipside = yside == :f ? :c : :f
        zflipside = zside == :f ? :c : :f
        inactive_cells[idx] =  dir == :x ? 
                               :(inactive_node(i + $c, j, k, ibg, $xflipside, $yflipside, $zflipside)) :
                               dir == :y ?
                               :(inactive_node(i, j + $c, k, ibg, $xflipside, $yflipside, $zflipside)) :
                               :(inactive_node(i, j, k + $c, ibg, $xflipside, $yflipside, $zflipside))
    end

    return :($(inactive_cells...),)
end

# For an immersed boundary grid, we compute the reduced order based on the inactive cells around the
# reconstruction interface (either face or center). 
#
# Below an example for an 12th (or 11th for upwind schemes) order reconstruction performed on interface `X`.
# Note that the buffer size is 6, and we represent reconstructions based on the buffer size, not the formal order.
# The check follows the following logic:
#
# - if at least one between 1 or 12 are inactive, reduce from 6 to 5.
# - if at least one between 2 or 11 are inactive, reduce from 5 to 4.
# - if at least one between 3 or 10 are inactive, reduce from 4 to 3.
# - if at least one between 4 or  9 are inactive, reduce from 3 to 2.
# - if at least one between 5 or  8 are inactive, reduce from 2 to 1.
#     
#      1     2     3     4     5     6  X  7     8     9    10    11    12
#   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
#   |     |     |     |     |     └── 1st ────|     |     |     |     |     |
#   |     |     |     |     └── 2nd ─────────────── |     |     |     |     |   
#   |     |     |     └── 3rd ────────────────────────────|     |     |     |    
#   |     |     └── 4th ────────────────────────────────────────|     |     |
#   |     └── 5th ────────────────────────────────────────────────────|     |
#   └── 6th ────────────────────────────────────────────────────────────────|
#
#
for (Loc, loc) in zip((:face, :center), (:f, :c)), dir in (:x, :y, :z)
    compute_reduced_order = Symbol(:compute_, Loc,:_reduced_order_, dir)
    @eval begin 
        # Faces symmetric
        @inline $compute_reduced_order(i, j, k, ibg::IBG, ::A{1}) = 1

        @inline function $compute_reduced_order(i, j, k, ibg::IBG, ::A{2}) 
            I = $(inside_immersed_boundary(2, dir, loc; xside = loc))
            to1 = @inbounds (I[1] | I[4]) # Check only first and last
            return ifelse(to1, 1, 2) 
        end

        @inline function $compute_reduced_order(i, j, k, ibg::IBG, ::A{3}) 
            I = $(inside_immersed_boundary(3, dir, loc; xside = loc))
            to2 = @inbounds (I[1] | I[6])
            to1 = @inbounds (I[2] | I[5]) 
            return ifelse(to1, 1, 
                   ifelse(to2, 2, 3))
        end

        @inline function $compute_reduced_order(i, j, k, ibg::IBG, ::A{4}) 
            I = $(inside_immersed_boundary(4, dir, loc; xside = loc))
            to3 = @inbounds (I[1] | I[8])
            to2 = @inbounds (I[2] | I[7]) 
            to1 = @inbounds (I[3] | I[6])
            return ifelse(to1, 1, 
                   ifelse(to2, 2, 
                   ifelse(to3, 3, 4)))
        end

        @inline function $compute_reduced_order(i, j, k, ibg::IBG, ::A{5}) 
            I = $(inside_immersed_boundary(5, dir, loc; xside = loc))
            to4 = @inbounds (I[1] | I[10])
            to3 = @inbounds (I[2] | I[9])
            to2 = @inbounds (I[3] | I[8]) 
            to1 = @inbounds (I[4] | I[7])
            return ifelse(to1, 1, 
                   ifelse(to2, 2, 
                   ifelse(to3, 3, 
                   ifelse(to4, 4, 5))))
        end

        @inline function $compute_reduced_order(i, j, k, ibg::IBG, ::A{6}) 
            I = $(inside_immersed_boundary(6, dir, loc; xside = loc))
            to5 = @inbounds (I[1] | I[12])
            to4 = @inbounds (I[2] | I[11])
            to3 = @inbounds (I[3] | I[10])
            to2 = @inbounds (I[4] | I[9]) 
            to1 = @inbounds (I[5] | I[8])
            return ifelse(to1, 1, 
                   ifelse(to2, 2, 
                   ifelse(to3, 3, 
                   ifelse(to4, 4, 
                   ifelse(to5, 5, 6)))))
        end
    end
end
