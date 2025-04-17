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
@inline function inside_immersed_boundary(buffer, shift, dir, side;
                                          xside = :f, yside = :f, zside = :f)

    N = buffer * 2
    if shift != :none
        N -=1
    end

    if shift == :interior
        rng = 1:N+1
    elseif shift == :right
        rng = 2:N+1
    else
        rng = 1:N
    end

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

    return inactive_cells
end

for (Loc, loc) in zip((:face, :center), (:f, :c)), dir in (:x, :y, :z)
    compute_reduced_order = Symbol(:compute_, Loc,:_reduced_order_, dir)
    @eval begin 
        # Faces symmetric
        @inline $compute_reduced_order(i, j, k, ibg::IBG, ::CenteredScheme{1}) = 1

        @inline function $compute_reduced_order(i, j, k, ibg::IBG, ::CenteredScheme{2}) 
            I = $(inside_immersed_boundary(2, :none, dir, loc; xside = loc)...)
            to1 = @inbounds (I[1] | I[2]) # Check only first and last
            return ifelse(to1, 1, 2) 
        end

        @inline function $compute_reduced_order(i, j, k, ibg::IBG, ::CenteredScheme{3}) 
            I = $(inside_immersed_boundary(3, :none, dir, loc; xside = loc)...)
            to2 = @inbounds (I[1] | I[4])
            to1 = @inbounds (I[2] | I[3]) 
            return ifelse(to1, 1, ifelse(to2, 2, 3))
        end

        @inline function $compute_reduced_order(i, j, k, ibg::IBG, ::CenteredScheme{4}) 
            I = $(inside_immersed_boundary(3, :none, dir, loc; xside = loc)...)
            to3 = @inbounds (I[1] | I[6])
            to2 = @inbounds (I[2] | I[5]) 
            to1 = @inbounds (I[3] | I[4])
            return ifelse(to1, 1, ifelse(to2, 2, ifelse(to3, 3, 4)))
        end

        @inline function $compute_reduced_order(i, j, k, ibg::IBG, ::CenteredScheme{5}) 
            I = $(inside_immersed_boundary(3, :none, dir, loc; xside = loc)...)
            to4 = @inbounds (I[1] | I[8])
            to3 = @inbounds (I[2] | I[7])
            to2 = @inbounds (I[3] | I[6]) 
            to1 = @inbounds (I[4] | I[5])
            return ifelse(to1, 1, ifelse(to2, 2, ifelse(to3, 3, ifelse(to4, 4, 5))))
        end

        @inline function $compute_reduced_order(i, j, k, ibg::IBG, ::CenteredScheme{6}) 
            I = $(inside_immersed_boundary(3, :none, dir, loc; xside = loc)...)
            to5 = @inbounds (I[1] | I[10])
            to4 = @inbounds (I[2] | I[9])
            to3 = @inbounds (I[3] | I[8])
            to2 = @inbounds (I[4] | I[7]) 
            to1 = @inbounds (I[5] | I[6])
            return ifelse(to1, 1, ifelse(to2, 2, ifelse(to3, 3, ifelse(to4, 4, ifelse(to5, 5, 6)))))
        end

        # # Faces biased
        # @inline function compute_face_reduced_order_x(i, j, k, ibg::IBG, ::UpwindScheme{$B})
        #     return $B # - min(sum($(inside_immersed_boundary(B, :interior, :x, :f; xside = :f))) ÷ 2, $B-1)
        # end

        # @inline function compute_face_reduced_order_y(i, j, k, ibg::IBG, ::UpwindScheme{$B}) 
        #     return $B # - min(sum($(inside_immersed_boundary(B, :interior, :y, :f; yside = :f))) ÷ 2, $B-1)
        # end

        # @inline function compute_face_reduced_order_z(i, j, k, ibg::IBG, ::UpwindScheme{$B}) 
        #     return $B # - min(sum($(inside_immersed_boundary(B, :interior, :z, :f; zside = :f))) ÷ 2, $B-1)
        # end

        # # Centers biased
        # @inline function compute_center_reduced_order_x(i, j, k, ibg::IBG, ::UpwindScheme{$B}) 
        #     return $B # - min(sum($(inside_immersed_boundary(B, :interior, :x, :c; xside = :c))) ÷ 2, $B-1)
        # end

        # @inline function compute_center_reduced_order_y(i, j, k, ibg::IBG, ::UpwindScheme{$B})
        #     return $B # - min(sum($(inside_immersed_boundary(B, :interior, :y, :c; yside = :c))) ÷ 2, $B-1)
        # end

        # @inline function compute_center_reduced_order_z(i, j, k, ibg::IBG, ::UpwindScheme{$B})
        #     return $B # - min(sum($(inside_immersed_boundary(B, :interior, :z, :c; zside = :c))) ÷ 2, $B-1)
        # end
    end
end
