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
                                          xside = :ᶠ, yside = :ᶠ, zside = :ᶠ)

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
        c = side == :ᶠ ? n - buffer - 1 : n - buffer 
        xflipside = xside == :ᶠ ? :c : :f
        yflipside = yside == :ᶠ ? :c : :f
        zflipside = zside == :ᶠ ? :c : :f
        inactive_cells[idx] =  dir == :x ? 
                               :(inactive_node(i + $c, j, k, ibg, $xflipside, $yflipside, $zflipside)) :
                               dir == :y ?
                               :(inactive_node(i, j + $c, k, ibg, $xflipside, $yflipside, $zflipside)) :
                               :(inactive_node(i, j, k + $c, ibg, $xflipside, $yflipside, $zflipside))
    end

    return inactive_cells
end

for side in (:ᶜ, :ᶠ)
    near_x_boundary_symm = Symbol(:near_x_immersed_boundary_symmetric, side)
    near_y_boundary_symm = Symbol(:near_y_immersed_boundary_symmetric, side)
    near_z_boundary_symm = Symbol(:near_z_immersed_boundary_symmetric, side)

    near_x_boundary_bias = Symbol(:near_x_immersed_boundary_biased, side)
    near_y_boundary_bias = Symbol(:near_y_immersed_boundary_biased, side)
    near_z_boundary_bias = Symbol(:near_z_immersed_boundary_biased, side)

    @eval begin
        @inline $near_x_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{0}, args...) = false
        @inline $near_y_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{0}, args...) = false
        @inline $near_z_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{0}, args...) = false

        @inline $near_x_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{0}, args...) = false
        @inline $near_y_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{0}, args...) = false
        @inline $near_z_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{0}, args...) = false
    end

    for buffer in advection_buffers
        @eval begin
            @inline $near_x_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(buffer, :none, :x, side; xside = side)...))
            @inline $near_y_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(buffer, :none, :y, side; yside = side)...))
            @inline $near_z_boundary_symm(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(buffer, :none, :z, side; zside = side)...))
    
            @inline $near_x_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(buffer, :interior, :x, side; xside = side)...))
            @inline $near_y_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(buffer, :interior, :y, side; yside = side)...))
            @inline $near_z_boundary_bias(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = (|)($(inside_immersed_boundary(buffer, :interior, :z, side; zside = side)...))
        end
    end
end

for bias in (:symmetric, :biased)
    for (d, ξ) in enumerate((:x, :y, :z))
        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ), alt in (:_, :__, :___, :____, :_____)
            code[d] = loc
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt_interp = Symbol(alt, interp)
            @eval begin
                import Oceananigans.Advection: $alt_interp
                using Oceananigans.Advection: $interp
            end
        end

        for loc in (:ᶜ, :ᶠ), (alt1, alt2) in zip((:_, :__, :___, :____, :_____), (:_____, :_, :__, :___, :____))
            code[d] = loc
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt1_interp = Symbol(alt1, interp)
            alt2_interp = Symbol(alt2, interp)

            near_boundary = Symbol(:near_, ξ, :_immersed_boundary_, bias, loc)

            @eval begin
                # Fallback for low order interpolation
                @inline $alt1_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::LOADV, args...) = $interp(i, j, k, ibg, scheme, args...)
                
                # Conditional high-order interpolation in Bounded directions
                @inline $alt1_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::HOADV, args...) =
                    ifelse($near_boundary(i, j, k, ibg, scheme),
                           $alt2_interp(i, j, k, ibg, scheme.buffer_scheme, args...),
                           $interp(i, j, k, ibg, scheme, args...))
            end
        end
    end
end
