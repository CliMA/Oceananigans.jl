using Oceananigans.Advection: AbstractAdvectionScheme
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑxᶜᵃᵃ, ℑyᵃᶠᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ 
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, AbstractTimeDiscretization

const ATC = AbstractTurbulenceClosure
const ATD = AbstractTimeDiscretization

"""
    conditional_flux(i, j, k, ibg::IBG, ℓx, ℓy, ℓz, qᴮ, qᴵ, nc)

Return either

    i) The boundary flux `qᴮ` if the node condition `nc` is true (default: `nc = peripheral_node`), or
    ii) The interior flux `qᴵ` otherwise.

This can be used either to condition intrinsic flux functions, or immersed boundary flux functions.
"""
@inline conditional_flux(i, j, k, ibg, ℓx, ℓy, ℓz, qᴮ, qᴵ) =
    ifelse(immersed_peripheral_node(i, j, k, ibg, ℓx, ℓy, ℓz), qᴮ, qᴵ)

# Conveniences
@inline conditional_flux_ccc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, c, c, qᴮ, qᴵ)
@inline conditional_flux_ffc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, f, f, c, qᴮ, qᴵ)
@inline conditional_flux_fcf(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, f, c, f, qᴮ, qᴵ)
@inline conditional_flux_cff(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, f, f, qᴮ, qᴵ)

@inline conditional_flux_fcc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, f, c, c, qᴮ, qᴵ)
@inline conditional_flux_cfc(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, f, c, qᴮ, qᴵ)
@inline conditional_flux_ccf(i, j, k, ibg::IBG, qᴮ, qᴵ) = conditional_flux(i, j, k, ibg, c, c, f, qᴮ, qᴵ)

#####
##### Diffusive fluxes
#####

# ccc, ffc, fcf
@inline _viscous_flux_ux(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, zero(eltype(ibg)), viscous_flux_ux(i, j, k, ibg, args...))
@inline _viscous_flux_uy(i, j, k, ibg::GFIBG, args...) = conditional_flux_ffc(i, j, k, ibg, zero(eltype(ibg)), viscous_flux_uy(i, j, k, ibg, args...))
@inline _viscous_flux_uz(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcf(i, j, k, ibg, zero(eltype(ibg)), viscous_flux_uz(i, j, k, ibg, args...))
 
 # ffc, ccc, cff
@inline _viscous_flux_vx(i, j, k, ibg::GFIBG, args...) = conditional_flux_ffc(i, j, k, ibg, zero(eltype(ibg)), viscous_flux_vx(i, j, k, ibg, args...))
@inline _viscous_flux_vy(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, zero(eltype(ibg)), viscous_flux_vy(i, j, k, ibg, args...))
@inline _viscous_flux_vz(i, j, k, ibg::GFIBG, args...) = conditional_flux_cff(i, j, k, ibg, zero(eltype(ibg)), viscous_flux_vz(i, j, k, ibg, args...))

 # fcf, cff, ccc
@inline _viscous_flux_wx(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcf(i, j, k, ibg, zero(eltype(ibg)), viscous_flux_wx(i, j, k, ibg, args...))
@inline _viscous_flux_wy(i, j, k, ibg::GFIBG, args...) = conditional_flux_cff(i, j, k, ibg, zero(eltype(ibg)), viscous_flux_wy(i, j, k, ibg, args...))
@inline _viscous_flux_wz(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, zero(eltype(ibg)), viscous_flux_wz(i, j, k, ibg, args...))

# fcc, cfc, ccf
@inline _diffusive_flux_x(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcc(i, j, k, ibg, zero(eltype(ibg)), diffusive_flux_x(i, j, k, ibg, args...))
@inline _diffusive_flux_y(i, j, k, ibg::GFIBG, args...) = conditional_flux_cfc(i, j, k, ibg, zero(eltype(ibg)), diffusive_flux_y(i, j, k, ibg, args...))
@inline _diffusive_flux_z(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccf(i, j, k, ibg, zero(eltype(ibg)), diffusive_flux_z(i, j, k, ibg, args...))

#####
##### Advective fluxes
#####

# dx(uu), dy(vu), dz(wu)
# ccc,    ffc,    fcf
@inline _advective_momentum_flux_Uu(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, zero(eltype(ibg)), advective_momentum_flux_Uu(i, j, k, ibg, args...))
@inline _advective_momentum_flux_Vu(i, j, k, ibg::GFIBG, args...) = conditional_flux_ffc(i, j, k, ibg, zero(eltype(ibg)), advective_momentum_flux_Vu(i, j, k, ibg, args...))
@inline _advective_momentum_flux_Wu(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcf(i, j, k, ibg, zero(eltype(ibg)), advective_momentum_flux_Wu(i, j, k, ibg, args...))

# dx(uv), dy(vv), dz(wv)
# ffc,    ccc,    cff
@inline _advective_momentum_flux_Uv(i, j, k, ibg::GFIBG, args...) = conditional_flux_ffc(i, j, k, ibg, zero(eltype(ibg)), advective_momentum_flux_Uv(i, j, k, ibg, args...))
@inline _advective_momentum_flux_Vv(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, zero(eltype(ibg)), advective_momentum_flux_Vv(i, j, k, ibg, args...))
@inline _advective_momentum_flux_Wv(i, j, k, ibg::GFIBG, args...) = conditional_flux_cff(i, j, k, ibg, zero(eltype(ibg)), advective_momentum_flux_Wv(i, j, k, ibg, args...))

# dx(uw), dy(vw), dz(ww)
# fcf,    cff,    ccc
@inline _advective_momentum_flux_Uw(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcf(i, j, k, ibg, zero(eltype(ibg)), advective_momentum_flux_Uw(i, j, k, ibg, args...))
@inline _advective_momentum_flux_Vw(i, j, k, ibg::GFIBG, args...) = conditional_flux_cff(i, j, k, ibg, zero(eltype(ibg)), advective_momentum_flux_Vw(i, j, k, ibg, args...))
@inline _advective_momentum_flux_Ww(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, zero(eltype(ibg)), advective_momentum_flux_Ww(i, j, k, ibg, args...))

@inline _advective_tracer_flux_x(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcc(i, j, k, ibg, zero(eltype(ibg)), advective_tracer_flux_x(i, j, k, ibg, args...))
@inline _advective_tracer_flux_y(i, j, k, ibg::GFIBG, args...) = conditional_flux_cfc(i, j, k, ibg, zero(eltype(ibg)), advective_tracer_flux_y(i, j, k, ibg, args...))
@inline _advective_tracer_flux_z(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccf(i, j, k, ibg, zero(eltype(ibg)), advective_tracer_flux_z(i, j, k, ibg, args...))

#####
##### "Boundary-aware" reconstruct
#####
##### Don't reconstruct with immersed cells!
#####

"""
    calc_inactive_stencil(buffer, shift, dir, side;
                          xside = :ᶠ, yside = :ᶠ, zside = :ᶠ,
                          xshift = 0, yshift = 0, zshift = 0) 

Calculate the correct stencil needed for each indiviual reconstruction (i.e., symmetric, left biased and right biased, 
on `Face`s and on `Center`s)

Example
=======

```
julia> calc_inactive_cells(2, :none, :z, :ᶜ)
4-element Vector{Any}:
 :(immersed_inactive_node(i, j, k + -1, ibg, c, c, f))
 :(immersed_inactive_node(i, j, k + 0,  ibg, c, c, f))
 :(immersed_inactive_node(i, j, k + 1,  ibg, c, c, f))
 :(immersed_inactive_node(i, j, k + 2,  ibg, c, c, f))

julia> calc_inactive_cells(3, :left, :x, :ᶠ)
5-element Vector{Any}:
 :(immersed_inactive_node(i + -3, j, k, ibg, c, c, c))
 :(immersed_inactive_node(i + -2, j, k, ibg, c, c, c))
 :(immersed_inactive_node(i + -1, j, k, ibg, c, c, c))
 :(immersed_inactive_node(i + 0,  j, k, ibg, c, c, c))
 :(immersed_inactive_node(i + 1,  j, k, ibg, c, c, c))
```
"""
@inline function calc_inactive_stencil(buffer, shift, dir, side;
                                       xside = :ᶠ, yside = :ᶠ, zside = :ᶠ,
                                       xshift = 0, yshift = 0, zshift = 0)

    N = buffer * 2
    if shift != :none
        N -=1
    end
    inactive_cells  = Vector(undef, N)

    rng = 1:N
    if shift == :right
        rng = rng .+ 1
    end

    for (idx, n) in enumerate(rng)
        c = side == :ᶠ ? n - buffer - 1 : n - buffer 
        xflipside = xside == :ᶠ ? :c : :f
        yflipside = yside == :ᶠ ? :c : :f
        zflipside = zside == :ᶠ ? :c : :f
        inactive_cells[idx] =  dir == :x ? 
                               :(immersed_inactive_node(i + $(c + xshift), j + $yshift, k + $zshift, ibg, $xflipside, $yflipside, $zflipside)) :
                               dir == :y ?
                               :(immersed_inactive_node(i + $xshift, j + $(c + yshift), k + $zshift, ibg, $xflipside, $yflipside, $zflipside)) :
                               :(immersed_inactive_node(i + $xshift, j + $yshift, k + $(c + zshift), ibg, $xflipside, $yflipside, $zflipside))
    end

    return inactive_cells
end

for (bias, shift) in zip((:symmetric, :left_biased, :right_biased), (:none, :left, :right)), side in (:ᶜ, :ᶠ)
    near_x_boundary = Symbol(:near_x_immersed_boundary_, bias, side)
    near_y_boundary = Symbol(:near_y_immersed_boundary_, bias, side)
    near_z_boundary = Symbol(:near_z_immersed_boundary_, bias, side)

    @eval begin
        @inline $near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
        @inline $near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
        @inline $near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
    end

    for buffer in [1, 2, 3, 4, 5, 6]
        @eval begin
            @inline $near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = @inbounds (|)($(calc_inactive_stencil(buffer, shift, :x, side; xside = side)...))
            @inline $near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = @inbounds (|)($(calc_inactive_stencil(buffer, shift, :y, side; yside = side)...))
            @inline $near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = @inbounds (|)($(calc_inactive_stencil(buffer, shift, :z, side; zside = side)...))
        end
    end
end

# Horizontal inactive stencil calculation for vector invariant WENO schemes that use velocity as a smoothness indicator
for (bias, shift) in zip((:symmetric, :left_biased, :right_biased), (:none, :left, :right))
    near_x_horizontal_boundary = Symbol(:near_x_horizontal_boundary_, bias)
    near_y_horizontal_boundary = Symbol(:near_y_horizontal_boundary_, bias)
    
    for buffer in [1, 2, 3, 4, 5, 6]
        @eval begin
            @inline $near_x_horizontal_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = 
                @inbounds (|)($(calc_inactive_stencil(buffer+1, shift, :x, :ᶜ; yside = :ᶜ)...), 
                              $(calc_inactive_stencil(buffer,   shift, :x, :ᶜ; xside = :ᶜ)...), 
                              $(calc_inactive_stencil(buffer,   shift, :x, :ᶜ; xside = :ᶜ, yshift = 1)...))

            @inline $near_y_horizontal_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) =
                @inbounds (|)($(calc_inactive_stencil(buffer+1, shift, :y, :ᶜ; xside = :ᶜ)...),
                              $(calc_inactive_stencil(buffer,   shift, :y, :ᶜ; yside = :ᶜ)...),
                              $(calc_inactive_stencil(buffer,   shift, :y, :ᶜ; yside = :ᶜ, xshift = 1)...))
        end
    end
end

using Oceananigans.Advection: LOADV, HOADV, WENO
using Oceananigans.Advection: AbstractSmoothnessStencil, VelocityStencil, DefaultStencil

for bias in (:symmetric, :left_biased, :right_biased)
    for (d, ξ) in enumerate((:x, :y, :z))

        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ)
            code[d] = loc
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt_interp = Symbol(:_, interp)

            near_boundary = Symbol(:near_, ξ, :_immersed_boundary_, bias, loc)

            @eval begin
                import Oceananigans.Advection: $alt_interp
                using Oceananigans.Advection: $interp

                # Fallback for low order interpolation
                @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::LOADV, args...) = $alt_interp(i, j, k, ibg.underlying_grid, scheme, args...)

                # Conditional high-order interpolation in Bounded directions
                @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::HOADV, args...) =
                    ifelse($near_boundary(i, j, k, ibg, scheme),
                           $alt_interp(i, j, k, ibg, scheme.buffer_scheme, args...),
                           $alt_interp(i, j, k, ibg.underlying_grid, scheme, args...))
                            
                # Conditional high-order interpolation for Vector Invariant WENO in Bounded directions
                @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENO, ζ, VI::AbstractSmoothnessStencil, args...) =
                    ifelse($near_boundary(i, j, k, ibg, scheme),
                            $alt_interp(i, j, k, ibg, scheme.buffer_scheme, ζ, VI, args...),
                            $alt_interp(i, j, k, ibg.underlying_grid, scheme, ζ, VI, args...))
            end
        end
    end
end

for bias in (:left_biased, :right_biased)
    for (d, dir) in zip((:x, :y), (:xᶜᵃᵃ, :yᵃᶜᵃ))
        interp     = Symbol(bias, :_interpolate_, dir)
        alt_interp = Symbol(:_, interp)

        near_horizontal_boundary = Symbol(:near_, d, :_horizontal_boundary_, bias)

        @eval begin
            # Conditional Interpolation for VelocityStencil WENO vector invariant scheme
            @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENO, ζ, ::VelocityStencil, args...) =
                ifelse($near_horizontal_boundary(i, j, k, ibg, scheme),
                    $alt_interp(i, j, k, ibg, scheme, ζ, DefaultStencil(), args...),
                    $alt_interp(i, j, k, ibg.underlying_grid, scheme, ζ, VelocityStencil(), args...))
        end
    end
end
