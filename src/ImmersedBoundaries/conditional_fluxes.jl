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
    Calculate the correct stencil needed for each indiviual reconstruction (i.e., symmetric, left biased and right biased, 
on `Face`s and on `Center`s)

example

julia> calc_inactive_cells(2, :none, :z, :ᶜ)
4-element Vector{Any}:
 :(inactive_node(i, j, k + -1, ibg, c, c, f))
 :(inactive_node(i, j, k + 0,  ibg, c, c, f))
 :(inactive_node(i, j, k + 1,  ibg, c, c, f))
 :(inactive_node(i, j, k + 2,  ibg, c, c, f))

julia> calc_inactive_cells(3, :left, :x, :ᶠ)
5-element Vector{Any}:
 :(inactive_node(i + -3, j, k, ibg, c, c, c))
 :(inactive_node(i + -2, j, k, ibg, c, c, c))
 :(inactive_node(i + -1, j, k, ibg, c, c, c))
 :(inactive_node(i + 0,  j, k, ibg, c, c, c))
 :(inactive_node(i + 1,  j, k, ibg, c, c, c))

"""
@inline function calc_inactive_stencil(buffer, shift, dir, side; xside = :ᶠ, yside = :ᶠ, zside = :ᶠ, xshift = 0, yshift = 0, zshift = 0) 
   
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
                               :(inactive_node(i + $(c + xshift), j + $yshift, k + $zshift, ibg, $xflipside, $yflipside, $zflipside)) :
                               dir == :y ?
                               :(inactive_node(i + $xshift, j + $(c + yshift), k + $zshift, ibg, $xflipside, $yflipside, $zflipside)) :
                               :(inactive_node(i + $xshift, j + $yshift, k + $(c + zshift), ibg, $xflipside, $yflipside, $zflipside))                    
    end

    return inactive_cells
end

for side in (:ᶜ, :ᶠ)
    near_x_boundary = Symbol(:near_x_immersed_boundary_symmetric, side)
    near_y_boundary = Symbol(:near_y_immersed_boundary_symmetric, side)
    near_z_boundary = Symbol(:near_z_immersed_boundary_symmetric, side)

    @eval begin
        @inline $near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
        @inline $near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
        @inline $near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
    end

    for buffer in [1, 2, 3, 4, 5, 6]
        @eval begin
            @inline $near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = @inbounds (|)($(calc_inactive_stencil(buffer, :none, :x, side; xside = side)...))
            @inline $near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = @inbounds (|)($(calc_inactive_stencil(buffer, :none, :y, side; yside = side)...))
            @inline $near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}) = @inbounds (|)($(calc_inactive_stencil(buffer, :none, :z, side; zside = side)...))
        end
    end
end

for side in (:ᶜ, :ᶠ)
    near_x_boundary = Symbol(:near_x_immersed_boundary_biased, side)
    near_y_boundary = Symbol(:near_y_immersed_boundary_biased, side)
    near_z_boundary = Symbol(:near_z_immersed_boundary_biased, side)

    @eval begin
        @inline $near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}, ::Val{D}) where D = false
        @inline $near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}, ::Val{D}) where D = false
        @inline $near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}, ::Val{D}) where D = false
    end

    for buffer in [1, 2, 3, 4, 5, 6]
        @eval begin
            @inline $near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, ::Val{:left})  = @inbounds (|)($(calc_inactive_stencil(buffer,  :left, :x, side; xside = side)...))
            @inline $near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, ::Val{:left})  = @inbounds (|)($(calc_inactive_stencil(buffer,  :left, :y, side; yside = side)...))
            @inline $near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, ::Val{:left})  = @inbounds (|)($(calc_inactive_stencil(buffer,  :left, :z, side; zside = side)...))
            @inline $near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, ::Val{:right}) = @inbounds (|)($(calc_inactive_stencil(buffer, :right, :x, side; xside = side)...))
            @inline $near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, ::Val{:right}) = @inbounds (|)($(calc_inactive_stencil(buffer, :right, :y, side; yside = side)...))
            @inline $near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, ::Val{:right}) = @inbounds (|)($(calc_inactive_stencil(buffer, :right, :z, side; zside = side)...))
        end
    end
end

# Horizontal inactive stencil calculation for vector invariant WENO schemes that use velocity as a smoothness indicator
for buffer in [1, 2, 3, 4, 5, 6]
    @eval begin
        @inline near_x_horizontal_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, ::Val{:left}) = 
            @inbounds (|)($(calc_inactive_stencil(buffer+1, :left, :x, :ᶜ; yside = :ᶜ)...), 
                          $(calc_inactive_stencil(buffer,   :left, :x, :ᶜ; xside = :ᶜ)...), 
                          $(calc_inactive_stencil(buffer,   :left, :x, :ᶜ; xside = :ᶜ, yshift = 1)...))

        @inline near_y_horizontal_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, ::Val{:left}) = 
            @inbounds (|)($(calc_inactive_stencil(buffer+1, :left, :y, :ᶜ; xside = :ᶜ)...), 
                          $(calc_inactive_stencil(buffer,   :left, :y, :ᶜ; yside = :ᶜ)...), 
                          $(calc_inactive_stencil(buffer,   :left, :y, :ᶜ; yside = :ᶜ, xshift = 1)...))

        @inline near_x_horizontal_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, ::Val{:right}) = 
            @inbounds (|)($(calc_inactive_stencil(buffer+1, :right, :x, :ᶜ; yside = :ᶜ)...), 
                          $(calc_inactive_stencil(buffer,   :right, :x, :ᶜ; xside = :ᶜ)...), 
                          $(calc_inactive_stencil(buffer,   :right, :x, :ᶜ; xside = :ᶜ, yshift = 1)...))

        @inline near_y_horizontal_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{$buffer}, ::Val{:right}) = 
            @inbounds (|)($(calc_inactive_stencil(buffer+1, :right, :y, :ᶜ; xside = :ᶜ)...), 
                          $(calc_inactive_stencil(buffer,   :right, :y, :ᶜ; yside = :ᶜ)...), 
                          $(calc_inactive_stencil(buffer,   :right, :y, :ᶜ; yside = :ᶜ, xshift = 1)...))
    end
end

using Oceananigans.Advection: LOADV, HOADV, WENO
using Oceananigans.Advection: AbstractSmoothnessStencil, VelocityStencil, DefaultStencil

for (d, ξ) in enumerate((:x, :y, :z))

    code = [:ᵃ, :ᵃ, :ᵃ]

    for loc in (:ᶜ, :ᶠ)
        code[d] = loc
        interp = Symbol(:symmetric_interpolate_, ξ, code...)
        alt_interp = Symbol(:_, interp)

        near_boundary = Symbol(:near_, ξ, :_immersed_boundary_symmetric, loc)

        @eval begin
            import Oceananigans.Advection: $alt_interp
            using Oceananigans.Advection: $interp

            # Fallback for low order interpolation
            @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::LOADV, args...) = $interp(i, j, k, ibg, scheme, args...)

            # Conditional high-order interpolation in Bounded directions
            @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::HOADV, args...) =
                ifelse($near_boundary(i, j, k, ibg, scheme),
                        $alt_interp(i, j, k, ibg, scheme.buffer_scheme, args...),
                        $interp(i, j, k, ibg, scheme, args...))
        end    
    end
end

# TODO: Change back to Val{D}

for (d, ξ) in enumerate((:x, :y, :z))

    code = [:ᵃ, :ᵃ, :ᵃ]

    for loc in (:ᶜ, :ᶠ)
        code[d] = loc
        interp = Symbol(:biased_interpolate_, ξ, code...)
        alt_interp = Symbol(:_, interp)

        near_boundary = Symbol(:near_, ξ, :_immersed_boundary_biased, loc)

        @eval begin
            import Oceananigans.Advection: $alt_interp
            using Oceananigans.Advection: $interp

            # Fallback for low order interpolation
            @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::LOADV, ::Val{:left},  args...) = $interp(i, j, k, ibg, scheme, Val(:left),  args...)
            @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::LOADV, ::Val{:right}, args...) = $interp(i, j, k, ibg, scheme, Val(:right), args...)

            # Conditional high-order interpolation in Bounded directions
            @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::HOADV, ::Val{:left}, args...) =
                ifelse($near_boundary(i, j, k, ibg, scheme, Val(:left)),
                        $alt_interp(i, j, k, ibg, scheme.buffer_scheme, Val(:left), args...),
                        $interp(i, j, k, ibg, scheme, Val(:left), args...))
            
            # Conditional high-order interpolation for Vector Invariant WENO in Bounded directions
            @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENO, ::Val{:left}, ζ, VI::AbstractSmoothnessStencil, u, v) =
                ifelse($near_boundary(i, j, k, ibg, scheme, Val(:left)),
                        $alt_interp(i, j, k, ibg, scheme.buffer_scheme, Val(:left), ζ, VI, u, v),
                        $interp(i, j, k, ibg, scheme, Val(:left), ζ, VI, u, v))

            # Conditional high-order interpolation in Bounded directions
            @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::HOADV, ::Val{:right}, args...) =
                ifelse($near_boundary(i, j, k, ibg, scheme, Val(:right)),
                        $alt_interp(i, j, k, ibg, scheme.buffer_scheme, Val(:right), args...),
                        $interp(i, j, k, ibg, scheme, Val(:right), args...))
        
            # Conditional high-order interpolation for Vector Invariant WENO in Bounded directions
            @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENO, ::Val{:right}, ζ, VI::AbstractSmoothnessStencil, u, v) =
                ifelse($near_boundary(i, j, k, ibg, scheme, Val(:right)),
                        $alt_interp(i, j, k, ibg, scheme.buffer_scheme, Val(:right), ζ, VI, u, v),
                        $interp(i, j, k, ibg, scheme, Val(:right), ζ, VI, u, v))
        end    
    end
end

for (d, dir) in zip((:x, :y), (:xᶜᵃᵃ, :yᵃᶜᵃ))
    interp     = Symbol(:biased_interpolate_, dir)
    alt_interp = Symbol(:_, interp)

    near_horizontal_boundary = Symbol(:near_, d, :_horizontal_boundary)

    @eval begin
        # Conditional Interpolation for VelocityStencil WENO vector invariant scheme
        @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENO, ::Val{:left}, ζ, ::VelocityStencil, u, v) =
            ifelse($near_horizontal_boundary(i, j, k, ibg, scheme, Val(:left)),
                $alt_interp(i, j, k, ibg, scheme, Val(:left), ζ, DefaultStencil(), u, v),
                $interp(i, j, k, ibg, scheme, Val(:left), ζ, VelocityStencil(), u, v))

                        # Conditional Interpolation for VelocityStencil WENO vector invariant scheme
        @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENO, ::Val{:right}, ζ, ::VelocityStencil, u, v) =
            ifelse($near_horizontal_boundary(i, j, k, ibg, scheme, Val(:right)),
                $alt_interp(i, j, k, ibg, scheme, Val(:right), ζ, DefaultStencil(), u, v),
                $interp(i, j, k, ibg, scheme, Val(:right), ζ, VelocityStencil(), u, v))
    end
end
