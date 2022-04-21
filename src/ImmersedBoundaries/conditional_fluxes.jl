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
    ifelse(immersed_peripheral_node(ℓx, ℓy, ℓz, i, j, k, ibg), qᴮ, qᴵ)

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

@inline _advective_tracer_flux_x(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcc(i, j, k, ibg, zero(eltype(ibg)), advective_tracer_flux_x(i, j, ibg, args...))
@inline _advective_tracer_flux_y(i, j, k, ibg::GFIBG, args...) = conditional_flux_cfc(i, j, k, ibg, zero(eltype(ibg)), advective_tracer_flux_y(i, j, ibg, args...))
@inline _advective_tracer_flux_z(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccf(i, j, k, ibg, zero(eltype(ibg)), advective_tracer_flux_z(i, j, ibg, args...))

#####
##### "Boundary-aware" reconstruct
#####
##### Don't reconstruct with immersed cells!
#####

@inline near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
@inline near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
@inline near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false

@inline near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{1}) = inactive_cell(i-1, j, k, ibg) | inactive_cell(i, j, k, ibg) | inactive_cell(i+1, j, k, ibg)
@inline near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{1}) = inactive_cell(i, j-1, k, ibg) | inactive_cell(i, j, k, ibg) | inactive_cell(i, j+1, k, ibg)
@inline near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{1}) = inactive_cell(i, j, k-1, ibg) | inactive_cell(i, j, k, ibg) | inactive_cell(i, j, k+1, ibg)

@inline near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{2}) = inactive_cell(i-2, j, k, ibg) | inactive_cell(i-1, j, k, ibg) | inactive_cell(i, j, k, ibg) | inactive_cell(i+1, j, k, ibg) | inactive_cell(i+2, j, k, ibg)
@inline near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{2}) = inactive_cell(i, j-2, k, ibg) | inactive_cell(i, j-1, k, ibg) | inactive_cell(i, j, k, ibg) | inactive_cell(i, j+1, k, ibg) | inactive_cell(i, j+2, k, ibg)
@inline near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{2}) = inactive_cell(i, j, k-2, ibg) | inactive_cell(i, j, k-1, ibg) | inactive_cell(i, j, k, ibg) | inactive_cell(i, j, k+1, ibg) | inactive_cell(i, j, k+2, ibg)

using Oceananigans.Advection: WENOVectorInvariantVel, VorticityStencil, VelocityStencil

@inline function near_horizontal_boundary_x(i, j, k, ibg, scheme::WENOVectorInvariantVel) 
    return inactive_node(c, f, c, i, j, k, ibg)     |       
           inactive_node(c, f, c, i-3, j, k, ibg)   | inactive_node(c, f, c, i+3, j, k, ibg) |
           inactive_node(c, f, c, i-2, j, k, ibg)   | inactive_node(c, f, c, i+2, j, k, ibg) |
           inactive_node(c, f, c, i-1, j, k, ibg)   | inactive_node(c, f, c, i+1, j, k, ibg) | 
           inactive_node(f, c, c, i, j, k, ibg)     |
           inactive_node(f, c, c, i-2, j, k, ibg)   | inactive_node(f, c, c, i+2, j, k, ibg) |
           inactive_node(f, c, c, i-1, j, k, ibg)   | inactive_node(f, c, c, i+1, j, k, ibg) |
           inactive_node(f, c, c, i-2, j+1, k, ibg) | inactive_node(f, c, c, i+2, j+1, k, ibg) |
           inactive_node(f, c, c, i-1, j+1, k, ibg) | inactive_node(f, c, c, i+1, j+1, k, ibg) |
           inactive_node(f, c, c, i, j+1, k, ibg) 
end

@inline function near_horizontal_boundary_y(i, j, k, ibg, scheme::WENOVectorInvariantVel) 
    return inactive_node(f, c, c, i, j, k, ibg)     | 
           inactive_node(f, c, c, i, j+3, k, ibg)   | inactive_node(f, c, c, i, j+3, k, ibg) |
           inactive_node(f, c, c, i, j+2, k, ibg)   | inactive_node(f, c, c, i, j+2, k, ibg) |
           inactive_node(f, c, c, i, j+1, k, ibg)   | inactive_node(f, c, c, i, j+1, k, ibg) |     
           inactive_node(c, f, c, i, j, k, ibg)     | 
           inactive_node(c, f, c, i, j-2, k, ibg)   | inactive_node(c, f, c, i, j+2, k, ibg) |
           inactive_node(c, f, c, i, j-1, k, ibg)   | inactive_node(c, f, c, i, j+1, k, ibg) | 
           inactive_node(c, f, c, i+1, j-2, k, ibg) | inactive_node(c, f, c, i+1, j+2, k, ibg) |
           inactive_node(c, f, c, i+1, j-1, k, ibg) | inactive_node(c, f, c, i+1, j+1, k, ibg) |
           inactive_node(c, f, c, i+1, j, k, ibg) 
end

# Takes forever to compile, but works.
# @inline near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{buffer}) where buffer = any(ntuple(δ -> inactive_node(i - buffer - 1 + δ, j, k, ibg), Val(2buffer + 1)))
# @inline near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{buffer}) where buffer = any(ntuple(δ -> inactive_node(i, j - buffer - 1 + δ, k, ibg), Val(2buffer + 1)))
# @inline near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{buffer}) where buffer = any(ntuple(δ -> inactive_node(i, j, k - buffer - 1 + δ, ibg), Val(2buffer + 1)))

for bias in (:symmetric, :left_biased, :right_biased)
    for (d, ξ) in enumerate((:x, :y, :z))

        code = [:ᵃ, :ᵃ, :ᵃ]

        for loc in (:ᶜ, :ᶠ)
            code[d] = loc
            second_order_interp = Symbol(:ℑ, ξ, code...)
            interp = Symbol(bias, :_interpolate_, ξ, code...)
            alt_interp = Symbol(:_, interp)

            near_boundary = Symbol(:near_, ξ, :_boundary)

            # Conditional high-order interpolation in Bounded directions
            @eval begin
                import Oceananigans.Advection: $alt_interp
                using Oceananigans.Advection: $interp

                @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme, ψ) =
                    ifelse($near_boundary(i, j, k, ibg, scheme),
                           $second_order_interp(i, j, k, ibg.underlying_grid, ψ),
                           $interp(i, j, k, ibg.underlying_grid, scheme, ψ))
            end
            if ξ == :z
                @eval begin
                    import Oceananigans.Advection: $alt_interp
                    using Oceananigans.Advection: $interp
    
                    @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENOVectorInvariant, ∂z, VI, u) =
                        ifelse($near_boundary(i, j, k, ibg, scheme),
                            $second_order_interp(i, j, k, ibg.underlying_grid, ∂z, u),
                            $interp(i, j, k, ibg.underlying_grid, scheme, ∂z, VI, u))
                end
            else    
                @eval begin
                    import Oceananigans.Advection: $alt_interp
                    using Oceananigans.Advection: $interp
    
                    @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENOVectorInvariant, ζ, VI, u, v) =
                        ifelse($near_boundary(i, j, k, ibg, scheme),
                                $second_order_interp(i, j, k, ibg.underlying_grid, ζ, u, v),
                                $interp(i, j, k, ibg.underlying_grid, scheme, ζ, VI, u, v))
                end    
            end
        end
    end
end

for bias in (:left_biased, :right_biased)
    for (d, dir) in zip((:x, :y), (:xᶜᵃᵃ, :yᵃᶜᵃ))
        interp     = Symbol(bias, :_interpolate_, dir)
        alt_interp = Symbol(:_, interp)

        near_horizontal_boundary = Symbol(:near_horizontal_boundary_, d)
        @eval begin
            @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENOVectorInvariantVel, ζ, ::Type{VelocityStencil}, u, v) =
            ifelse($near_horizontal_boundary(i, j, k, ibg, scheme),
               $alt_interp(i, j, k, ibg, scheme, ζ, VorticityStencil, u, v),
               $interp(i, j, k, ibg.underlying_grid, scheme, ζ, VelocityStencil, u, v))
        end
    end
end
