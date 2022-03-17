using Oceananigans.Advection: AbstractAdvectionScheme
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑxᶜᵃᵃ, ℑyᵃᶠᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ 
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, AbstractTimeDiscretization

const ATC = AbstractTurbulenceClosure
const ATD = AbstractTimeDiscretization

@inline conditional_flux_ccc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(c, c, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_flux_ffc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(f, f, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_flux_fcf(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(f, c, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_flux_cff(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(c, f, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))

@inline conditional_flux_fcc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(f, c, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_flux_cfc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(c, f, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_flux_ccf(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(c, c, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))

#####
##### Advective fluxes
#####

# ccc, ffc, fcf
 @inline _viscous_flux_ux(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, viscous_flux_ux, args...)
 @inline _viscous_flux_uy(i, j, k, ibg::GFIBG, args...) = conditional_flux_ffc(i, j, k, ibg, viscous_flux_uy, args...)
 @inline _viscous_flux_uz(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcf(i, j, k, ibg, viscous_flux_uz, args...)
 
 # ffc, ccc, cff
 @inline _viscous_flux_vx(i, j, k, ibg::GFIBG, args...) = conditional_flux_ffc(i, j, k, ibg, viscous_flux_vx, args...)
 @inline _viscous_flux_vy(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, viscous_flux_vy, args...)
 @inline _viscous_flux_vz(i, j, k, ibg::GFIBG, args...) = conditional_flux_cff(i, j, k, ibg, viscous_flux_vz, args...)
 
 # fcf, cff, ccc
 @inline _viscous_flux_wx(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcf(i, j, k, ibg, viscous_flux_wx, args...)
 @inline _viscous_flux_wy(i, j, k, ibg::GFIBG, args...) = conditional_flux_cff(i, j, k, ibg, viscous_flux_wy, args...)
 @inline _viscous_flux_wz(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, viscous_flux_wz, args...)

# fcc, cfc, ccf
@inline _diffusive_flux_x(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcc(i, j, k, ibg, diffusive_flux_x, args...)
@inline _diffusive_flux_y(i, j, k, ibg::GFIBG, args...) = conditional_flux_cfc(i, j, k, ibg, diffusive_flux_y, args...)
@inline _diffusive_flux_z(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccf(i, j, k, ibg, diffusive_flux_z, args...)

#####
##### Advective fluxes
#####

# dx(uu), dy(vu), dz(wu)
# ccc,    ffc,    fcf
@inline _advective_momentum_flux_Uu(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, advective_momentum_flux_Uu, args...)
@inline _advective_momentum_flux_Vu(i, j, k, ibg::GFIBG, args...) = conditional_flux_ffc(i, j, k, ibg, advective_momentum_flux_Vu, args...)
@inline _advective_momentum_flux_Wu(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcf(i, j, k, ibg, advective_momentum_flux_Wu, args...)

# dx(uv), dy(vv), dz(wv)
# ffc,    ccc,    cff
@inline _advective_momentum_flux_Uv(i, j, k, ibg::GFIBG, args...) = conditional_flux_ffc(i, j, k, ibg, advective_momentum_flux_Uv, args...)
@inline _advective_momentum_flux_Vv(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, advective_momentum_flux_Vv, args...)
@inline _advective_momentum_flux_Wv(i, j, k, ibg::GFIBG, args...) = conditional_flux_cff(i, j, k, ibg, advective_momentum_flux_Wv, args...)

# dx(uw), dy(vw), dz(ww)
# fcf,    cff,    ccc
@inline _advective_momentum_flux_Uw(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcf(i, j, k, ibg, advective_momentum_flux_Uw, args...)
@inline _advective_momentum_flux_Vw(i, j, k, ibg::GFIBG, args...) = conditional_flux_cff(i, j, k, ibg, advective_momentum_flux_Vw, args...)
@inline _advective_momentum_flux_Ww(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccc(i, j, k, ibg, advective_momentum_flux_Ww, args...)

   @inline _advective_tracer_flux_x(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcc(i, j, k, ibg, advective_tracer_flux_x, args...)
   @inline _advective_tracer_flux_y(i, j, k, ibg::GFIBG, args...) = conditional_flux_cfc(i, j, k, ibg, advective_tracer_flux_y, args...)
   @inline _advective_tracer_flux_z(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccf(i, j, k, ibg, advective_tracer_flux_z, args...)

#####
##### "Boundary-aware" interpolation
#####
##### Don't interpolate dead cells.
#####

@inline near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
@inline near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false
@inline near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{0}) = false

@inline near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{1}) = solid_node(i - 1, j, k, ibg) | solid_node(i, j, k, ibg) | solid_node(i + 1, j, k, ibg)
@inline near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{1}) = solid_node(i, j - 1, k, ibg) | solid_node(i, j, k, ibg) | solid_node(i, j + 1, k, ibg)
@inline near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{1}) = solid_node(i, j, k - 1, ibg) | solid_node(i, j, k, ibg) | solid_node(i, j, k + 1, ibg)

@inline near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{2}) = solid_node(i - 2, j, k, ibg) | solid_node(i - 1, j, k, ibg) | solid_node(i, j, k, ibg) | solid_node(i + 1, j, k, ibg) | solid_node(i + 2, j, k, ibg)
@inline near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{2}) = solid_node(i, j - 2, k, ibg) | solid_node(i, j - 1, k, ibg) | solid_node(i, j, k, ibg) | solid_node(i, j + 1, k, ibg) | solid_node(i, j + 2, k, ibg)
@inline near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{2}) = solid_node(i, j, k - 2, ibg) | solid_node(i, j, k - 1, ibg) | solid_node(i, j, k, ibg) | solid_node(i, j, k + 1, ibg) | solid_node(i, j, k + 2, ibg)

using Oceananigans.Advection: WENOVectorInvariantVel, VorticityStencil, VelocityStencil

@inline function near_horizontal_boundary_x(i, j, k, ibg, scheme::WENOVectorInvariantVel) 
    return solid_node(f, c, c, i, j, k, ibg)     | solid_node(c, f, c, i, j, k, ibg) |       
           solid_node(c, f, c, i-3, j, k, ibg)   | solid_node(c, f, c, i+3, j, k, ibg) |
           solid_node(c, f, c, i-2, j, k, ibg)   | solid_node(c, f, c, i+2, j, k, ibg) |
           solid_node(c, f, c, i-1, j, k, ibg)   | solid_node(c, f, c, i+1, j, k, ibg) | 
           solid_node(f, c, c, i-2, j, k, ibg)   | solid_node(f, c, c, i+2, j, k, ibg) |
           solid_node(f, c, c, i-1, j, k, ibg)   | solid_node(f, c, c, i+1, j, k, ibg) |
           solid_node(f, c, c, i-2, j+1, k, ibg) | solid_node(f, c, c, i+2, j+1, k, ibg) |
           solid_node(f, c, c, i-1, j+1, k, ibg) | solid_node(f, c, c, i+1, j+1, k, ibg) 
end

@inline function near_horizontal_boundary_y(i, j, k, ibg, scheme::WENOVectorInvariantVel) 
    return solid_node(f, c, c, i, j, k, ibg)     | solid_node(c, f, c, i, j, k, ibg) | 
           solid_node(f, c, c, i, j+3, k, ibg)   | solid_node(f, c, c, i, j+3, k, ibg) |
           solid_node(f, c, c, i, j+2, k, ibg)   | solid_node(f, c, c, i, j+2, k, ibg) |
           solid_node(f, c, c, i, j+1, k, ibg)   | solid_node(f, c, c, i, j+1, k, ibg) |     
           solid_node(c, f, c, i, j-2, k, ibg)   | solid_node(c, f, c, i, j+2, k, ibg) |
           solid_node(c, f, c, i, j-1, k, ibg)   | solid_node(c, f, c, i, j+1, k, ibg) | 
           solid_node(c, f, c, i+1, j-2, k, ibg) | solid_node(c, f, c, i+1, j+2, k, ibg) |
           solid_node(c, f, c, i+1, j-1, k, ibg) | solid_node(c, f, c, i+1, j+1, k, ibg) 
end

# Takes forever to compile, but works.
# @inline near_x_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{buffer}) where buffer = any(ntuple(δ -> solid_node(i - buffer - 1 + δ, j, k, ibg), Val(2buffer + 1)))
# @inline near_y_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{buffer}) where buffer = any(ntuple(δ -> solid_node(i, j - buffer - 1 + δ, k, ibg), Val(2buffer + 1)))
# @inline near_z_boundary(i, j, k, ibg, ::AbstractAdvectionScheme{buffer}) where buffer = any(ntuple(δ -> solid_node(i, j, k - buffer - 1 + δ, ibg), Val(2buffer + 1)))

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
                           $second_order_interp(i, j, k, ibg.grid, ψ),
                           $interp(i, j, k, ibg.grid, scheme, ψ))

                @inline $alt_interp(i, j, k, ibg::ImmersedBoundaryGrid, scheme::WENOVectorInvariant, ζ, VI, u, v) =
                    ifelse($near_boundary(i, j, k, ibg, scheme),
                           $second_order_interp(i, j, k, ibg.grid, ζ, u, v),
                           $interp(i, j, k, ibg.grid, scheme, ζ, VI, u, v))
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
               $interp(i, j, k, ibg.grid, scheme, ζ, VelocityStencil, u, v))
        end
    end
end


