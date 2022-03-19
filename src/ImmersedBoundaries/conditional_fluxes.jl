using Oceananigans.Advection: AbstractAdvectionScheme
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑxᶜᵃᵃ, ℑyᵃᶠᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, ℑzᵃᵃᶜ, 
                              ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxzᶠᵃᶜ, ℑxzᶜᵃᶠ, ℑyzᵃᶠᶜ, ℑyzᵃᶜᶠ,
                              Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶠ
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, AbstractTimeDiscretization
using Oceananigans: fields

const ATC = AbstractTurbulenceClosure
const ATD = AbstractTimeDiscretization

####
#### Drag functions
####

const κVK = 0.4 # van Karman's const
const z₀ = 0.02 # roughness length (meters), user defined in future?

@inline west_drag_const(i, j, k, grid)   = @inbounds -(κVK / log(0.5 * Δxᶠᶜᶜ(i, j, k, grid) / z₀))^2 
@inline south_drag_const(i, j, k, grid)  = @inbounds -(κVK / log(0.5 * Δyᶜᶠᶜ(i, j, k, grid) / z₀))^2 
@inline bottom_drag_const(i, j, k, grid) = @inbounds -(κVK / log(0.5 * Δzᶜᶜᶠ(i, j, k, grid) / z₀))^2 

@inline τˣᶻ_drag(i, j, k, grid, U) = @inbounds bottom_drag_const(i, j, k, grid) * U.u[i, j, k] * (U.u[i, j, k]^2 + ℑxyᶠᶜᵃ(i, j, k, grid, U.v)^2)^(0.5)
@inline τʸᶻ_drag(i, j, k, grid, U) = @inbounds bottom_drag_const(i, j, k, grid) * U.v[i, j, k] * (U.v[i, j, k]^2 + ℑxyᶜᶠᵃ(i, j, k, grid, U.u)^2)^(0.5)

@inline τˣʸ_drag(i, j, k, grid, U) = @inbounds south_drag_const(i, j, k, grid)  * U.u[i, j, k] * (U.u[i, j, k]^2 + ℑxzᶠᵃᶜ(i, j, k, grid, U.w)^2)^(0.5)
@inline τᶻʸ_drag(i, j, k, grid, U) = @inbounds south_drag_const(i, j, k, grid)  * U.w[i, j, k] * (U.w[i, j, k]^2 + ℑxzᶜᵃᶠ(i, j, k, grid, U.u)^2)^(0.5)

@inline τʸˣ_drag(i, j, k, grid, U) = @inbounds west_drag_const(i, j, k, grid)   * U.u[i, j, k] * (U.u[i, j, k]^2 + ℑyzᵃᶠᶜ(i, j, k, grid, U.w)^2)^(0.5)
@inline τᶻˣ_drag(i, j, k, grid, U) = @inbounds west_drag_const(i, j, k, grid)   * U.w[i, j, k] * (U.w[i, j, k]^2 + ℑyzᵃᶜᶠ(i, j, k, grid, U.v)^2)^(0.5)


# These will only work for face nodes of GridFittedBoundary grids
@inline  east_fluid_solid_interface(::Face, LY, Lz, i, j, k, grid) = solid_node(Center(), LY, Lz, i-1, j, k, grid) & !solid_node(Center(), LY, Lz, i, j, k, grid)
@inline north_fluid_solid_interface(LX, ::Face, Lz, i, j, k, grid) = solid_node(LX, Center(), Lz, i, j-1, k, grid) & !solid_node(LX, Center(), Lz, i, j, k, grid)
@inline   top_fluid_solid_interface(LX, LY, ::Face, i, j, k, grid) = solid_node(LX, LY, Center(), i, j, k-1, grid) & !solid_node(LX, LY, Center(), i, j, k, grid)


if false # For drag boundary conditions
    # will always be within cell for grid fitted
    @inline conditional_flux_ccc(i, j, k, ibg::IBG{FT}, flux, closure, disc, clock, U, args...) where FT = ifelse(solid_interface(c, c, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, closure, clock, U, args...))
    # tau xy, tau yx
    @inline conditional_flux_ffc(i, j, k, ibg::IBG{FT}, flux, closure, disc, clock, U, args...) where FT = ifelse(solid_interface(f, f, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, closure, clock, U, args...))
    # tau xz, tau zx
    @inline function conditional_flux_fcf(i, j, k, ibg::IBG{FT}, flux, closure, disc, clock, U, args...) where FT 
        println("AAAA")
        return ifelse(top_solid_interface(f, c, f, i, j, k, ibg), τˣᶻ_drag(i, j, k, ibg, U), flux(i, j, k, ibg, closure, disc, clock, U, args...))
    end

    # tau yz, tau zy
    @inline conditional_flux_cff(i, j, k, ibg::IBG{FT}, flux, closure, disc, clock, U, args...) where FT = ifelse(solid_interface(c, f, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, closure, disc, clock, U, args...))
                                                            
    @inline conditional_flux_fcc(i, j, k, ibg::IBG{FT}, flux, closure, disc, clock, U, args...) where FT = ifelse(solid_interface(f, c, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, closure, disc, clock, U, args...))
    @inline conditional_flux_cfc(i, j, k, ibg::IBG{FT}, flux, closure, disc, clock, U, args...) where FT = ifelse(solid_interface(c, f, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, closure, disc, clock, U, args...))
    @inline conditional_flux_ccf(i, j, k, ibg::IBG{FT}, flux, closure, disc, clock, U, args...) where FT = ifelse(solid_interface(c, c, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, closure, disc, clock, U, args...))

else

    @inline conditional_flux_ccc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(c, c, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
    @inline conditional_flux_ffc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(f, f, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
    @inline function conditional_flux_fcf(i, j, k, ibg::IBG{FT}, flux, args...) where FT
        println("BBBB")
        return ifelse(solid_interface(f, c, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
    end
    @inline conditional_flux_cff(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(c, f, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
    @inline conditional_flux_fcc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(f, c, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
    @inline conditional_flux_cfc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(c, f, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
    @inline conditional_flux_ccf(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(c, c, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
end




#####
##### Viscous fluxes
#####

# ccc, ffc, fcf
@inline _viscous_flux_ux(i, j, k, ibg::GFIBG, closure, disc::ATD, clock, U, args...) = conditional_flux_ccc(i, j, k, ibg, viscous_flux_ux, closure, disc, clock, U, args...)
@inline _viscous_flux_uy(i, j, k, ibg::GFIBG, closure, disc::ATD, clock, U, args...) = conditional_flux_ffc(i, j, k, ibg, viscous_flux_uy, closure, disc, clock, U, args...)
@inline _viscous_flux_uz(i, j, k, ibg::GFIBG, closure, disc::ATD, clock, U, args...) = conditional_flux_fcf(i, j, k, ibg, viscous_flux_uz, closure, disc, clock, U, args...)

# ffc, ccc, cff
@inline _viscous_flux_vx(i, j, k, ibg::GFIBG, closure, disc::ATD, clock, U, args...) = conditional_flux_ffc(i, j, k, ibg, viscous_flux_vx, closure, disc, clock, U, args...)
@inline _viscous_flux_vy(i, j, k, ibg::GFIBG, closure, disc::ATD, clock, U, args...) = conditional_flux_ccc(i, j, k, ibg, viscous_flux_vy, closure, disc, clock, U, args...)
@inline _viscous_flux_vz(i, j, k, ibg::GFIBG, closure, disc::ATD, clock, U, args...) = conditional_flux_cff(i, j, k, ibg, viscous_flux_vz, closure, disc, clock, U, args...)

# fcf, cff, ccc
@inline _viscous_flux_wx(i, j, k, ibg::GFIBG, closure, disc::ATD, clock, U, args...) = conditional_flux_fcf(i, j, k, ibg, viscous_flux_wx, closure, disc, clock, U, args...)
@inline _viscous_flux_wy(i, j, k, ibg::GFIBG, closure, disc::ATD, clock, U, args...) = conditional_flux_cff(i, j, k, ibg, viscous_flux_wy, closure, disc, clock, U, args...)
@inline _viscous_flux_wz(i, j, k, ibg::GFIBG, closure, disc::ATD, clock, U, args...) = conditional_flux_ccc(i, j, k, ibg, viscous_flux_wz, closure, disc, clock, U, args...)

# fcc, cfc, ccf
@inline _diffusive_flux_x(i, j, k, ibg::GFIBG, args...) = conditional_flux_fcc(i, j, k, ibg, diffusive_flux_x, args...)
@inline _diffusive_flux_y(i, j, k, ibg::GFIBG, args...) = conditional_flux_cfc(i, j, k, ibg, diffusive_flux_y, args...)
@inline _diffusive_flux_z(i, j, k, ibg::GFIBG, args...) = conditional_flux_ccf(i, j, k, ibg, diffusive_flux_z, args...)

#####
##### Advective fluxes
#####

@inline conditional_advective_flux_ccc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(c, c, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_advective_flux_ffc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(f, f, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_advective_flux_fcf(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(f, c, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_advective_flux_cff(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(c, f, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_advective_flux_fcc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(f, c, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_advective_flux_cfc(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(c, f, c, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))
@inline conditional_advective_flux_ccf(i, j, k, ibg::IBG{FT}, flux, args...) where FT = ifelse(solid_interface(c, c, f, i, j, k, ibg), zero(FT), flux(i, j, k, ibg, args...))

# dx(uu), dy(vu), dz(wu)
# ccc,    ffc,    fcf
@inline _advective_momentum_flux_Uu(i, j, k, ibg::GFIBG, args...) = conditional_advective_flux_ccc(i, j, k, ibg, advective_momentum_flux_Uu, args...)
@inline _advective_momentum_flux_Vu(i, j, k, ibg::GFIBG, args...) = conditional_advective_flux_ffc(i, j, k, ibg, advective_momentum_flux_Vu, args...)
@inline _advective_momentum_flux_Wu(i, j, k, ibg::GFIBG, args...) = conditional_advective_flux_fcf(i, j, k, ibg, advective_momentum_flux_Wu, args...)

# dx(uv), dy(vv), dz(wv)
# ffc,    ccc,    cff
@inline _advective_momentum_flux_Uv(i, j, k, ibg::GFIBG, args...) = conditional_advective_flux_ffc(i, j, k, ibg, advective_momentum_flux_Uv, args...)
@inline _advective_momentum_flux_Vv(i, j, k, ibg::GFIBG, args...) = conditional_advective_flux_ccc(i, j, k, ibg, advective_momentum_flux_Vv, args...)
@inline _advective_momentum_flux_Wv(i, j, k, ibg::GFIBG, args...) = conditional_advective_flux_cff(i, j, k, ibg, advective_momentum_flux_Wv, args...)

# dx(uw), dy(vw), dz(ww)
# fcf,    cff,    ccc
@inline _advective_momentum_flux_Uw(i, j, k, ibg::GFIBG, args...) = conditional_advective_flux_fcf(i, j, k, ibg, advective_momentum_flux_Uw, args...)
@inline _advective_momentum_flux_Vw(i, j, k, ibg::GFIBG, args...) = conditional_advective_flux_cff(i, j, k, ibg, advective_momentum_flux_Vw, args...)
@inline _advective_momentum_flux_Ww(i, j, k, ibg::GFIBG, args...) = conditional_advective_flux_ccc(i, j, k, ibg, advective_momentum_flux_Ww, args...)

@inline _advective_tracer_flux_x(i, j, k, ibg::GFIBG, args...) = conditional_advective_flux_fcc(i, j, k, ibg, advective_tracer_flux_x, args...)
@inline _advective_tracer_flux_y(i, j, k, ibg::GFIBG, args...) = conditional_advective_flux_cfc(i, j, k, ibg, advective_tracer_flux_y, args...)
@inline _advective_tracer_flux_z(i, j, k, ibg::GFIBG, args...) = conditional_advective_flux_ccf(i, j, k, ibg, advective_tracer_flux_z, args...)

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

                # @inline $alt_interp(i, j, k, ibg::IBG, scheme, ψ) = $interp(i, j, k, ibg.grid, scheme, ψ)
            end
        end
    end
end
