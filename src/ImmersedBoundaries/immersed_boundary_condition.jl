using Oceananigans.BoundaryConditions: FBC, VBC

######
###### Fluxes across immersed boundaries...
######

struct ImmersedBoundaryCondition{W, E, S, N, B, T}
    west :: W                  
    east :: E
    south :: S   
    north :: N
    bottom :: B
    top :: T
end

ImmersedBoundaryCondition(; west=nothing, east=nothing, south=nothing, north=nothing, bottom=nothing, top=nothing) =
    ImmersedBoundaryCondition(west, east, south, north, bottom, top)

# Easy cases: Nothing and FluxBoundaryCondition.
for side in (:west, :east, :south, :north, :bottom, :top)
    get_flux = Symbol(side, :_interface_flux)
    @eval $get_flux(::Nothing, i, j, k, ibg, args...) = zero(eltype(ibg))
    @eval $get_flux(bc::FBC, i, j, k, ibg, closure, args...) = getbc(bc, i, j, k, ibg, args...)
end

# Harder... ValueBoundaryCondition
west_interface_flux(bc::VBC, i, j, k, ibg, closure::AbstractScalarDiffusivity, K, args...) = getbc(east_bc, i, j, k, ibg, args...)


@inline function immersed_∂ⱼ_τ₁ⱼ(i, j, k, ibg::GFIBG, u_bc, clock, model_fields, closure)
     west_ib_flux  =   west_interface_flux(u_bc.west,   i, j, k, ibg, closure, clock, model_fields)
     east_ib_flux  =   east_interface_flux(u_bc.east,   i, j, k, ibg, closure, clock, model_fields)
    south_ib_flux  =  south_interface_flux(u_bc.south,  i, j, k, ibg, closure, clock, model_fields)
    north_ib_flux  =  north_interface_flux(u_bc.north,  i, j, k, ibg, closure, clock, model_fields)
    bottom_ib_flux = bottom_interface_flux(u_bc.bottom, i, j, k, ibg, closure, clock, model_fields)
      top_ib_flux  =    top_interface_flux(u_bc.top,    i, j, k, ibg, closure, clock, model_fields)

    # τ₁ⱼ at fcc, so...
    west_flux   = conditional_flux_ccc(i-1, j,   k,   ibg, west_ib_flux,   zero(eltype(ibg)))
    east_flux   = conditional_flux_ccc(i,   j,   k,   ibg, east_ib_flux,   zero(eltype(ibg)))
    south_flux  = conditional_flux_ffc(i,   j,   k,   ibg, south_ib_flux,  zero(eltype(ibg)))
    north_flux  = conditional_flux_ffc(i,   j+1, k,   ibg, north_ib_flux,  zero(eltype(ibg)))
    bottom_flux = conditional_flux_fcf(i,   j,   k,   ibg, bottom_ib_flux, zero(eltype(ibg)))
    top_flux    = conditional_flux_fcf(i,   j,   k+1, ibg, top_ib_flux,    zero(eltype(ibg)))

    return 1 / Vᶠᶜᶜ(i, j, k, grid) * (Axᶜᶜᶜ(i-1, j, k, ibg) * west_flux   - Axᶜᶜᶜ(i, j, k, grid, ibg)   * east_flux +
                                      Ayᶠᶠᶜ(i, j, k, ibg)   * south_flux  - Ayᶠᶠᶜ(i, j+1, k, grid, ibg) * north_flux +
                                      Azᶠᶜᶠ(i, j, k, ibg)   * bottom_flux - Azᶠᶜᶠ(i, j, k+1, grid, ibg) * top_flux)
end

@inline function immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid::GFIBG, args...)
    return 0
end

@inline function immersed_∂ⱼ_τ₃ⱼ(i, j, k, grid::GFIBG, args...)
    return 0
end

@inline function immresed_∇_dot_qᶜ(i, j, k, grid::GFIBG, args...)
    return 0
end

