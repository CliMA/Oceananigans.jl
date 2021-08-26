import Oceananigans.BoundaryConditions:
    apply_x_bcs!,
    apply_y_bcs!,
    apply_z_bcs!,
    apply_x_east_bc!,
    apply_x_west_bc!,
    apply_y_south_bc!,
    apply_y_north_bc!,
    apply_z_top_bc!,
    apply_z_bottom_bc!

# Bunch o' shortcuts for halo communication bcs
apply_x_bcs!(Gc, ::AbstractGrid, c, ::HaloCommunicationBC, ::HaloCommunicationBC, ::AbstractArchitecture, args...) = NoneEvent()
apply_y_bcs!(Gc, ::AbstractGrid, c, ::HaloCommunicationBC, ::HaloCommunicationBC, ::AbstractArchitecture, args...) = NoneEvent()
apply_z_bcs!(Gc, ::AbstractGrid, c, ::HaloCommunicationBC, ::HaloCommunicationBC, ::AbstractArchitecture, args...) = NoneEvent()

@inline apply_x_east_bc!(  Gc, loc, ::HaloCommunicationBC, args...) = nothing
@inline apply_x_west_bc!(  Gc, loc, ::HaloCommunicationBC, args...) = nothing
@inline apply_y_north_bc!( Gc, loc, ::HaloCommunicationBC, args...) = nothing
@inline apply_y_south_bc!( Gc, loc, ::HaloCommunicationBC, args...) = nothing
@inline apply_z_top_bc!(   Gc, loc, ::HaloCommunicationBC, args...) = nothing
@inline apply_z_bottom_bc!(Gc, loc, ::HaloCommunicationBC, args...) = nothing

