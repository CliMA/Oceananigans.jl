#####
##### Nothing happens when your boundary condition is nothing
#####

fill_west_halo!(c,   ::Nothing, args...; kwargs...) = NoneEvent()
fill_east_halo!(c,   ::Nothing, args...; kwargs...) = NoneEvent()
fill_south_halo!(c,  ::Nothing, args...; kwargs...) = NoneEvent()
fill_north_halo!(c,  ::Nothing, args...; kwargs...) = NoneEvent()
fill_top_halo!(c,    ::Nothing, args...; kwargs...) = NoneEvent()
fill_bottom_halo!(c, ::Nothing, args...; kwargs...) = NoneEvent()

fill_west_and_east_halo!(c,  ::Nothing, ::Nothing, args...; kwargs...) = NoneEvent()
fill_south_and_north_halo!(c,::Nothing, ::Nothing, args...; kwargs...) = NoneEvent()
fill_bottom_and_top_halo!(c, ::Nothing, ::Nothing, args...; kwargs...) = NoneEvent()

