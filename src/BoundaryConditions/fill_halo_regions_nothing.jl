#####
##### Nothing happens when your boundary condition is nothing
#####

fill_west_halo!(c, ::Nothing, arch, barrier, grid, args...) = NoneEvent()
fill_east_halo!(c, ::Nothing, arch, barrier, grid, args...) = NoneEvent()

fill_north_halo!(c, ::Nothing, arch, barrier, grid, args...) = NoneEvent()
fill_south_halo!(c, ::Nothing, arch, barrier, grid, args...) = NoneEvent()

fill_top_halo!(c, ::Nothing, arch, barrier, grid, args...)    = NoneEvent()
fill_bottom_halo!(c, ::Nothing, arch, barrier, grid, args...) = NoneEvent()
