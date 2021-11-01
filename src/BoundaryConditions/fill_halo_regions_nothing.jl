#####
##### Nothing happens when your boundary condition is nothing
#####

fill_west_halo!(c, ::Nothing, arch, barrier, grid, args...) = nothing
fill_east_halo!(c, ::Nothing, arch, barrier, grid, args...) = nothing

fill_north_halo!(c, ::Nothing, arch, barrier, grid, args...) = nothing
fill_south_halo!(c, ::Nothing, arch, barrier, grid, args...) = nothing

fill_top_halo!(c, ::Nothing, arch, barrier, grid, args...)    = nothing
fill_bottom_halo!(c, ::Nothing, arch, barrier, grid, args...) = nothing
