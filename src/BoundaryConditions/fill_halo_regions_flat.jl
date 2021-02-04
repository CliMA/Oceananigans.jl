using Oceananigans.Grids: AbstractGrid, Flat

#####
##### Suprise: there is no halo-filling in `Flat` directions
#####

fill_west_halo!(c, ::Nothing, arch, barrier, grid, args...) = nothing
fill_east_halo!(c, ::Nothing, arch, barrier, grid, args...) = nothing

fill_north_halo!(c, ::Nothing, arch, barrier, grid, args...) = nothing
fill_south_halo!(c, ::Nothing, arch, barrier, grid, args...) = nothing

fill_top_halo!(c, ::Nothing, arch, barrier, grid, args...) = nothing
fill_bottom_halo!(c, ::Nothing, arch, barrier, grid, args...) = nothing
