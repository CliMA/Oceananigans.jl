using Oceananigans.Grids: AbstractGrid, Flat

#####
##### Suprise: there is no halo-filling in `Flat` directions
#####

fill_west_halo!(c, ::Nothing, arch, barrier, grid::AbstractGrid{FT, Flat}, args...) where {FT} = nothing
fill_east_halo!(c, ::Nothing, arch, barrier, grid::AbstractGrid{FT, Flat}, args...) where {FT} = nothing

fill_north_halo!(c, ::Nothing, arch, barrier, grid::AbstractGrid{FT, TX, Flat}, args...) where {FT, TX} = nothing
fill_south_halo!(c, ::Nothing, arch, barrier, grid::AbstractGrid{FT, TX, Flat}, args...) where {FT, TX} = nothing

fill_top_halo!(c, ::Nothing, arch, barrier, grid::AbstractGrid{FT, TX, TY, Flat}, args...) where {FT, TX, TY} = nothing
fill_bottom_halo!(c, ::Nothing, arch, barrier, grid::AbstractGrid{FT, TX, TY, Flat}, args...) where {FT, TX, TY} = nothing
