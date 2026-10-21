# Beyond a Closed side a flux condition leaves the halo without data, the flux entering through the tendency; the exterior
# cell beyond an Open side continues the interior with zero gradient
@inline _fill_west_halo!(j, k, grid, c, ::FBC, args...)   = fill_open_exterior!(topology(grid, 1), West(),   j, k, grid, c)
@inline _fill_east_halo!(j, k, grid, c, ::FBC, args...)   = fill_open_exterior!(topology(grid, 1), East(),   j, k, grid, c)
@inline _fill_south_halo!(i, k, grid, c, ::FBC, args...)  = fill_open_exterior!(topology(grid, 2), South(),  i, k, grid, c)
@inline _fill_north_halo!(i, k, grid, c, ::FBC, args...)  = fill_open_exterior!(topology(grid, 2), North(),  i, k, grid, c)
@inline _fill_bottom_halo!(i, j, grid, c, ::FBC, args...) = fill_open_exterior!(topology(grid, 3), Bottom(), i, j, grid, c)
@inline _fill_top_halo!(i, j, grid, c, ::FBC, args...)    = fill_open_exterior!(topology(grid, 3), Top(),    i, j, grid, c)

@inline fill_open_exterior!(topology, side, i, j, grid, c) = nothing

@inline fill_open_exterior!(::Type{<:Sides{Open}},        ::West,   j, k, grid, c) = @inbounds c[0, j, k] = c[1, j, k]
@inline fill_open_exterior!(::Type{<:Sides{<:Any, Open}}, ::East,   j, k, grid, c) = @inbounds c[grid.Nx + 1, j, k] = c[grid.Nx, j, k]
@inline fill_open_exterior!(::Type{<:Sides{Open}},        ::South,  i, k, grid, c) = @inbounds c[i, 0, k] = c[i, 1, k]
@inline fill_open_exterior!(::Type{<:Sides{<:Any, Open}}, ::North,  i, k, grid, c) = @inbounds c[i, grid.Ny + 1, k] = c[i, grid.Ny, k]
@inline fill_open_exterior!(::Type{<:Sides{Open}},        ::Bottom, i, j, grid, c) = @inbounds c[i, j, 0] = c[i, j, 1]
@inline fill_open_exterior!(::Type{<:Sides{<:Any, Open}}, ::Top,    i, j, grid, c) = @inbounds c[i, j, grid.Nz + 1] = c[i, j, grid.Nz]
