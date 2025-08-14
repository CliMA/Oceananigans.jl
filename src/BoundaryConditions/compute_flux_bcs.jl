using Oceananigans: instantiated_location
using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.Grids
using Oceananigans.Grids: AbstractGrid

#####
##### Algorithm for adding fluxes associated with non-trivial flux boundary conditions.
##### Value and Gradient boundary conditions are handled by filling halos.
#####

# Unpack
compute_x_bcs!(Gc, c, args...) = compute_x_bcs!(Gc, Gc.grid, c, c.boundary_conditions.west,   c.boundary_conditions.east, args...)
compute_y_bcs!(Gc, c, args...) = compute_y_bcs!(Gc, Gc.grid, c, c.boundary_conditions.south,  c.boundary_conditions.north, args...)
compute_z_bcs!(Gc, c, args...) = compute_z_bcs!(Gc, Gc.grid, c, c.boundary_conditions.bottom, c.boundary_conditions.top, args...)

# Shortcuts for...
#
# Nothing tendencies.
compute_x_bcs!(::Nothing, args...) = nothing
compute_y_bcs!(::Nothing, args...) = nothing
compute_z_bcs!(::Nothing, args...) = nothing

# Not-flux boundary conditions
const NotFluxBC = Union{PBC, MCBC, DCBC, VBC, GBC, OBC, ZFBC, Nothing}

compute_x_bcs!(Gc, ::AbstractGrid, c, ::NotFluxBC, ::NotFluxBC, ::AbstractArchitecture, args...) = nothing
compute_y_bcs!(Gc, ::AbstractGrid, c, ::NotFluxBC, ::NotFluxBC, ::AbstractArchitecture, args...) = nothing
compute_z_bcs!(Gc, ::AbstractGrid, c, ::NotFluxBC, ::NotFluxBC, ::AbstractArchitecture, args...) = nothing

# The real deal
"""
Apply flux boundary conditions to a field `c` by adding the associated flux divergence to
the source term `Gc` at the left and right.
"""
compute_x_bcs!(Gc, grid::AbstractGrid, c, west_bc, east_bc, arch::AbstractArchitecture, args...) =
    launch!(arch, grid, :yz, _compute_x_bcs!, Gc, instantiated_location(Gc), grid, west_bc, east_bc, Tuple(args))

"""
Apply flux boundary conditions to a field `c` by adding the associated flux divergence to
the source term `Gc` at the left and right.
"""
compute_y_bcs!(Gc, grid::AbstractGrid, c, south_bc, north_bc, arch::AbstractArchitecture, args...) =
    launch!(arch, grid, :xz, _compute_y_bcs!, Gc, instantiated_location(Gc), grid, south_bc, north_bc, Tuple(args))

"""
Apply flux boundary conditions to a field `c` by adding the associated flux divergence to
the source term `Gc` at the top and bottom.
"""
compute_z_bcs!(Gc, grid::AbstractGrid, c, bottom_bc, top_bc, arch::AbstractArchitecture, args...) =
    launch!(arch, grid, :xy, _compute_z_bcs!, Gc, instantiated_location(Gc), grid, bottom_bc, top_bc, Tuple(args))

"""
    _compute_x_bcs!(Gc, grid, west_bc, east_bc, args...)

Apply a west and/or east boundary condition to variable `c`.
"""
@kernel function _compute_x_bcs!(Gc, loc, grid, west_bc, east_bc, args)
    j, k = @index(Global, NTuple)
    compute_x_west_bc!(Gc, loc, west_bc, j, k, grid, args...)
    compute_x_east_bc!(Gc, loc, east_bc, j, k, grid, args...)
end

"""
    _compute_y_bcs!(Gc, grid, south_bc, north_bc, args...)

Apply a south and/or north boundary condition to variable `c`.
"""
@kernel function _compute_y_bcs!(Gc, loc, grid, south_bc, north_bc, args)
    i, k = @index(Global, NTuple)
    compute_y_south_bc!(Gc, loc, south_bc, i, k, grid, args...)
    compute_y_north_bc!(Gc, loc, north_bc, i, k, grid, args...)
end

"""
    _compute_z_bcs!(Gc, grid, bottom_bc, top_bc, args...)

Apply a top and/or bottom boundary condition to variable `c`.
"""
@kernel function _compute_z_bcs!(Gc, loc, grid, bottom_bc, top_bc, args)
    i, j = @index(Global, NTuple)
    compute_z_bottom_bc!(Gc, loc, bottom_bc, i, j, grid, args...)
       compute_z_top_bc!(Gc, loc, top_bc,    i, j, grid, args...)
end

# Shortcuts for zero flux or non-flux boundary conditions
@inline   compute_x_east_bc!(Gc, loc, ::NotFluxBC, args...) = nothing
@inline   compute_x_west_bc!(Gc, loc, ::NotFluxBC, args...) = nothing
@inline  compute_y_north_bc!(Gc, loc, ::NotFluxBC, args...) = nothing
@inline  compute_y_south_bc!(Gc, loc, ::NotFluxBC, args...) = nothing
@inline    compute_z_top_bc!(Gc, loc, ::NotFluxBC, args...) = nothing
@inline compute_z_bottom_bc!(Gc, loc, ::NotFluxBC, args...) = nothing

# shortcut for the zipper BC
@inline compute_y_north_bc!(Gc, loc, ::ZBC, args...) = nothing

@inline flip(::Center) = Face()
@inline flip(::Face) = Center()

"""
    compute_x_west_bc!(Gc, loc, west_flux::BC{<:Flux}, j, k, grid, args...)

Add the flux divergence associated with a west flux boundary condition on `c`.
Note that because

    `tendency = ∂c/∂t = Gc = - ∇ ⋅ flux`

a positive west flux is associated with an *increase* in `Gc` near the west boundary.
If `west_bc.condition` is a function, the function must have the signature

    `west_bc.condition(j, k, grid, boundary_condition_args...)`

The same logic holds for south and bottom boundary conditions in `y`, and `z`, respectively.
"""
@inline function compute_x_west_bc!(Gc, loc, west_flux::BC{<:Flux}, j, k, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[1, j, k] += getbc(west_flux, j, k, grid, args...) * Ax(1, j, k, grid, flip(LX), LY, LZ) / volume(1, j, k, grid, LX, LY, LZ)
    return nothing
end

@inline function compute_y_south_bc!(Gc, loc, south_flux::BC{<:Flux}, i, k, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[i, 1, k] += getbc(south_flux, i, k, grid, args...) * Ay(i, 1, k, grid, LX, flip(LY), LZ) / volume(i, 1, k, grid, LX, LY, LZ)
    return nothing
end

@inline function compute_z_bottom_bc!(Gc, loc, bottom_flux::BC{<:Flux}, i, j, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[i, j, 1] += getbc(bottom_flux, i, j, grid, args...) * Az(i, j, 1, grid, LX, LY, flip(LZ)) / volume(i, j, 1, grid, LX, LY, LZ)
    return nothing
end

"""
    compute_x_east_bc!(Gc, loc, east_flux::BC{<:Flux}, j, k, grid, args...)

Add the part of flux divergence associated with a east boundary condition on `c`.
Note that because

    `tendency = ∂c/∂t = Gc = - ∇ ⋅ flux`

a positive east flux is associated with a *decrease* in `Gc` near the east boundary.
If `east_bc.condition` is a function, the function must have the signature

    `east_bc.condition(i, j, grid, boundary_condition_args...)`

The same logic holds for north and top boundary conditions in `y`, and `z`, respectively.
"""
@inline function compute_x_east_bc!(Gc, loc, east_flux::BC{<:Flux}, j, k, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[grid.Nx, j, k] -= getbc(east_flux, j, k, grid, args...) * Ax(grid.Nx+1, j, k, grid, flip(LX), LY, LZ) / volume(grid.Nx, j, k, grid, LX, LY, LZ)
    return nothing
end

@inline function compute_y_north_bc!(Gc, loc, north_flux::BC{<:Flux}, i, k, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[i, grid.Ny, k] -= getbc(north_flux, i, k, grid, args...) * Ay(i, grid.Ny+1, k, grid, LX, flip(LY), LZ) / volume(i, grid.Ny, k, grid, LX, LY, LZ)
    return nothing
end

@inline function compute_z_top_bc!(Gc, loc, top_flux::BC{<:Flux}, i, j, grid, args...)
    LX, LY, LZ = loc
    @inbounds Gc[i, j, grid.Nz] -= getbc(top_flux, i, j, grid, args...) * Az(i, j, grid.Nz+1, grid, LX, LY, flip(LZ)) / volume(i, j, grid.Nz, grid, LX, LY, LZ)
    return nothing
end
