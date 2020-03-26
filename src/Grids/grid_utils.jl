#####
##### Convinience functions
#####

"""
    total_extent(topology, H, Δ, L)

Returns the total extent, including halo regions, of constant-spaced
`Periodic` and `Flat` dimensions with number of halo points `H`, 
constant grid spacing `Δ`, and interior extent `L`.
"""
total_extent(topology, H, Δ, L) = L + (2H - 1) * Δ

"""
    total_extent(::Type{Bounded}, H, Δ, L)

Returns the total extent of, including halo regions, of constant-spaced
`Bounded` and `Flat` dimensions with number of halo points `H`,
constant grid spacing `Δ`, and interior extent `L`.
"""
total_extent(::Type{Bounded}, H, Δ, L) = L + 2H * Δ

"""
    total_length(loc, topo, N, H=0)

Returns the total length (number of nodes), including halo points, of a field 
located at `Cell` centers along a grid dimension of length `N` and with halo points `H`.
"""
total_length(loc, topo, N, H=0) = N + 2H

"""
    total_length(::Type{Face}, ::Type{Bounded}, N, H=0)

Returns the total length, including halo points, of a field located at 
cell `Face`s along a grid dimension of length `N` and with halo points `H`.
"""
total_length(::Type{Face}, ::Type{Bounded}, N, H=0) = N + 1 + 2H

#####
##### << Nodes >> 
#####

@inline interior_indices(loc, topo, N) = 1:N
@inline interior_indices(::Type{Face}, ::Type{Bounded}, N) = 1:N+1

@inline interior_x_indices(loc, grid) = interior_indices(loc, topology(grid, 1), grid.Nx)
@inline interior_y_indices(loc, grid) = interior_indices(loc, topology(grid, 2), grid.Ny)
@inline interior_z_indices(loc, grid) = interior_indices(loc, topology(grid, 3), grid.Nz)

@inline interior_parent_indices(loc, topo, N, H) = 1+H:N+H
@inline interior_parent_indices(::Type{Face}, ::Type{Bounded}, N, H) = 1+H:N+1+H

# Node by node
@inline xnode(::Type{Cell}, i, grid) = @inbounds grid.xC[i, 1, 1]
@inline xnode(::Type{Face}, i, grid) = @inbounds grid.xF[i, 1, 1]

@inline ynode(::Type{Cell}, j, grid) = @inbounds grid.yC[1, j, 1]
@inline ynode(::Type{Face}, j, grid) = @inbounds grid.yF[1, j, 1]

@inline znode(::Type{Cell}, k, grid) = @inbounds grid.zC[1, 1, k]
@inline znode(::Type{Face}, k, grid) = @inbounds grid.zF[1, 1, k]

"""
    xnodes(loc, grid)

Returns a view over the interior `loc=Cell` or loc=Face` nodes
on `grid` in the x-direction. For `Bounded` directions,
`Face` nodes include the boundary points.

See `znodes` for examples.
"""
xnodes(::Type{Cell}, grid) = view(grid.xC, 1:grid.Nx, :, :)

"""
    ynodes(loc, grid)

Returns a view over the interior `loc=Cell` or loc=Face` nodes
on `grid` in the y-direction. For `Bounded` directions,
`Face` nodes include the boundary points.

See `znodes` for examples.
"""
ynodes(::Type{Cell}, grid) = view(grid.yC, :, 1:grid.Ny, :)

"""
    znodes(loc, grid)

Returns a view over the interior `loc=Cell` or loc=Face` nodes
on `grid` in the z-direction. For `Bounded` directions,
`Face` nodes include the boundary points.

Examples
========

```jldoctest
julia> using Oceananigans, Oceananigans.Grids

julia> horz_periodic_grid = RegularCartesianGrid(size=(3, 3, 3), extent=(2π, 2π, 1), 
                                                 topology=(Periodic, Periodic, Bounded));

julia> zC = znodes(Cell, horz_periodic_grid)
1×1×3 view(OffsetArray(reshape(::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}, 1, 1, 5), 1:1, 1:1, 0:4), :, :, 1:3) with eltype Float64 with indices 1:1×1:1×Base.OneTo(3):
[:, :, 1] =
 -0.8333333333333331

[:, :, 2] =
 -0.4999999999999999

[:, :, 3] =
 -0.16666666666666652

julia> zF = znodes(Face, horz_periodic_grid)
1×1×4 view(OffsetArray(reshape(::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}, 1, 1, 6), 1:1, 1:1, 0:5), :, :, 1:4) with eltype Float64 with indices 1:1×1:1×Base.OneTo(4):
[:, :, 1] =
 -1.0

[:, :, 2] =
 -0.6666666666666666

[:, :, 3] =
 -0.33333333333333337

[:, :, 4] =
 -4.44089209850063e-17
```
"""
znodes(::Type{Cell}, grid) = view(grid.zC, :, :, 1:grid.Nz)

xnodes(::Type{Face}, grid) = view(grid.xF, interior_indices(Face, topology(grid, 1), grid.Nx), :, :)
ynodes(::Type{Face}, grid) = view(grid.yF, :, interior_indices(Face, topology(grid, 2), grid.Ny), :)
znodes(::Type{Face}, grid) = view(grid.zF, :, :, interior_indices(Face, topology(grid, 3), grid.Nz))

"""
    nodes(loc, grid)

Returns a 3-tuple of views over the interior nodes
at the locations in `loc` in `x, y, z`.

See `xnodes`, `ynodes`, and `znodes`.
"""
nodes(loc, grid::AbstractGrid) = (xnodes(loc[1], grid),
                                  ynodes(loc[2], grid),
                                  znodes(loc[3], grid))

#####
##### Convinience functions
#####

unpack_grid(grid) = grid.Nx, grid.Ny, grid.Nz, grid.Lx, grid.Ly, grid.Lz

#####
##### Input validation
#####

function validate_topology(topology)
    for T in topology
        if !isa(T(), AbstractTopology)
            e = "$T is not a valid topology! " *
                "Valid topologies are: Periodic, Bounded, Flat."
            throw(ArgumentError(e))
        end
    end

    return topology
end

"""Validate that an argument tuple is the right length and has elements of type `argtype`."""
function validate_tupled_argument(arg, argtype, argname)
    length(arg) == 3        || throw(ArgumentError("length($argname) must be 3."))
    all(isa.(arg, argtype)) || throw(ArgumentError("$argname=$arg must contain $argtype s."))
    all(arg .> 0)           || throw(ArgumentError("Elements of $argname=$arg must be > 0!"))
    return nothing
end

coordinate_name(i) = i == 1 ? "x" : i == 2 ? "y" : "z"

function validate_dimension_specification(i, c)
    name = coordinate_name(i)
    length(c) == 2       || throw(ArgumentError("$name length($c) must be 2."))
    all(isa.(c, Number)) || throw(ArgumentError("$name=$c should contain numbers."))
    c[2] >= c[1]         || throw(ArgumentError("$name=$c should be an increasing interval."))
    return nothing
end

function validate_regular_grid_size_and_extent(FT, size, extent, halo, x, y, z)
    validate_tupled_argument(size, Integer, "size")
    validate_tupled_argument(halo, Integer, "halo")

    # Find domain endpoints or domain extent, depending on user input:
    if !isnothing(extent) # the user has specified an extent!

        (!isnothing(x) || !isnothing(y) || !isnothing(z)) &&
            throw(ArgumentError("Cannot specify both length and x, y, z keyword arguments."))

        validate_tupled_argument(extent, Number, "extent")

        Lx, Ly, Lz = extent

        # An "oceanic" default domain:
        x = (  0, Lx)
        y = (  0, Ly)
        z = (-Lz,  0)

    else # isnothing(extent) === true implies that user has not specified a length

        (isnothing(x) || isnothing(y) || isnothing(z)) &&
            throw(ArgumentError("Must supply length or x, y, z keyword arguments."))

        for (i, c) in enumerate((x, y, z))
            validate_dimension_specification(i, c)
        end

        Lx = x[2] - x[1]
        Ly = y[2] - y[1]
        Lz = z[2] - z[1]
    end

    return FT(Lx), FT(Ly), FT(Lz), FT.(x), FT.(y), FT.(z)
end

function validate_vertically_stretched_grid_size_and_xy(FT, size, halo, x, y)
    validate_tupled_argument(size, Integer, "size")
    validate_tupled_argument(halo, Integer, "halo")

    for (i, c) in enumerate((x, y))
            validate_dimension_specification(i, c)
        end

        Lx = x[2] - x[1]
        Ly = y[2] - y[1]

    return FT(Lx), FT(Ly), FT.(x), FT.(y)
end

