#####
##### Convinience functions
#####

Base.length(loc, topo, N) = N
Base.length(::Type{Face}, ::Type{Bounded}, N) = N+1

function Base.size(loc, grid::AbstractGrid)
    N = (grid.Nx, grid.Ny, grid.Nz)
    return Tuple(length(loc[d], topology(grid, d), N[d]) for d in 1:3)
end

Base.size(loc, grid, d) = size(loc, grid)[d]

total_size(a) = size(a) # fallback

"""
    total_size(loc, grid)

Returns the "total" size of a field at `loc` on `grid`.
This is a 3-tuple of integers corresponding to the number of grid points
contained by `f` along `x, y, z`.
"""
@inline total_size(loc, grid) = (total_length(loc[1], topology(grid, 1), grid.Nx, grid.Hx),
                                 total_length(loc[2], topology(grid, 2), grid.Ny, grid.Hy),
                                 total_length(loc[3], topology(grid, 3), grid.Nz, grid.Hz))

"""
    total_extent(topology, H, Δ, L)

Returns the total extent, including halo regions, of constant-spaced
`Periodic` and `Flat` dimensions with number of halo points `H`,
constant grid spacing `Δ`, and interior extent `L`.
"""
@inline total_extent(topology, H, Δ, L) = L + (2H - 1) * Δ

"""
    total_extent(::Type{Bounded}, H, Δ, L)

Returns the total extent of, including halo regions, of constant-spaced
`Bounded` and `Flat` dimensions with number of halo points `H`,
constant grid spacing `Δ`, and interior extent `L`.
"""
@inline total_extent(::Type{Bounded}, H, Δ, L) = L + 2H * Δ

"""
    total_length(loc, topo, N, H=0)

Returns the total length (number of nodes), including halo points, of a field
located at `Cell` centers along a grid dimension of length `N` and with halo points `H`.
"""
@inline total_length(loc, topo, N, H=0) = N + 2H

"""
    total_length(::Type{Face}, ::Type{Bounded}, N, H=0)

Returns the total length, including halo points, of a field located at
cell `Face`s along a grid dimension of length `N` and with halo points `H`.
"""
@inline total_length(::Type{Face}, ::Type{Bounded}, N, H=0) = N + 1 + 2H

# Grid domains
@inline domain(ξ, topo) = ξ[1], ξ[end]
@inline domain(ξ, ::Type{Bounded}) = ξ[1], ξ[end-1]

@inline x_domain(grid) = domain(grid.xF, topology(grid, 1))
@inline y_domain(grid) = domain(grid.yF, topology(grid, 2))
@inline z_domain(grid) = domain(grid.zF, topology(grid, 3))

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

# All indices including halos.
@inline all_indices(loc, topo, N, H) = 1-H:N+H
@inline all_indices(loc::Type{Face}, ::Type{Bounded}, N, H) = 1-H:N+1+H

@inline all_x_indices(loc, grid) = all_indices(loc, topology(grid, 1), grid.Nx, grid.Hx)
@inline all_y_indices(loc, grid) = all_indices(loc, topology(grid, 2), grid.Ny, grid.Hy)
@inline all_z_indices(loc, grid) = all_indices(loc, topology(grid, 3), grid.Nz, grid.Hz)

@inline all_parent_indices(loc, topo, N, H) = 1:N+2H
@inline all_parent_indices(::Type{Face}, ::Type{Bounded}, N, H) = 1:N+1+2H

@inline all_parent_x_indices(loc, grid) = all_parent_indices(loc, topology(grid, 1), grid.Nx, grid.Hx)
@inline all_parent_y_indices(loc, grid) = all_parent_indices(loc, topology(grid, 2), grid.Ny, grid.Hy)
@inline all_parent_z_indices(loc, grid) = all_parent_indices(loc, topology(grid, 3), grid.Nz, grid.Hz)

# Node by node
@inline xnode(::Type{Cell}, i, grid) = @inbounds grid.xC[i]
@inline xnode(::Type{Face}, i, grid) = @inbounds grid.xF[i]

@inline ynode(::Type{Cell}, j, grid) = @inbounds grid.yC[j]
@inline ynode(::Type{Face}, j, grid) = @inbounds grid.yF[j]

@inline znode(::Type{Cell}, k, grid) = @inbounds grid.zC[k]
@inline znode(::Type{Face}, k, grid) = @inbounds grid.zF[k]

# Convenience is king
@inline xC(i, grid) = xnode(Cell, i, grid)
@inline xF(i, grid) = xnode(Face, i, grid)

@inline yC(j, grid) = ynode(Cell, j, grid)
@inline yF(j, grid) = ynode(Face, j, grid)

@inline zC(k, grid) = znode(Cell, k, grid)
@inline zF(k, grid) = znode(Face, k, grid)

"""
    xnodes(loc, grid, reshape=false)

Returns a view over the interior `loc=Cell` or `loc=Face` nodes
on `grid` in the x-direction. For `Bounded` directions,
`Face` nodes include the boundary points. `reshape=false` will
return a 1D array while `reshape=true` will return a 3D array
with size Nx×1×1.

See `znodes` for examples.
"""
xnodes(::Type{Cell}, grid; reshape=false) =
    reshape ? Base.reshape(view(grid.xC, 1:grid.Nx), grid.Nx, 1, 1) :
              view(grid.xC, 1:grid.Nx)

"""
    ynodes(loc, grid, reshape=false)

Returns a view over the interior `loc=Cell` or `loc=Face` nodes
on `grid` in the y-direction. For `Bounded` directions,
`Face` nodes include the boundary points. `reshape=false` will
return a 1D array while `reshape=true` will return a 3D array
with size 1×Ny×1.


See `znodes` for examples.
"""
ynodes(::Type{Cell}, grid; reshape=false) =
    reshape ? Base.reshape(view(grid.yC, 1:grid.Ny), 1, grid.Ny, 1) :
              view(grid.yC, 1:grid.Ny)

"""
    znodes(loc, grid, reshape=false)

Returns a view over the interior `loc=Cell` or `loc=Face` nodes
on `grid` in the z-direction. For `Bounded` directions,
`Face` nodes include the boundary points. `reshape=false` will
return a 1D array while `reshape=true` will return a 3D array
with size 1×1×Nz.


Examples
========

```jldoctest znodes
julia> using Oceananigans, Oceananigans.Grids

julia> horz_periodic_grid = RegularCartesianGrid(size=(3, 3, 3), extent=(2π, 2π, 1),
                                                 topology=(Periodic, Periodic, Bounded));

julia> zC = znodes(Cell, horz_periodic_grid)
3-element view(OffsetArray(::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}, 0:4), 1:3) with eltype Float64:
 -0.8333333333333331
 -0.4999999999999999
 -0.16666666666666652
```

``` jldoctest znodes
julia> zF = znodes(Face, horz_periodic_grid)
4-element view(OffsetArray(::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}, 0:5), 1:4) with eltype Float64:
 -1.0
 -0.6666666666666666
 -0.33333333333333337
 -4.44089209850063e-17
```
"""
znodes(::Type{Cell}, grid; reshape=false) =
    reshape ? Base.reshape(view(grid.zC, 1:grid.Nz), 1, 1, grid.Nz) :
              view(grid.zC, 1:grid.Nz)

function xnodes(::Type{Face}, grid; reshape=false)
    xF = view(grid.xF, interior_indices(Face, topology(grid, 1), grid.Nx))
    return reshape ? Base.reshape(xF, length(xF), 1, 1) : xF
end

function ynodes(::Type{Face}, grid; reshape=false)
    yF = view(grid.yF, interior_indices(Face, topology(grid, 2), grid.Ny))
    return reshape ? Base.reshape(yF, 1, length(yF), 1) : yF
end

function znodes(::Type{Face}, grid; reshape=false)
    zF = view(grid.zF, interior_indices(Face, topology(grid, 3), grid.Nz))
    return reshape ? Base.reshape(zF, 1, 1, length(zF)) : zF
end

"""
    nodes(loc, grid; reshape=false)

Returns a 3-tuple of views over the interior nodes
at the locations in `loc` in `x, y, z`.

If `reshape=true`, the views are reshaped to 3D arrays
with non-singleton dimensions 1, 2, 3 for `x, y, z`, respectively.
These reshaped arrays can then be used in broadcast operations with 3D fields
or arrays.

See `xnodes`, `ynodes`, and `znodes`.
"""
function nodes(loc, grid::AbstractGrid; reshape=false)
    if reshape
        x, y, z = nodes(loc, grid; reshape=false)

        N = (length(x), length(y), length(z))

        x = Base.reshape(x, N[1], 1, 1)
        y = Base.reshape(y, 1, N[2], 1)
        z = Base.reshape(z, 1, 1, N[3])

        return (x, y, z)
    else
        return (xnodes(loc[1], grid),
                ynodes(loc[2], grid),
                znodes(loc[3], grid))
    end
end

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
