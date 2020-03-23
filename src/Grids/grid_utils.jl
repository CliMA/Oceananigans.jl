#####
##### Convinience functions
#####

"""
Returns the total extent, including halo regions, of constant-spaced
`Periodic` and `Flat` dimensions with number of halo points `H`, 
constant grid spacing `Δ`, and interior extent `L`.
"""
total_extent(topology, H, Δ, L) = L + (2H - 1) * Δ

"""
Returns the total extent of, including halo regions, of constant-spaced
`Bounded` and `Flat` dimensions with number of halo points `H`,
constant grid spacing `Δ`, and interior extent `L`.
"""
total_extent(::Type{Bounded}, H, Δ, L) = L + 2H * Δ

"""
Returns the total length, including halo points, of a field located at 
`Cell` centers along a grid dimension of length `N` and with halo points `H`.
"""
total_length(loc, topo, N, H=0) = N + 2H

"""
Returns the total length, including halo points, of a field located at 
cell `Face`s along a grid dimension of length `N` and with halo points `H`.
"""
total_length(::Type{Face}, ::Type{Bounded}, N, H=0) = N + 1 + 2H

#####
##### Convinience functions
#####

unpack_grid(grid) = grid.Nx, grid.Ny, grid.Nz, grid.Lx, grid.Ly, grid.Lz

#####
##### Input validation
#####

instantiate_datatype(t::DataType) = t()
instantiate_datatype(t) = t

function validate_topology(topology)
    TX, TY, TZ = topology
    TX = instantiate_datatype(TX)
    TY = instantiate_datatype(TY)
    TZ = instantiate_datatype(TZ)

    for t in (TX, TY, TZ)
        if !isa(t, AbstractTopology)
            e = "$(typeof(t)) is not a valid topology! " *
                "Valid topologies are: Periodic, Bounded, Flat."
            throw(ArgumentError(e))
        end
    end

    return TX, TY, TZ
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

