using Oceananigans: tupleit

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

topological_tuple_length(TX, TY, TZ) = sum(T === Flat ? 0 : 1 for T in (TX, TY, TZ))

"""Validate that an argument tuple is the right length and has elements of type `argtype`."""
function validate_tupled_argument(arg, argtype, argname, len=3)
    length(arg) == len      || throw(ArgumentError("length($argname) must be $len."))
    all(isa.(arg, argtype)) || throw(ArgumentError("$argname=$arg must contain $argtype s."))
    all(arg .> 0)           || throw(ArgumentError("Elements of $argname=$arg must be > 0!"))
    return nothing
end

inflate_tuple(TX, TY, TZ, tup; default) = tup

inflate_tuple(::Type{Flat}, TY, TZ, tup; default) = tuple(default, tup[1], tup[2])
inflate_tuple(TY, ::Type{Flat}, TZ, tup; default) = tuple(tup[1], default, tup[2])
inflate_tuple(TY, TZ, ::Type{Flat}, tup; default) = tuple(tup[1], tup[2], default)

inflate_tuple(TX, ::Type{Flat}, ::Type{Flat}, tup; default) = (tup[1], default, default)
inflate_tuple(::Type{Flat}, TY, ::Type{Flat}, tup; default) = (default, tup[1], default)
inflate_tuple(::Type{Flat}, ::Type{Flat}, TZ, tup; default) = (default, default, tup[1])
inflate_tuple(::Type{Flat}, ::Type{Flat}, ::Type{Flat}, tup; default) = (default, default, default)

function validate_size(TX, TY, TZ, size)
    validate_tupled_argument(size, Integer, "size", topological_tuple_length(TX, TY, TZ))
    size = inflate_tuple(TX, TY, TZ, size, default=1)
    return size
end

function validate_halo(TX, TY, TZ, ::Nothing)
    halo = Tuple(1 for i = 1:topological_tuple_length(TX, TY, TZ))
    return validate_halo(TX, TY, TZ, halo)
end

function validate_halo(TX, TY, TZ, halo)
    validate_tupled_argument(halo, Integer, "halo", topological_tuple_length(TX, TY, TZ))
    halo = inflate_tuple(TX, TY, TZ, halo, default=0)
    return halo
end

coordinate_name(i) = i == 1 ? "x" : i == 2 ? "y" : "z"

function validate_dimension_specification(T, ξ, dir)

    isnothing(ξ)         && throw(ArgumentError("Must supply extent or $dir keyword arguments when the $dir-direction is $T"))
    length(ξ) == 2       || throw(ArgumentError("$dir length($ξ) must be 2."))
    all(isa.(ξ, Number)) || throw(ArgumentError("$dir=$ξ should contain numbers."))
    ξ[2] >= ξ[1]         || throw(ArgumentError("$dir=$ξ should be an increasing interval."))

    return ξ
end

validate_dimension_specification(::Type{Flat}, ξ, dir) = (0, 0)

default_horizontal_extent(T, L) = (0, L)
default_horizontal_extent(::Type{Flat}, L) = (0, 0)

default_vertical_extent(T, L) = (-L, 0)
default_vertical_extent(::Type{Flat}, L) = (0, 0)

validate_xyz(T, ξ, dir) =
    isnothing(ξ) && throw(ArgumentError("Must supply extent or $dir keyword arguments when the $dir-direction is $TX"))

validate_xyz(::Type{Flat}, ξ, dir) = nothing

function validate_grid_size_and_halo(TX, TY, TZ, size, halo)
    size = validate_size(TX, TY, TZ, size)
    halo = validate_halo(TX, TY, TZ, halo)
    return size, halo
end

function validate_regular_grid_extent(TX, TY, TZ, FT, extent, x, y, z)

    # Find domain endpoints or domain extent, depending on user input:
    if !isnothing(extent) # the user has specified an extent!

        (!isnothing(x) || !isnothing(y) || !isnothing(z)) &&
            throw(ArgumentError("Cannot specify both 'extent' and 'x, y, z' keyword arguments."))

        validate_tupled_argument(extent, Number, "extent", topological_tuple_length(TX, TY, TZ))

        Lx, Ly, Lz = extent

        # An "oceanic" default domain:
        x = default_horizontal_extent(TX, Lx)
        y = default_horizontal_extent(TY, Ly)
        z = default_vertical_extent(TZ, Lz)

    else # isnothing(extent) === true implies that user has not specified a length

        x = validate_dimension_specification(TX, x, :x)
        y = validate_dimension_specification(TY, y, :y)
        z = validate_dimension_specification(TZ, z, :z)

        Lx = x[2] - x[1]
        Ly = y[2] - y[1]
        Lz = z[2] - z[1]
    end

    return FT(Lx), FT(Ly), FT(Lz), FT.(x), FT.(y), FT.(z)
end

function validate_vertically_stretched_grid_xy(TX, TY, FT, x, y)
    validate_dimension_specification(TX, x, :x)
    validate_dimension_specification(TY, y, :y)

    Lx = x[2] - x[1]
    Ly = y[2] - y[1]

    return FT(Lx), FT(Ly), FT.(x), FT.(y)
end
