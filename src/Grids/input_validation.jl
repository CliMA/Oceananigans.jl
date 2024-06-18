using Oceananigans: tupleit

#####
##### Some validation tools
#####

# Tuple inflation/deflation for topologies with Flat dimensions

"adds tup element with `default` value for flat dimensions"
inflate_tuple(TX, TY, TZ, tup; default) = tup

inflate_tuple(::Type{Flat}, TY, TZ, tup; default) = (default, tup[1], tup[2])
inflate_tuple(TY, ::Type{Flat}, TZ, tup; default) = (tup[1], default, tup[2])
inflate_tuple(TY, TZ, ::Type{Flat}, tup; default) = (tup[1], tup[2], default)

inflate_tuple(TX, ::Type{Flat}, ::Type{Flat}, tup; default) = (tup[1], default, default)
inflate_tuple(::Type{Flat}, TY, ::Type{Flat}, tup; default) = (default, tup[1], default)
inflate_tuple(::Type{Flat}, ::Type{Flat}, TZ, tup; default) = (default, default, tup[1])

inflate_tuple(::Type{Flat}, ::Type{Flat}, ::Type{Flat}, tup; default) = (default, default, default)

"removes tup elements that correspond to flat dimensions"
deflate_tuple(TX, TY, TZ, tup) = tup

deflate_tuple(::Type{Flat}, TY, TZ, tup) = tuple(tup[2], tup[3])
deflate_tuple(TY, ::Type{Flat}, TZ, tup) = tuple(tup[1], tup[3])
deflate_tuple(TY, TZ, ::Type{Flat}, tup) = tuple(tup[1], tup[2])

deflate_tuple(TX, ::Type{Flat}, ::Type{Flat}, tup) = (tup[1],)
deflate_tuple(::Type{Flat}, TY, ::Type{Flat}, tup) = (tup[2],)
deflate_tuple(::Type{Flat}, ::Type{Flat}, TZ, tup) = (tup[3],)

deflate_tuple(::Type{Flat}, ::Type{Flat}, ::Type{Flat}, tup) = ()

topological_tuple_length(TX, TY, TZ) = sum(T === Flat ? 0 : 1 for T in (TX, TY, TZ))

"""Validate that an argument tuple is the right length and has elements of type `argtype`."""
function validate_tupled_argument(arg, argtype, argname, len=3; greater_than=0)
    length(arg) == len       || throw(ArgumentError("length($argname) must be $len."))
    all(isa.(arg, argtype))  || throw(ArgumentError("$argname=$arg must contain $(argtype)s."))
    all(arg .> greater_than) || throw(ArgumentError("Elements of $argname=$arg must be > $(greater_than)!"))
    return nothing
end

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

function validate_size(TX, TY, TZ, sz)
    sz = tupleit(sz)
    validate_tupled_argument(sz, Integer, "size", topological_tuple_length(TX, TY, TZ))
    return inflate_tuple(TX, TY, TZ, sz, default=1)
end

# Note that if provided with halo=nothing, the default halo size for coord i
# is the min(default_halo_size, size[i]).
# While this is easy to change, many of tests might fail so this situation needs to be
# cleaned up.
function validate_halo(TX, TY, TZ, size, ::Nothing)
    maximum_halo = size
    default_halo = (3, 3, 3)
    halo = map(min, default_halo, maximum_halo)
    halo = deflate_tuple(TX, TY, TZ, halo)
    return validate_halo(TX, TY, TZ, size, halo)
end

coordinate_name(i) = i == 1 ? "x" : i == 2 ? "y" : "z"

function validate_halo(TX, TY, TZ, size, halo)
    halo = tupleit(halo)
    validate_tupled_argument(halo, Integer, "halo", topological_tuple_length(TX, TY, TZ))
    halo = inflate_tuple(TX, TY, TZ, halo, default=0)

    for i in 1:2
        !(halo[i] ≤ size[i]) && throw(ArgumentError("halo must be ≤ size for coordinate $(coordinate_name(i))"))
    end

    return halo
end

function validate_dimension_specification(T, ξ, dir, N, FT)

    isnothing(ξ)         && throw(ArgumentError("Must supply extent or $dir keyword when $dir-direction is $T"))
    length(ξ) == 2       || throw(ArgumentError("$dir length($ξ) must be 2."))
    all(isa.(ξ, Number)) || throw(ArgumentError("$dir=$ξ should contain numbers."))
    ξ[2] ≥ ξ[1]          || throw(ArgumentError("$dir=$ξ should be an increasing interval."))

    return FT.(ξ)
end

function validate_rectilinear_domain(TX, TY, TZ, FT, size, extent, x, y, z)

    # Find domain endpoints or domain extent, depending on user input:
    if !isnothing(extent) # the user has specified an extent!

        (!isnothing(x) || !isnothing(y) || !isnothing(z)) &&
            throw(ArgumentError("Cannot specify both 'extent' and 'x, y, z' keyword arguments."))

        extent = tupleit(extent)
        validate_tupled_argument(extent, Number, "extent", topological_tuple_length(TX, TY, TZ))
        Lx, Ly, Lz = extent = inflate_tuple(TX, TY, TZ, extent, default=0)

        # An "oceanic" default domain:
        x = TX() isa Flat ? nothing : (zero(FT), convert(FT, Lx))
        y = TY() isa Flat ? nothing : (zero(FT), convert(FT, Ly))
        z = TZ() isa Flat ? nothing : (-convert(FT, Lz), zero(FT))

    else # isnothing(extent) === true implies that user has not specified a length
        x = validate_dimension_specification(TX, x, :x, size[1], FT)
        y = validate_dimension_specification(TY, y, :y, size[2], FT)
        z = validate_dimension_specification(TZ, z, :z, size[3], FT)
    end

    return x, y, z
end

function validate_dimension_specification(T, ξ::AbstractVector, dir, N, FT)
    ξ = FT.(ξ)

    ξ[end] ≥ ξ[1] || throw(ArgumentError("$dir=$ξ should have increasing values."))

    # Validate the length of ξ: error is ξ is too short, warn if ξ is too long.
    Nξ = length(ξ)
    N⁺¹ = N + 1
    if Nξ < N⁺¹
        throw(ArgumentError("length($dir) = $Nξ has too few interfaces for the dimension size $(N)!"))
    elseif Nξ > N⁺¹
        msg = "length($dir) = $Nξ is greater than $N+1, where $N was passed to `size`.\n" *
              "$dir cell interfaces will be constructed from $dir[1:$N⁺¹]."
        @warn msg
    end

    return ξ
end

function validate_dimension_specification(T, ξ::Function, dir, N, FT)
    ξ(N) ≥ ξ(1) || throw(ArgumentError("$dir should have increasing values."))
    return ξ
end

validate_dimension_specification(::Type{Flat}, ξ::AbstractVector, dir, N, FT) = (FT(ξ[1]), FT(ξ[1]))
validate_dimension_specification(::Type{Flat}, ξ::Function,       dir, N, FT) = (FT(ξ(1)), FT(ξ(1)))
validate_dimension_specification(::Type{Flat}, ξ::Tuple,  dir, N, FT) = map(FT, ξ)
validate_dimension_specification(::Type{Flat}, ::Nothing, dir, N, FT) = nothing
validate_dimension_specification(::Type{Flat}, ξ::Number, dir, N, FT) = convert(FT, ξ)

default_horizontal_extent(T, extent) = (0, extent[i])
default_vertical_extent(T, extent) = (-extent[3], 0)

function validate_vertically_stretched_grid_xy(TX, TY, FT, x, y)
    x = validate_dimension_specification(TX, x, :x, FT)
    y = validate_dimension_specification(TY, y, :y, FT)

    Lx = x[2] - x[1]
    Ly = y[2] - y[1]

    return FT(Lx), FT(Ly), FT.(x), FT.(y)
end

validate_unit_vector(ê::ZDirection, FT::DataType=Float64) = ê
validate_unit_vector(ê::NegativeZDirection, FT::DataType=Float64) = ê

function validate_unit_vector(ê, FT::DataType=Float64)
    length(ê) == 3 || throw(ArgumentError("unit vector must have length 3"))

    ex, ey, ez = ê

    ex^2 + ey^2 + ez^2 ≈ 1 ||
        throw(ArgumentError("unit vector `ê` must satisfy ê[1]² + ê[2]² + ê[3]² ≈ 1"))

    return tuple(FT(ex), FT(ey), FT(ez))
end

function validate_index(idx, loc, topo, N, H)
    isinteger(idx) && return validate_index(Int(idx), loc, topo, N, H)
    return throw(ArgumentError("$idx are not supported window indices for Field!"))
end

validate_index(::Colon, loc, topo, N, H) = Colon()
validate_index(idx::UnitRange, ::Nothing, topo, N, H) = UnitRange(1, 1)

function validate_index(idx::UnitRange, loc, topo, N, H)
    all_idx = all_indices(loc, topo, N, H)
    (first(idx) ∈ all_idx && last(idx) ∈ all_idx) || throw(ArgumentError("The indices $idx must slice $all_idx"))
    return idx
end

validate_index(idx::Int, args...) = validate_index(UnitRange(idx, idx), args...)

validate_indices(indices, loc, grid::AbstractGrid) =
    validate_indices(indices, loc, topology(grid), size(grid, loc), halo_size(grid))

validate_indices(indices, loc, topo, sz, halo_sz) =
    map(validate_index, indices, map(instantiate, loc), map(instantiate, topo), sz, halo_sz)
