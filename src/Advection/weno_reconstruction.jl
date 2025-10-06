import Oceananigans

#####
##### Weighted Essentially Non-Oscillatory (WENO) advection scheme
#####

struct WENO{N, FT, FT2, PP, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT}

    "Bounds for maximum-principle-satisfying WENO scheme"
    bounds :: PP

    "Advection scheme used near boundaries"
    buffer_scheme :: CA
    "Reconstruction scheme used for symmetric interpolation"
    advecting_velocity_scheme :: SI

    function WENO{N, FT, FT2}(bounds::PP, buffer_scheme::CA,
                              advecting_velocity_scheme :: SI) where {N, FT, FT2, PP, CA, SI}

        return new{N, FT, FT2, PP, CA, SI}(bounds, buffer_scheme, advecting_velocity_scheme)
    end
end

"""
    WENO([FT=Float64, FT2=Float32;]
         order = 5,
         bounds = nothing)

Construct a weighted essentially non-oscillatory advection scheme of order `order` with precision `FT`.

Arguments
=========

- `FT`: The floating point type used in the scheme. Default: `Oceananigans.defaults.FloatType`
- `FT2`: The floating point type used in some performance-critical parts of the scheme. Default: `Float32`

Keyword arguments
=================

- `order`: The order of the WENO advection scheme. Default: 5
- `bounds` (experimental): Whether to use bounds-preserving WENO, which produces a reconstruction
                           that attempts to restrict a quantity to lie between a `bounds` tuple.
                           Default: `nothing`, which does not use a boundary-preserving scheme.

Examples
========

To build the default 5th-order scheme:

```jldoctest weno
julia> using Oceananigans

julia> WENO()
WENO{3, Float64, Float32}(order=5)
├── buffer_scheme: WENO{2, Float64, Float32}(order=3)
└── advection_velocity_scheme: Centered(order=4)
```

To build a 9th-order scheme (often a good choice for a stable
yet minimally-dissipative advection scheme):

```jldoctest weno
julia> WENO(order=9)
WENO{5, Float64, Float32}(order=9)
├── buffer_scheme: WENO{4, Float64, Float32}(order=7)
└── advection_velocity_scheme: Centered(order=8)
```

```jldoctest weno
julia> WENO(order=9, bounds=(0, 1))
WENO{5, Float64, Float32}(order=9)
├── bounds: (0, 1)
├── buffer_scheme: WENO{4, Float64, Float32}(order=7)
└── advection_velocity_scheme: Centered(order=8)
```
"""
function WENO(FT::DataType=Oceananigans.defaults.FloatType, FT2::DataType=Float32;
              order = 5,
              bounds = nothing)

    mod(order, 2) == 0 && throw(ArgumentError("WENO reconstruction scheme is defined only for odd orders"))

    if order < 3
        # WENO(order=1) is equivalent to UpwindBiased(order=1)
        return UpwindBiased(FT; order=1)
    else
        advecting_velocity_scheme = Centered(FT; order=order-1)
        buffer_scheme = WENO(FT, FT2; order=order-2, bounds)

        N = Int((order + 1) ÷ 2)
        return WENO{N, FT, FT2}(bounds, buffer_scheme, advecting_velocity_scheme)
    end
end

weno_order(::WENO{N}) where N = 2N-1
Base.eltype(::WENO{N, FT}) where {N, FT} = FT
eltype2(::WENO{N, FT, FT2}) where {N, FT, FT2} = FT2
Base.summary(a::WENO{N, FT, FT2}) where {N, FT, FT2} = string("WENO{$N, $FT, $FT2}(order=", 2N-1, ")")

function Base.show(io::IO, a::WENO)
    print(io, summary(a), '\n')

    if !isnothing(a.bounds)
        print(io, "├── bounds: ", string(a.bounds), '\n')
    end

    print(io, "├── buffer_scheme: ", summary(a.buffer_scheme) , '\n',
              "└── advection_velocity_scheme: ", summary(a.advecting_velocity_scheme))
end

Adapt.adapt_structure(to, scheme::WENO{N, FT, FT2}) where {N, FT, FT2} =
     WENO{N, FT, FT2}(Adapt.adapt(to, scheme.bounds),
                      Adapt.adapt(to, scheme.buffer_scheme),
                      Adapt.adapt(to, scheme.advecting_velocity_scheme))

on_architecture(to, scheme::WENO{N, FT, FT2}) where {N, FT, FT2} =
    WENO{N, FT, FT2}(on_architecture(to, scheme.bounds),
                     on_architecture(to, scheme.buffer_scheme),
                     on_architecture(to, scheme.advecting_velocity_scheme))

