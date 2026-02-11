import Oceananigans

#####
##### Weighted Essentially Non-Oscillatory (WENO) advection scheme
#####

struct WENO{N, FT, FT2, PP, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT}

    "Bounds for maximum-principle-satisfying WENO scheme"
    bounds :: PP

    "Reconstruction scheme used for symmetric interpolation"
    advecting_velocity_scheme :: SI

    "Minimum buffer for the reduced upwind order near boundaries (below this, use centered 2nd-order)"
    minimum_buffer_upwind_order :: Int

    function WENO{N, FT, FT2}(bounds::PP,
                              advecting_velocity_scheme :: SI,
                              minimum_buffer_upwind_order :: Int) where {N, FT, FT2, PP, SI}

        return new{N, FT, FT2, PP, SI}(bounds, advecting_velocity_scheme, minimum_buffer_upwind_order)
    end
end

"""
    WENO([FT=Float64, FT2=Float32;]
         order = 5,
         bounds = nothing,
         minimum_buffer_upwind_order = 3)

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
- `minimum_buffer_upwind_order`: The minimum buffer for the upwind reconstruction near boundaries.
                                 When the reduced order near a boundary would fall below this value,
                                 the reconstruction falls back to centered 2nd-order interpolation
                                 instead of continuing to decrease the upwind order.
                                 Must be between 1 and `(order + 1) ÷ 2`.
                                 Default: 1 (preserves existing behavior).

Examples
========

To build the default 5th-order scheme:

```jldoctest weno
julia> using Oceananigans

julia> WENO()
WENO{3, Float64, Float32}(order=5)
└── advection_velocity_scheme: Centered(order=4)
```

To build a 9th-order scheme (often a good choice for a stable
yet minimally-dissipative advection scheme):

```jldoctest weno
julia> WENO(order=9)
WENO{5, Float64, Float32}(order=9)
└── advection_velocity_scheme: Centered(order=8)
```

```jldoctest weno
julia> WENO(order=9, bounds=(0, 1))
WENO{5, Float64, Float32}(order=9)
├── bounds: (0, 1)
└── advection_velocity_scheme: Centered(order=8)
```
"""
function WENO(FT::DataType=Oceananigans.defaults.FloatType, FT2::DataType=Float32;
              order = 5,
              bounds = nothing,
              minimum_buffer_upwind_order = 3)

    mod(order, 2) == 0 && throw(ArgumentError("WENO reconstruction scheme is defined only for odd orders"))

    if !isnothing(bounds)
        bounds isa NTuple{2} || throw(ArgumentError("bounds must be nothing or a tuple of two values"))
        bounds = (convert(FT, bounds[1]), convert(FT, bounds[2]))
    end

    if order < 3
        # WENO(order=1) is equivalent to UpwindBiased(order=1)
        return UpwindBiased(FT; order=1)
    else
        advecting_velocity_scheme = Centered(FT; order=order-1)

        N = Int((order + 1) ÷ 2)
        minimum_buffer_upwind_order = max(1, min(N, Int(minimum_buffer_upwind_order)))
        return WENO{N, FT, FT2}(bounds, advecting_velocity_scheme, minimum_buffer_upwind_order)
    end
end

weno_order(::WENO{N}) where N = 2N-1
Base.eltype(::WENO{N, FT}) where {N, FT} = FT
eltype2(::WENO{N, FT, FT2}) where {N, FT, FT2} = FT2
Base.summary(a::WENO{N, FT, FT2, Nothing}) where {N, FT, FT2} = string("WENO{$N, $FT, $FT2}(order=", 2N-1, ")")
Base.summary(a::WENO{N, FT, FT2, PP}) where {N, FT, FT2, PP} = string("WENO{$N, $FT, $FT2}(order=", 2N-1, ", bounds=", string(a.bounds), ")")

function Base.show(io::IO, a::WENO)
    print(io, summary(a), '\n')

    if !isnothing(a.bounds)
        print(io, "├── bounds: ", string(a.bounds), '\n')
    end

    if a.minimum_buffer_upwind_order > 1
        print(io, "├── minimum_buffer_upwind_order: ", a.minimum_buffer_upwind_order, '\n')
    end

    print(io, "└── advection_velocity_scheme: ", summary(a.advecting_velocity_scheme))
end

Adapt.adapt_structure(to, scheme::WENO{N, FT, FT2}) where {N, FT, FT2} =
     WENO{N, FT, FT2}(Adapt.adapt(to, scheme.bounds),
                      Adapt.adapt(to, scheme.advecting_velocity_scheme),
                      scheme.minimum_buffer_upwind_order)

on_architecture(to, scheme::WENO{N, FT, FT2}) where {N, FT, FT2} =
    WENO{N, FT, FT2}(on_architecture(to, scheme.bounds),
                     on_architecture(to, scheme.advecting_velocity_scheme),
                     scheme.minimum_buffer_upwind_order)
