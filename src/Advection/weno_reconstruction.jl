import Oceananigans

#####
##### Weighted Essentially Non-Oscillatory (WENO) advection scheme
#####

struct WENO{N, FT, FT2, NDC, PP, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT}
    bounds :: PP
    buffer_scheme :: CA
    advecting_velocity_scheme :: SI

    function WENO{N, FT, FT2, NDC}(bounds::PP, buffer_scheme::CA,
                              advecting_velocity_scheme :: SI) where {N, FT, FT2, NDC, PP, CA, SI}

        return new{N, FT, FT2, NDC, PP, CA, SI}(bounds, buffer_scheme, advecting_velocity_scheme)
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
- `newton_div`: The type of approximate division to use in performance-critical parts of the scheme.
                  Default: `Oceananigans.Utils.NewtonDivWithConversion{FT2}`
- `order`: The order of the WENO advection scheme. Default: 5
- `bounds` (experimental): Whether to use bounds-preserving WENO, which produces a reconstruction
                           that attempts to restrict a quantity to lie between a `bounds` tuple.
                           Default: `nothing`, which does not use a boundary-preserving scheme.
- `minimum_buffer_upwind_order`: The minimum upwind order for buffer schemes. When the buffer
                                 scheme order reaches this value, subsequent buffers use
                                 `Centered(order=2)` instead of continuing to decrease the
                                 upwind order. Default: 1 (preserves existing behavior).

Examples
========

To build the default 5th-order scheme:

```jldoctest weno
julia> using Oceananigans

julia> WENO()
WENO{3, Float64, Float32}(order=5)
├── buffer_scheme: WENO{2, Float64, Float32}(order=3)
│   └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=4)
```

To build a 9th-order scheme (often a good choice for a stable
yet minimally-dissipative advection scheme):

```jldoctest weno
julia> WENO(order=9)
WENO{5, Float64, Float32}(order=9)
├── buffer_scheme: WENO{4, Float64, Float32}(order=7)
│   └── buffer_scheme: WENO{3, Float64, Float32}(order=5)
│       └── buffer_scheme: WENO{2, Float64, Float32}(order=3)
│           └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=8)
```

To build a 9th-order scheme with `minimum_buffer_upwind_order=5`,
which uses `Centered(order=2)` as the innermost buffer scheme:

```jldoctest weno
julia> WENO(order=9, minimum_buffer_upwind_order=5)
WENO{5, Float64, Float32}(order=9)
├── buffer_scheme: WENO{4, Float64, Float32}(order=7)
│   └── buffer_scheme: WENO{3, Float64, Float32}(order=5)
│       └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=8)
```

```jldoctest weno
julia> WENO(order=9, bounds=(0, 1))
WENO{5, Float64, Float32}(order=9, bounds=(0.0, 1.0))
├── buffer_scheme: WENO{4, Float64, Float32}(order=7, bounds=(0.0, 1.0))
│   └── buffer_scheme: WENO{3, Float64, Float32}(order=5, bounds=(0.0, 1.0))
│       └── buffer_scheme: WENO{2, Float64, Float32}(order=3, bounds=(0.0, 1.0))
│           └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=8)
```
"""
function WENO(FT::DataType=Oceananigans.defaults.FloatType,
              FT2::DataType=Float32;
              newton_div::DataType=Oceananigans.Utils.NewtonDivWithConversion{FT2},
              order = 5,
              buffer_scheme = DecreasingOrderAdvectionScheme(),
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

        if buffer_scheme isa DecreasingOrderAdvectionScheme
            if order ≤ minimum_buffer_upwind_order
                # At minimum order, switch to Centered scheme
                buffer_scheme = Centered(FT; order=2)
            else
                buffer_scheme = WENO(FT, FT2; order=order-2, bounds, minimum_buffer_upwind_order)
            end
        end

        N = Int((order + 1) ÷ 2)
        return WENO{N, FT, FT2, newton_div}(bounds, buffer_scheme, advecting_velocity_scheme)
    end
end

weno_order(::WENO{N}) where N = 2N-1
Base.eltype(::WENO{N, FT}) where {N, FT} = FT
eltype2(::WENO{N, FT, FT2}) where {N, FT, FT2} = FT2
Base.summary(a::WENO{N, FT, FT2, NDC, Nothing}) where {N, FT, FT2, NDC} = string("WENO{$N, $FT, $FT2, $NDC}(order=", 2N-1, ")")
Base.summary(a::WENO{N, FT, FT2, NDC, PP}) where {N, FT, FT2, NDC, PP} = string("WENO{$N, $FT, $FT2, $NDC}(order=", 2N-1, ", bounds=", string(a.bounds), ")")

function Base.show(io::IO, a::WENO)
    print(io, summary(a), '\n')

    # Print buffer scheme tree recursively
    if !isnothing(a.buffer_scheme)
        print_buffer_scheme_tree(io, a.buffer_scheme, "", false)
        println(io)
    else
        print(io, "├── buffer_scheme: ", summary(a.buffer_scheme), '\n')
    end

    print(io, "└── advecting_velocity_scheme: ", summary(a.advecting_velocity_scheme))
end

Adapt.adapt_structure(to, scheme::WENO{N, FT, FT2, NDC}) where {N, FT, FT2, NDC} =
     WENO{N, FT, FT2, NDC}(Adapt.adapt(to, scheme.bounds),
                           Adapt.adapt(to, scheme.buffer_scheme),
                           Adapt.adapt(to, scheme.advecting_velocity_scheme))

on_architecture(to, scheme::WENO{N, FT, FT2, NDC}) where {N, FT, FT2, NDC} =
    WENO{N, FT, FT2, NDC}(on_architecture(to, scheme.bounds),
                          on_architecture(to, scheme.buffer_scheme),
                          on_architecture(to, scheme.advecting_velocity_scheme))
