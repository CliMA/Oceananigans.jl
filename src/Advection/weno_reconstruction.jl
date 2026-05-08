#####
##### Weighted Essentially Non-Oscillatory (WENO) advection scheme
#####

struct WENO{N, FT, WCT, PP, SI, M} <: AbstractUpwindBiasedAdvectionScheme{N, FT, M}

    "Bounds for maximum-principle-satisfying WENO scheme"
    bounds :: PP

    "Reconstruction scheme used for symmetric interpolation"
    advecting_velocity_scheme :: SI

    function WENO{N, FT, WCT, M}(bounds::PP,
                                 advecting_velocity_scheme :: SI) where {N, FT, WCT, M, PP, SI}

        return new{N, FT, WCT, PP, SI, M}(bounds, advecting_velocity_scheme)
    end
end

"""
    WENO([FT=Float64;]
         order = 5,
         bounds = nothing,
         weight_computation = Nothing,
         minimum_buffer_upwind_order = 1)

Construct a weighted essentially non-oscillatory advection scheme of order `order` with precision `FT`.

Arguments
=========

- `FT`: The floating point type used in the scheme. Default: `Oceananigans.defaults.FloatType`

Keyword arguments
=================
- `weight_computation`: The type of approximate division to used when computing WENO weights.
                        Default: `Nothing` (deferred; a architecture-dependent default is assigned in
                        `materialize_advection`)
- `order`: The order of the WENO advection scheme. Default: 5
- `bounds` (experimental): Whether to use bounds-preserving WENO, which produces a reconstruction
                           that attempts to restrict a quantity to lie between a `bounds` tuple.
                           Default: `nothing`, which does not use a boundary-preserving scheme.
- `weight_computation`: The type of approximate division used when computing WENO weights. One of
                        `NormalDivision`, `BackendOptimizedDivision`, `ConvertingDivision{FT}`.
                        Default: `Nothing` (deferred; an architecture-dependent default is assigned
                        in `materialize_advection`).
- `minimum_buffer_upwind_order`: The minimum buffer for the upwind reconstruction near boundaries.
                                 When the reduced order near a boundary would fall below this value,
                                 the reconstruction falls back to centered 2nd-order interpolation
                                 instead of continuing to decrease the upwind order.
                                 Must be between 1 and `(order + 1) ÷ 2`.
                                 Default: 1 (preserves upwind behavior at all boundaries).

Examples
========

To build the default 5th-order scheme:

```jldoctest weno
julia> using Oceananigans

julia> WENO()
WENO{3, Float64, Nothing}(order=5)
└── advection_velocity_scheme: Centered(order=4)
```

To build a 9th-order scheme (often a good choice for a stable
yet minimally-dissipative advection scheme):

```jldoctest weno
julia> WENO(order=9)
WENO{5, Float64, Nothing}(order=9)
└── advection_velocity_scheme: Centered(order=8)
```

```jldoctest weno
julia> WENO(order=9, bounds=(0, 1))
WENO{5, Float64, Nothing}(order=9, bounds=(0.0, 1.0))
├── bounds: (0.0, 1.0)
└── advection_velocity_scheme: Centered(order=8)
```

To build a WENO scheme that uses approximate division on a GPU to execute faster:
```jldoctest weno
julia> WENO(;weight_computation=Oceananigans.Utils.BackendOptimizedDivision)
WENO{3, Float64, Oceananigans.Utils.BackendOptimizedDivision}(order=5)
├── buffer_scheme: WENO{2, Float64, Oceananigans.Utils.BackendOptimizedDivision}(order=3)
│   └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=4)
```
"""
function WENO(FT::DataType=Oceananigans.defaults.FloatType;
              order = 5,
              bounds = nothing,
              weight_computation::Type = Nothing,
              minimum_buffer_upwind_order = 1)

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
        return WENO{N, FT, weight_computation, minimum_buffer_upwind_order}(bounds, advecting_velocity_scheme)
    end
end

weno_order(::WENO{N}) where N = 2N-1
Base.eltype(::WENO{N, FT}) where {N, FT} = FT
weight_computation_type(::WENO{N, FT, WCT}) where {N, FT, WCT} = WCT
Base.summary(a::WENO{N, FT, WCT, Nothing}) where {N, FT, WCT} = string("WENO{$N, $FT, $WCT}(order=", 2N-1, ")")
Base.summary(a::WENO{N, FT, WCT, PP}) where {N, FT, WCT, PP} = string("WENO{$N, $FT, $WCT}(order=", 2N-1, ", bounds=", string(a.bounds), ")")

function Base.show(io::IO, a::WENO)
    print(io, summary(a), '\n')

    if !isnothing(a.bounds)
        print(io, "├── bounds: ", string(a.bounds), '\n')
    end

    if minimum_buffer_upwind_order(a) > 1
        print(io, "├── minimum_buffer_upwind_order: ", minimum_buffer_upwind_order(a), '\n')
    end

    print(io, "└── advection_velocity_scheme: ", summary(a.advecting_velocity_scheme))
end

Adapt.adapt_structure(to, scheme::WENO{N, FT, WCT, PP, SI, M}) where {N, FT, WCT, PP, SI, M} =
     WENO{N, FT, WCT, M}(Adapt.adapt(to, scheme.bounds),
                         Adapt.adapt(to, scheme.advecting_velocity_scheme))

Architectures.on_architecture(to, scheme::WENO{N, FT, WCT, PP, SI, M}) where {N, FT, WCT, PP, SI, M} =
    WENO{N, FT, WCT, M}(on_architecture(to, scheme.bounds),
                        on_architecture(to, scheme.advecting_velocity_scheme))
