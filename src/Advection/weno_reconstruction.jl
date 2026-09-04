#####
##### Weighted Essentially Non-Oscillatory (WENO) advection scheme
#####

struct WENO{N, FT, WCT, TD, PP, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT, TD}
    bounds :: PP
    buffer_scheme :: CA
    advecting_velocity_scheme :: SI
    time_discretization :: TD
    function WENO{N, FT, WCT}(bounds::PP, buffer_scheme::CA,
                              advecting_velocity_scheme :: SI,
                              time_discretization :: TD) where {N, FT, WCT, PP, CA, SI, TD}

        return new{N, FT, WCT, TD, PP, CA, SI}(bounds, buffer_scheme, advecting_velocity_scheme, time_discretization)
    end
end

"""
    WENO([FT=Oceananigans.defaults.FloatType;]
         weight_computation=Nothing,
         order = 5,
         buffer_scheme = DecreasingOrderAdvectionScheme(),
         time_discretization = ExplicitTimeDiscretization(),
         bounds = nothing)

Construct a weighted essentially non-oscillatory advection scheme of order `order` with precision `FT`.

Arguments
=========

- `FT`: The floating point type used in the scheme. Default: `Oceananigans.defaults.FloatType`

Keyword arguments
=================
- `weight_computation`: The type of approximate division to used when computing WENO weights.
                        Default: `Nothing` (deferred; a architecture-dependent default is assigned in
                        `materialize_advection`)
- `order`: The order of the WENO advection scheme. Default: 5.
- `bounds` (experimental): Whether to use bounds-preserving WENO, which produces a reconstruction
                           that attempts to restrict a quantity to lie between a `bounds` tuple.
                           Default: `nothing`, which does not use a boundary-preserving scheme.
- `buffer_scheme`: The scheme used where the stencil does not fit. `DecreasingOrderAdvectionScheme()` steps the
                   order down by two until third order and then uses its `boundary_scheme`, which defaults to
                   `Centered(order=2)`.

Examples
========

To build the default 5th-order scheme:

```jldoctest weno
julia> using Oceananigans

julia> WENO()
WENO{3, Float64, Nothing}(order=5)
├── buffer_scheme: WENO{2, Float64, Nothing}(order=3)
│   └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=4)
```

To build a 9th-order scheme (often a good choice for a stable
yet minimally-dissipative advection scheme):

```jldoctest weno
julia> WENO(order=9)
WENO{5, Float64, Nothing}(order=9)
├── buffer_scheme: WENO{4, Float64, Nothing}(order=7)
│   └── buffer_scheme: WENO{3, Float64, Nothing}(order=5)
│       └── buffer_scheme: WENO{2, Float64, Nothing}(order=3)
│           └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=8)
```

To terminate the chain in a boundary reconstruction other than the default:

```jldoctest weno
julia> WENO(order=7, buffer_scheme=DecreasingOrderAdvectionScheme(boundary_scheme=UpwindBiased(order=1)))
WENO{4, Float64, Nothing}(order=7)
├── buffer_scheme: WENO{3, Float64, Nothing}(order=5)
│   └── buffer_scheme: WENO{2, Float64, Nothing}(order=3)
│       └── buffer_scheme: UpwindBiased(order=1)
└── advecting_velocity_scheme: Centered(order=6)
```

```jldoctest weno
julia> WENO(order=9, bounds=(0, 1))
WENO{5, Float64, Nothing}(order=9, bounds=(0.0, 1.0))
├── buffer_scheme: WENO{4, Float64, Nothing}(order=7, bounds=(0.0, 1.0))
│   └── buffer_scheme: WENO{3, Float64, Nothing}(order=5, bounds=(0.0, 1.0))
│       └── buffer_scheme: WENO{2, Float64, Nothing}(order=3, bounds=(0.0, 1.0))
│           └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=8)
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
              weight_computation::DataType=Nothing,
              order = 5,
              buffer_scheme = DecreasingOrderAdvectionScheme(),
              time_discretization = ExplicitTimeDiscretization(),
              bounds = nothing)

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
            if order ≤ 3 # the boundary reconstruction takes over here
                buffer_scheme = something(buffer_scheme.boundary_scheme, Centered(FT; order=2))
            else
                buffer_scheme = WENO(FT; order=order-2, bounds, weight_computation,
                                     buffer_scheme = DecreasingOrderAdvectionScheme(; boundary_scheme = buffer_scheme.boundary_scheme))
            end
        end

        N = Int((order + 1) ÷ 2)
        return WENO{N, FT, weight_computation}(bounds, buffer_scheme, advecting_velocity_scheme, time_discretization)
    end
end

weno_order(::WENO{N}) where N = 2N-1
Base.eltype(::WENO{N, FT}) where {N, FT} = FT
Base.summary(a::WENO{N, FT, WCT, TD, Nothing}) where {N, FT, WCT, TD} = string("WENO{$N, $FT, $WCT}(order=", 2N-1, ")")
Base.summary(a::WENO{N, FT, WCT, TD, PP}) where {N, FT, WCT, TD, PP} = string("WENO{$N, $FT, $WCT}(order=", 2N-1, ", bounds=", string(a.bounds), ")")

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

Adapt.adapt_structure(to, scheme::WENO{N, FT, WCT}) where {N, FT, WCT} =
     WENO{N, FT, WCT}(Adapt.adapt(to, scheme.bounds),
                      Adapt.adapt(to, scheme.buffer_scheme),
                      Adapt.adapt(to, scheme.advecting_velocity_scheme),
                      Adapt.adapt(to, scheme.time_discretization))

Architectures.on_architecture(to, scheme::WENO{N, FT, WCT}) where {N, FT, WCT} =
    WENO{N, FT, WCT}(on_architecture(to, scheme.bounds),
                     on_architecture(to, scheme.buffer_scheme),
                     on_architecture(to, scheme.advecting_velocity_scheme),
                     on_architecture(to, scheme.time_discretization))

# Select the default WENO weight computation
# Specific backends may override
default_weno_weight_computation(arch) = Oceananigans.Utils.BackendOptimizedDivision
