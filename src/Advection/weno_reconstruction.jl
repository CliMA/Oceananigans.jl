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
$(MethodSignatures(; defaults = true))

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
- `bounds` (experimental): A [`BoundsPreservation`](@ref), or a tuple `(cᵐⁱⁿ, cᵐᵃˣ)` as shorthand for
                           `BoundsPreservation(cᵐⁱⁿ, cᵐᵃˣ)`, switching on the maximum-principle-satisfying limiter of
                           [Zhang and Shu (2010)](@cite ZhangShu2010), which rescales the reconstruction of
                           every cell towards the cell mean so that the advective update stays within `bounds`.
                           One rescaling factor is shared by the three directions, so the bound is
                           three-dimensional. The options `maximum_cfl`, `adaptive_weight` and `flux_limiter`
                           are set in the [`BoundsPreservation`](@ref).
                           Default: `nothing`, which does not use a bounds-preserving scheme.
- `minimum_buffer_upwind_order`: The minimum upwind order for buffer schemes. When the buffer
                                 scheme order reaches this value, subsequent buffers use
                                 `Centered(order=2)` instead of continuing to decrease the
                                 upwind order. Default: 3.

Examples
========

To build the default 5th-order scheme:

```jldoctest weno
julia> using Oceananigans

julia> WENO()
WENO{3, Float64, Nothing}(order=5)
├── buffer_scheme: WENO{2, Float64, Nothing}(order=3)
│   └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=2)
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
└── advecting_velocity_scheme: Centered(order=2)
```

To build a 9th-order scheme with `minimum_buffer_upwind_order=5`,
which uses `Centered(order=2)` as the innermost buffer scheme:

```jldoctest weno
julia> WENO(order=9, minimum_buffer_upwind_order=5)
WENO{5, Float64, Nothing}(order=9)
├── buffer_scheme: WENO{4, Float64, Nothing}(order=7)
│   └── buffer_scheme: WENO{3, Float64, Nothing}(order=5)
│       └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=2)
```

```jldoctest weno
julia> WENO(order=9, bounds=(0, 1))
WENO{5, Float64, Nothing}(order=9, bounds=(0.0, 1.0))
├── buffer_scheme: WENO{4, Float64, Nothing}(order=7, bounds=(0.0, 1.0))
│   └── buffer_scheme: WENO{3, Float64, Nothing}(order=5, bounds=(0.0, 1.0))
│       └── buffer_scheme: WENO{2, Float64, Nothing}(order=3, bounds=(0.0, 1.0))
│           └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=2)
```

To also limit the fluxes between cells, so that the tracer stays within `bounds` for any time step:

```jldoctest weno
julia> WENO(order=5, bounds=BoundsPreservation(0, 1; flux_limiter=true))
WENO{3, Float64, Nothing}(order=5, bounds=(0.0, 1.0), flux_limiter=FluxLimiter())
├── buffer_scheme: WENO{2, Float64, Nothing}(order=3, bounds=(0.0, 1.0), flux_limiter=FluxLimiter())
│   └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=2)
```

To let the quadrature weight of the rescaling adapt to each cell's Courant number, so that smooth extrema
touching a bound are not clipped:

```jldoctest weno
julia> WENO(order=5, bounds=BoundsPreservation(0, 1; adaptive_weight=true))
WENO{3, Float64, Nothing}(order=5, bounds=(0.0, 1.0), adaptive_weight=true)
├── buffer_scheme: WENO{2, Float64, Nothing}(order=3, bounds=(0.0, 1.0), adaptive_weight=true)
│   └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=2)
```

To build a WENO scheme that uses approximate division on a GPU to execute faster:
```jldoctest weno
julia> WENO(;weight_computation=Oceananigans.Utils.BackendOptimizedDivision)
WENO{3, Float64, Oceananigans.Utils.BackendOptimizedDivision}(order=5)
├── buffer_scheme: WENO{2, Float64, Oceananigans.Utils.BackendOptimizedDivision}(order=3)
│   └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=2)
```

References
==========
* Zhang, X., & Shu, C.-W. (2010). "On maximum-principle-satisfying high order schemes for scalar
  conservation laws." Journal of Computational Physics, 229(9), 3091-3120.
"""
function WENO(FT::DataType=Oceananigans.defaults.FloatType;
              weight_computation::DataType=Nothing,
              order = 5,
              buffer_scheme = DecreasingOrderAdvectionScheme(),
              time_discretization = ExplicitTimeDiscretization(),
              bounds = nothing,
              minimum_buffer_upwind_order = 3)

    mod(order, 2) == 0 && throw(ArgumentError("WENO reconstruction scheme is defined only for odd orders"))

    if bounds isa NTuple{2}
        bounds = BoundsPreservation(bounds...)
    end

    if order < 3
        # WENO(order=1) is equivalent to UpwindBiased(order=1)
        return UpwindBiased(FT; order=1)
    else
        advecting_velocity_scheme = Centered(FT; order=2)

        if buffer_scheme isa DecreasingOrderAdvectionScheme
            if order ≤ minimum_buffer_upwind_order
                # At minimum order, switch to Centered scheme
                buffer_scheme = Centered(FT; order=2)
            else
                buffer_scheme = WENO(FT; order=order-2, bounds, minimum_buffer_upwind_order, weight_computation)
            end
        end

        N = Int((order + 1) ÷ 2)
        preserved_bounds = with_order_and_float_type(order, FT, bounds)

        return WENO{N, FT, weight_computation}(preserved_bounds, buffer_scheme, advecting_velocity_scheme, time_discretization)
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
