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
         bounds = nothing,
         maximum_cfl = nothing,
         flux_limiter = nothing,
         minimum_buffer_upwind_order = 3,
         maximum_courant_number = nothing)

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
- `bounds` (experimental): A [`BoundsPreservation`](@ref), or a tuple `(cᵐⁱⁿ, cᵐᵃˣ)` as shorthand for one with
                           the options below, switching on the maximum-principle-satisfying limiter of
                           [Zhang and Shu (2010)](@cite ZhangShu2010), which rescales the reconstruction of
                           every cell towards the cell mean so that the advective update stays within `bounds`.
                           One rescaling factor is shared by the three directions, so the bound is
                           three-dimensional. It is guaranteed
                           for a forward Euler or strong-stability-preserving update whose Courant numbers
                           satisfy `λˣ + λʸ + λᶻ ≤ maximum_cfl`.
                           Default: `nothing`, which does not use a bounds-preserving scheme.
- `maximum_cfl`: With a tuple `bounds`, the Courant number `ω̂₁ ∈ (0, 1/2)` up to which `bounds` is guaranteed.
                 Raising it admits a larger time step and limits more aggressively.
                 Default: `nothing`, for the `BoundsPreservation` default `5//18`.
- `flux_limiter` (experimental): With a tuple `bounds`, whether to also limit the fluxes between cells to what
                                 the coming update can afford, so that the tracer stays within `bounds` for any
                                 time step; `maximum_cfl` then only sets how hard the rescaling clips. `true`
                                 for the default [`FluxLimiter`](@ref), or a `FluxLimiter` such as
                                 `FluxLimiter(credit=ColumnSweep())`, which a vertical Courant number above
                                 one needs. Default: `nothing`, for none.
- `minimum_buffer_upwind_order`: The minimum upwind order for buffer schemes. When the buffer
                                 scheme order reaches this value, subsequent buffers use
                                 `Centered(order=2)` instead of continuing to decrease the
                                 upwind order. Default: 3.
- `maximum_courant_number`: Deprecated name of `maximum_cfl`.

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
julia> WENO(order=5, bounds=(0, 1), flux_limiter=true)
WENO{3, Float64, Nothing}(order=5, bounds=(0.0, 1.0), flux_limiter=FluxLimiter())
├── buffer_scheme: WENO{2, Float64, Nothing}(order=3, bounds=(0.0, 1.0), flux_limiter=FluxLimiter())
│   └── buffer_scheme: Centered(order=2)
└── advecting_velocity_scheme: Centered(order=2)
```

The same, grouping the interval with its options in a `BoundsPreservation`:

```jldoctest weno
julia> WENO(order=5, bounds=BoundsPreservation(0, 1; maximum_cfl=1//4, flux_limiter=FluxLimiter(credit=ColumnSweep())))
WENO{3, Float64, Nothing}(order=5, bounds=(0.0, 1.0), flux_limiter=FluxLimiter(credit=ColumnSweep()))
├── buffer_scheme: WENO{2, Float64, Nothing}(order=3, bounds=(0.0, 1.0), flux_limiter=FluxLimiter(credit=ColumnSweep()))
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
              maximum_cfl = nothing,
              flux_limiter = nothing,
              minimum_buffer_upwind_order = 3,
              maximum_courant_number = nothing)

    mod(order, 2) == 0 && throw(ArgumentError("WENO reconstruction scheme is defined only for odd orders"))

    if !isnothing(maximum_courant_number)
        @warn "The WENO keyword argument `maximum_courant_number` is deprecated, use `maximum_cfl`" maxlog=1
        maximum_cfl = something(maximum_cfl, maximum_courant_number)
    end

    # A tuple is shorthand for a `BoundsPreservation` with the flat keyword arguments as options
    if bounds isa NTuple{2}
        maximum_cfl = something(maximum_cfl, 5//18)
        bounds = BoundsPreservation(bounds...; maximum_cfl, flux_limiter)
    elseif bounds isa BoundsPreservation
        isnothing(maximum_cfl) && isnothing(flux_limiter) ||
            throw(ArgumentError("maximum_cfl and flux_limiter are options of the BoundsPreservation passed as bounds"))
    elseif !isnothing(bounds)
        throw(ArgumentError("bounds must be nothing, a tuple of two values or a BoundsPreservation"))
    elseif !(isnothing(flux_limiter) || flux_limiter === false)
        throw(ArgumentError("a flux_limiter requires bounds"))
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
        preserved_bounds = with_float_type(FT, bounds)

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
