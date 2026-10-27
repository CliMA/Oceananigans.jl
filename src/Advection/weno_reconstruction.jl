#####
##### Weighted Essentially Non-Oscillatory (WENO) advection scheme
#####

struct WENO{N, FT, WCT, TD, PP, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT, TD}
    bounds :: PP
    advecting_velocity_scheme :: SI
    time_discretization :: TD
    function WENO{N, FT, WCT}(bounds::PP, advecting_velocity_scheme::SI, time_discretization::TD) where {N, FT, WCT, PP, SI, TD}
        return new{N, FT, WCT, TD, PP, SI}(bounds, advecting_velocity_scheme, time_discretization)
    end
end

"""
    WENO([FT=Oceananigans.defaults.FloatType;]
         weight_computation=Nothing,
         order = 5,
         time_discretization = ExplicitTimeDiscretization(),
         bounds = nothing,
         maximum_courant_number = 5//18)

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
- `bounds` (experimental): A tuple `(cᵐⁱⁿ, cᵐᵃˣ)` switching on the maximum-principle-satisfying limiter of
                           Zhang and Shu (2010), which rescales the reconstruction of every cell towards the cell
                           mean so that the advective update stays within `bounds`. One rescaling factor is
                           shared by the three directions, so the bound is three-dimensional. It is guaranteed
                           for a forward Euler or strong-stability-preserving update whose Courant numbers
                           satisfy `λˣ + λʸ + λᶻ ≤ maximum_courant_number`.
                           Default: `nothing`, which does not use a bounds-preserving scheme.
- `maximum_courant_number`: The Courant number `ω̂₁ ∈ (0, 1/2)` up to which `bounds` is guaranteed. Raising it
                            admits a larger time step and limits more aggressively. Default: `5//18`.

Examples
========

To build the default 5th-order scheme:

```jldoctest weno
julia> using Oceananigans

julia> WENO()
WENO{3, Float64, Nothing}(order=5)
└── advecting_velocity_scheme: Centered(order=2)
```

To build a 9th-order scheme (often a good choice for a stable
yet minimally-dissipative advection scheme):

```jldoctest weno
julia> WENO(order=9)
WENO{5, Float64, Nothing}(order=9)
└── advecting_velocity_scheme: Centered(order=2)
```

```jldoctest weno
julia> WENO(order=9, bounds=(0, 1))
WENO{5, Float64, Nothing}(order=9, bounds=(0.0, 1.0))
└── advecting_velocity_scheme: Centered(order=2)
```

To build a WENO scheme that uses approximate division on a GPU to execute faster:
```jldoctest weno
julia> WENO(;weight_computation=Oceananigans.Utils.BackendOptimizedDivision)
WENO{3, Float64, Oceananigans.Utils.BackendOptimizedDivision}(order=5)
└── advecting_velocity_scheme: Centered(order=2)
```
"""
function WENO(FT::DataType=Oceananigans.defaults.FloatType;
              weight_computation::DataType=Nothing,
              order = 5,
              time_discretization = ExplicitTimeDiscretization(),
              bounds = nothing,
              maximum_courant_number = 5//18)

    mod(order, 2) == 0 && throw(ArgumentError("WENO reconstruction scheme is defined only for odd orders"))

    isnothing(bounds) || bounds isa NTuple{2} || throw(ArgumentError("bounds must be nothing or a tuple of two values"))
    0 < maximum_courant_number < 1//2 || throw(ArgumentError("maximum_courant_number must lie in (0, 1/2), found $maximum_courant_number"))

    if order < 3
        # WENO(order=1) is equivalent to UpwindBiased(order=1)
        return UpwindBiased(FT; order=1)
    else
        N = Int((order + 1) ÷ 2)
        preserved_bounds = isnothing(bounds) ? nothing :
            BoundsPreservation(convert(FT, bounds[1]), convert(FT, bounds[2]), convert(FT, maximum_courant_number), nothing)

        return WENO{N, FT, weight_computation}(preserved_bounds, Centered(FT; order=2), time_discretization)
    end
end

weno_order(::WENO{N}) where N = 2N-1
Base.eltype(::WENO{N, FT}) where {N, FT} = FT
Base.summary(a::WENO{N, FT, WCT, TD, Nothing}) where {N, FT, WCT, TD} = string("WENO{$N, $FT, $WCT}(order=", 2N-1, ")")
Base.summary(a::WENO{N, FT, WCT, TD, PP}) where {N, FT, WCT, TD, PP} = string("WENO{$N, $FT, $WCT}(order=", 2N-1, ", bounds=", string(a.bounds), ")")

Base.show(io::IO, a::WENO) =
    print(io, summary(a), '\n', "└── advecting_velocity_scheme: ", summary(a.advecting_velocity_scheme))

Adapt.adapt_structure(to, scheme::WENO{N, FT, WCT}) where {N, FT, WCT} =
     WENO{N, FT, WCT}(Adapt.adapt(to, scheme.bounds),
                      Adapt.adapt(to, scheme.advecting_velocity_scheme),
                      Adapt.adapt(to, scheme.time_discretization))

Architectures.on_architecture(to, scheme::WENO{N, FT, WCT}) where {N, FT, WCT} =
    WENO{N, FT, WCT}(on_architecture(to, scheme.bounds),
                     on_architecture(to, scheme.advecting_velocity_scheme),
                     on_architecture(to, scheme.time_discretization))

# Select the default WENO weight computation
# Specific backends may override
default_weno_weight_computation(arch) = Oceananigans.Utils.BackendOptimizedDivision
