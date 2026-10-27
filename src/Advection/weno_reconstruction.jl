#####
##### Weighted Essentially Non-Oscillatory (WENO) advection scheme
#####

struct WENO{N, FT, WCT, TD, PP, SI, SC} <: AbstractUpwindBiasedAdvectionScheme{N, FT, TD}
    bounds :: PP
    advecting_velocity_scheme :: SI
    time_discretization :: TD
    coefficients :: SC
    function WENO{N, FT, WCT}(bounds::PP, advecting_velocity_scheme::SI, time_discretization::TD,
                              coefficients::SC = nothing) where {N, FT, WCT, PP, SI, TD, SC}
        return new{N, FT, WCT, TD, PP, SI, SC}(bounds, advecting_velocity_scheme, time_discretization, coefficients)
    end
end

"""
    WENO([FT=Oceananigans.defaults.FloatType;]
         weight_computation=Nothing,
         order = 5,
         time_discretization = ExplicitTimeDiscretization(),
         bounds = nothing,
         maximum_courant_number = 5//18,
         stretched = false)

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
- `stretched`: The directions, among `:x`, `:y` and `:z`, along which the reconstruction coefficients, the smoothness
               indicators and the optimal weights are computed from the cell widths of the grid, built when the grid is
               known (see `materialize_advection`): a tuple such as `(:z,)` or `(:x, :z)`, `true` for all three or
               `false` for none. Uniform and unselected directions keep the constant coefficients. Default: `false`.

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
              maximum_courant_number = 5//18,
              stretched = false)

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

        directions = stretched === true ? (:x, :y, :z) : stretched === false ? () : stretched
        all(∈((:x, :y, :z)), directions) || throw(ArgumentError("stretched directions must be among :x, :y and :z, found $directions"))
        coefficients = isempty(directions) ? nothing : directions

        return WENO{N, FT, weight_computation}(preserved_bounds, Centered(FT; order=2), time_discretization, coefficients)
    end
end

weno_order(::WENO{N}) where N = 2N-1
Base.eltype(::WENO{N, FT}) where {N, FT} = FT
Base.summary(a::WENO{N, FT, WCT, TD, Nothing}) where {N, FT, WCT, TD} =
    string("WENO{$N, $FT, $WCT}(order=", 2N-1, isnothing(a.coefficients) ? "" : ", stretched", ")")
Base.summary(a::WENO{N, FT, WCT, TD, PP}) where {N, FT, WCT, TD, PP} =
    string("WENO{$N, $FT, $WCT}(order=", 2N-1, ", bounds=", string(a.bounds), isnothing(a.coefficients) ? "" : ", stretched", ")")

Base.show(io::IO, a::WENO) =
    print(io, summary(a), '\n', "└── advecting_velocity_scheme: ", summary(a.advecting_velocity_scheme))

Adapt.adapt_structure(to, scheme::WENO{N, FT, WCT}) where {N, FT, WCT} =
     WENO{N, FT, WCT}(Adapt.adapt(to, scheme.bounds),
                      Adapt.adapt(to, scheme.advecting_velocity_scheme),
                      Adapt.adapt(to, scheme.time_discretization),
                      Adapt.adapt(to, scheme.coefficients))

Architectures.on_architecture(to, scheme::WENO{N, FT, WCT}) where {N, FT, WCT} =
    WENO{N, FT, WCT}(on_architecture(to, scheme.bounds),
                     on_architecture(to, scheme.advecting_velocity_scheme),
                     on_architecture(to, scheme.time_discretization),
                     on_architecture(to, scheme.coefficients))

# Select the default WENO weight computation
# Specific backends may override
default_weno_weight_computation(arch) = Oceananigans.Utils.BackendOptimizedDivision
