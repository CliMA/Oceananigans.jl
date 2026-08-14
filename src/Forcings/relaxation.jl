using Oceananigans.Grids: node, xnodes, ynodes, znodes, ξnodes, ηnodes, Face, Center
using Oceananigans.OutputReaders: interpolate
using Oceananigans: instantiated_location
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES

@inline zerofunction(args...) = 0
@inline onefunction(args...) = 1

T_zerofunction = typeof(zerofunction)
T_onefunction = typeof(onefunction)

Base.summary(::T_zerofunction) = "0"
Base.summary(::T_onefunction) = "1"

"""
    struct Relaxation{R, M, T}

Callable object for restoring fields to a `target` at
some `rate` and within a `mask`ed region in `x, y, z`.
"""
struct Relaxation{R, M, T}
      rate :: R
      mask :: M
    target :: T
end

"""
    Relaxation(; rate, mask=onefunction, target=zerofunction)

Returns a `Forcing` that restores a field to `target(X..., t)`
at the specified `rate`, in the region `mask(X...)`.

The functions `onefunction` and `zerofunction` always return 1 and 0, respectively.
Thus the default `mask` leaves the whole domain uncovered, and the default `target` is zero.

Example
=======

* Restore a field to zero on a timescale of "3600" (equal
  to one hour if the time units of the simulation are seconds).

```jldoctest relaxation
using Oceananigans

damping = Relaxation(rate = 1/3600)

# output
Relaxation{Float64, typeof(Oceananigans.Forcings.onefunction), typeof(Oceananigans.Forcings.zerofunction)}
├── rate: 0.0002777777777777778
├── mask: 1
└── target: 0
```

* Restore a field to a linear z-gradient within the bottom 1/4 of a domain
  on a timescale of "60" (equal to one minute if the time units of the simulation
  are seconds).

```jldoctest relaxation
dTdz = 0.001 # ⁰C m⁻¹, temperature gradient

T₀ = 20 # ⁰C, surface temperature at z=0

Lz = 100 # m, depth of domain

bottom_sponge_layer = Relaxation(; rate = 1/60,
                                   target = LinearTarget{:z}(intercept=T₀, gradient=dTdz),
                                   mask = GaussianMask{:z}(center=-Lz, width=Lz/4))

# output
Relaxation{Float64, GaussianMask{:z, Float64}, LinearTarget{:z, Float64}}
├── rate: 0.016666666666666666
├── mask: exp(-(z + 100.0)^2 / (2 * 25.0^2))
└── target: 20.0 + 0.001 * z
```
"""
Relaxation(; rate, mask=onefunction, target=zerofunction) = Relaxation(rate, mask, target)

""" Wrap `forcing::Relaxation` in `ContinuousForcing` and add the appropriate field dependency. """
function materialize_forcing(forcing::Relaxation, field, field_name, model_field_names)
    continuous_relaxation = ContinuousForcing(forcing, field_dependencies=field_name)
    return materialize_forcing(continuous_relaxation, field, field_name, model_field_names)
end

"""
$(TYPEDEF)

Materialized `Relaxation` target that carries a `FieldTimeSeries` source, the
simulation-side location at which the relaxation is evaluated, the FTS's grid
(cached separately because `GPUAdaptedFieldTimeSeries` does not carry a grid
field), and the integer index of the forced field in `model_fields`. The
index is encoded as a type parameter `I::Int` (not a struct field) so that
`model_fields[I]` is a compile-time access — required for GPU kernel
compilation, since `model_fields` is a heterogeneous `NamedTuple` and a
runtime-integer index would force a dynamic getfield call that PTX cannot
lower. Constructed by `materialize_forcing` when a `Relaxation`'s `target`
is a `FieldTimeSeries`; not intended for direct user construction.
"""
struct FieldTimeSeriesTarget{L, F, G, I}
    location          :: L    # simulation-side instantiated location tuple
    field_time_series :: F
    fts_grid          :: G    # explicit copy of `field_time_series.grid` at materialize time
end

FieldTimeSeriesTarget(location, field_time_series, fts_grid, index::Int) =
    FieldTimeSeriesTarget{typeof(location), typeof(field_time_series), typeof(fts_grid), index}(location, field_time_series, fts_grid)

# Convenience: pull the grid off the FTS at construction. Used by
# `materialize_forcing` on the host side.
FieldTimeSeriesTarget(location, field_time_series, index::Int) =
    FieldTimeSeriesTarget(location, field_time_series, field_time_series.grid, index)

# Extract the field index from the type parameter. Used by the kernel callable
# so that `model_fields[_field_index(target)]` resolves to a compile-time index
# and the surrounding load can be inlined on GPU.
@inline _field_index(::FieldTimeSeriesTarget{<:Any, <:Any, <:Any, I}) where I = I

# Adapt every component individually so the resulting struct is isbits.
# In particular `field_time_series.grid` cannot be recovered from the
# adapted FTS (because `GPUAdaptedFieldTimeSeries` has no grid field), so
# the grid lives as its own field on `FieldTimeSeriesTarget` and is adapted
# in place. The index is reinjected from the type parameter so the resulting
# struct preserves the static-index property.
Adapt.adapt_structure(to, target::FieldTimeSeriesTarget{<:Any, <:Any, <:Any, I}) where I =
    FieldTimeSeriesTarget(Adapt.adapt(to, target.location),
                          Adapt.adapt(to, target.field_time_series),
                          Adapt.adapt(to, target.fts_grid),
                          I)

# Recursive Adapt for `Relaxation` so that an inner `FieldTimeSeriesTarget`
# is adapted on the path to the kernel. Without this, Relaxation's default
# Adapt fallback returns the host-side struct unchanged, and the FTS-target
# adapt above is never reached.
Adapt.adapt_structure(to, r::Relaxation) =
    Relaxation(Adapt.adapt(to, r.rate),
               Adapt.adapt(to, r.mask),
               Adapt.adapt(to, r.target))

const FieldTimeSeriesRelaxation{R, M, T<:FieldTimeSeriesTarget} = Relaxation{R, M, T}

@inline function (f::FieldTimeSeriesRelaxation)(i, j, k, grid, clock, model_fields)
    target = f.target
    fts = target.field_time_series
    X = node(i, j, k, grid, target.location...)
    @inbounds ϕ = model_fields[_field_index(target)][i, j, k]
    ϕᵣ = interpolate(X, Time(clock.time), fts, instantiated_location(fts), target.fts_grid)
    return f.rate * f.mask(X...) * (ϕᵣ - ϕ)
end

"""
$(TYPEDSIGNATURES)

Wrap a `Relaxation` with `FieldTimeSeries` target into a materialized form
carrying simulation-side location and an integer field index, so the kernel can
spatially+temporally interpolate `target` and read `ϕ` from `model_fields`.
"""
function materialize_forcing(forcing::Relaxation{R, M, <:FlavorOfFTS}, field,
                             field_name, model_field_names) where {R, M}
    validate_fts_target_extent(forcing.target, field)
    index = findfirst(==(field_name), model_field_names)
    target = FieldTimeSeriesTarget(instantiated_location(field), forcing.target, index)
    return Relaxation(forcing.rate, forcing.mask, target)
end

function validate_fts_target_extent(fts, field)
    fts_grid = fts.grid
    sim_grid = field.grid
    fts_loc = instantiated_location(fts)
    sim_loc = instantiated_location(field)

    # Check that every model sampling position (at the forced field's location)
    # lies within the FTS coverage at the FTS's own storage location. The
    # kernel queries `interpolate(X, ..., fts, fts_loc, fts_grid)` with
    # `X = node(i, j, k, sim_grid, sim_loc...)`; if X falls outside the FTS
    # node range, trilinear interpolation reads from FTS halos (which
    # `set!(fts[n], …)` does not fill), producing silently wrong values near
    # the boundary.
    for (label, nodes_fn) in (("x", xnodes), ("y", ynodes), ("z", znodes))
        sim_lo, sim_hi = extrema(nodes_fn(sim_grid, sim_loc...))
        fts_lo, fts_hi = extrema(nodes_fn(fts_grid, fts_loc...))
        (fts_lo ≤ sim_lo && sim_hi ≤ fts_hi) ||
            throw(ArgumentError(
                "FieldTimeSeries target $label-extent [$fts_lo, $fts_hi] does not " *
                "bracket model grid $label-extent [$sim_lo, $sim_hi]"))
    end
    return nothing
end

Base.summary(target::FieldTimeSeriesTarget) =
    "FieldTimeSeriesTarget(location=$(target.location), index=$(_field_index(target)))"

@inline (f::Relaxation)(x, y, z, t, field) =
    f.rate * f.mask(x, y, z) * (f.target(x, y, z, t) - field)

@inline (f::Relaxation{R, M, <:Number})(x, y, z, t, field) where {R, M} =
    f.rate * f.mask(x, y, z) * (f.target - field)

# Methods for grids with Flat dimensions:
# Here, the meaning of the coordinate xₙ depends on which dimension is Flat:
# for example, in the below method (x₁, x₂) may be (ξ, η), (ξ, r), or (η, r), where
# ξ, η, and r are the first, second, and third coordinates respectively.
@inline (f::Relaxation)(x₁, x₂, t, field) =
    f.rate * f.mask(x₁, x₂) * (f.target(x₁, x₂, t) - field)

@inline (f::Relaxation{R, M, <:Number})(x₁, x₂, t, field) where {R, M} =
    f.rate * f.mask(x₁, x₂) * (f.target - field)

# Below, the coordinate x₁ can be ξ, η, or r (see above)
@inline (f::Relaxation)(x₁, t, field) =
    f.rate * f.mask(x₁) * (f.target(x₁, t) - field)

@inline (f::Relaxation{R, M, <:Number})(x₁, t, field) where {R, M} =
    f.rate * f.mask(x₁) * (f.target - field)

"""Show the innards of a `Relaxation` in the REPL."""
Base.show(io::IO, relaxation::Relaxation{R, M, T}) where {R, M, T} =
    print(io, "Relaxation{$R, $M, $T}", "\n",
        "├── rate: $(relaxation.rate)", "\n",
        "├── mask: $(summary(relaxation.mask))", "\n",
        "└── target: $(summary(relaxation.target))")

Base.summary(relaxation::Relaxation) =
    "Relaxation(rate=$(relaxation.rate), mask=$(summary(relaxation.mask)), target=$(summary(relaxation.target)))"

#####
##### Flow-dependent rate
#####

"""
$(TYPEDSIGNATURES)

Rate for `Relaxation` that nudges toward `target` at `rate_in` where the local horizontal
normal velocity indicates inflow through a lateral boundary, and at `rate_out` where it
indicates outflow, following the ROMS/Marchesiello-et-al.-2001 nudging-layer convention.

Each of the four domain edges (`west_edge`, `east_edge`, `south_edge`, `north_edge`) gets
a Gaussian rim of the given `width` (same units as the grid's horizontal coordinates); the
four rims are combined with `max`, not summed, so that overlapping corners are not
double-relaxed. Whether a point is on the inflow or outflow side of a given edge is
determined by the true horizontal normal velocity there (`u` at west/east, `v` at
south/north) — so `u`, `v`, and every tracer relaxed near a given edge share the same
in/out gating.

Use `FlowDependentRate(grid; width, rate_in, rate_out)` to build one directly from a
grid's horizontal extent.
"""
struct FlowDependentRate{FT}
     west_edge :: FT
     east_edge :: FT
    south_edge :: FT
    north_edge :: FT
         width :: FT
       rate_in :: FT
      rate_out :: FT
end

"""
$(TYPEDSIGNATURES)

Build a [`FlowDependentRate`](@ref) spanning the horizontal extent of `grid`, with lateral
nudging layers of `width` (same units as the grid's horizontal coordinates) at each of the
four edges, restoring at `rate_in` on inflow and `rate_out` on outflow.

Example
=======

```jldoctest flowdependentrate
using Oceananigans

grid = LatitudeLongitudeGrid(size=(10, 10, 1), longitude=(-5, 20), latitude=(-57, -50), z=(-100, 0))

rate = FlowDependentRate(grid; width=0.1, rate_in=1/15minutes, rate_out=1/12hours)

# output
FlowDependentRate{Float64}
├──  west_edge: -3.75
├──  east_edge: 18.75
├── south_edge: -56.65
├── north_edge: -50.35
├──      width: 0.1
├──    rate_in: 0.0011111111111111111
└──   rate_out: 2.3148148148148147e-5
```
"""
function FlowDependentRate(grid; width, rate_in, rate_out)
    west_edge,  east_edge  = extrema(ξnodes(grid, Center(), Center(), Center()))
    south_edge, north_edge = extrema(ηnodes(grid, Center(), Center(), Center()))
    FT = promote_type(typeof(width), typeof(rate_in), typeof(rate_out))
    return FlowDependentRate(convert(FT, west_edge),  convert(FT, east_edge),
                              convert(FT, south_edge), convert(FT, north_edge),
                              convert(FT, width), convert(FT, rate_in), convert(FT, rate_out))
end

@inline rim(ξ, edge, width) = exp(-(ξ - edge)^2 / (2 * width^2))

# u, v interpolated to grid location `loc`, dispatched so u-points, v-points, and
# cell centers (tracers) each get the correctly-located stencil.
@inline function normal_velocities(i, j, k, model_fields, ::Tuple{Face, Center, Center})
    u = @inbounds model_fields.u[i, j, k]
    v = @inbounds 0.25 * (model_fields.v[i-1, j,   k] + model_fields.v[i, j,   k] +
                           model_fields.v[i-1, j+1, k] + model_fields.v[i, j+1, k])
    return u, v
end

@inline function normal_velocities(i, j, k, model_fields, ::Tuple{Center, Face, Center})
    u = @inbounds 0.25 * (model_fields.u[i,   j-1, k] + model_fields.u[i,   j, k] +
                           model_fields.u[i+1, j-1, k] + model_fields.u[i+1, j, k])
    v = @inbounds model_fields.v[i, j, k]
    return u, v
end

@inline function normal_velocities(i, j, k, model_fields, ::Tuple{Center, Center, Center})
    u = @inbounds 0.5 * (model_fields.u[i, j, k] + model_fields.u[i+1, j, k])
    v = @inbounds 0.5 * (model_fields.v[i, j, k] + model_fields.v[i, j+1, k])
    return u, v
end

@inline function evaluate_rate(r::FlowDependentRate, i, j, k, X, model_fields, loc)
    λ, φ = X[1], X[2]
    u_n, v_n = normal_velocities(i, j, k, model_fields, loc)

    west  = rim(λ, r.west_edge,  r.width) * ifelse(u_n > 0, r.rate_in, r.rate_out)
    east  = rim(λ, r.east_edge,  r.width) * ifelse(u_n < 0, r.rate_in, r.rate_out)
    south = rim(φ, r.south_edge, r.width) * ifelse(v_n > 0, r.rate_in, r.rate_out)
    north = rim(φ, r.north_edge, r.width) * ifelse(v_n < 0, r.rate_in, r.rate_out)

    # max, not sum: two rims can overlap at a corner and shouldn't double the rate
    return max(west, east, south, north)
end

@inline evaluate_target(target::Number, X, t) = target
@inline evaluate_target(target,         X, t) = target(X..., t)

"""
$(TYPEDEF)

Materialized target carrying the simulation-side location at which a `Relaxation` with a
[`FlowDependentRate`](@ref) is evaluated, together with the integer index of the forced
field in `model_fields`. Mirrors [`FieldTimeSeriesTarget`](@ref)'s use of a type parameter
`I::Int` for the index, so `model_fields[I]` is a compile-time access on GPU. Wraps the
user-supplied `target` (a `Number` or callable) unchanged; constructed by
`materialize_forcing` and not intended for direct user construction.

$(TYPEDFIELDS)
"""
struct MaterializedRelaxationTarget{L, T, I}
    "simulation-side instantiated location tuple"
    location :: L
    "user-supplied target, wrapped unchanged"
    target   :: T
end

MaterializedRelaxationTarget(location, target, index::Int) =
    MaterializedRelaxationTarget{typeof(location), typeof(target), index}(location, target)

@inline _field_index(::MaterializedRelaxationTarget{<:Any, <:Any, I}) where I = I

Adapt.adapt_structure(to, t::MaterializedRelaxationTarget{<:Any, <:Any, I}) where I =
    MaterializedRelaxationTarget(Adapt.adapt(to, t.location), Adapt.adapt(to, t.target), I)

Base.summary(t::MaterializedRelaxationTarget) =
    "MaterializedRelaxationTarget(location=$(t.location), index=$(_field_index(t)))"

const FlowDependentRelaxation{M, T<:MaterializedRelaxationTarget} = Relaxation{<:FlowDependentRate, M, T}

@inline function (f::FlowDependentRelaxation)(i, j, k, grid, clock, model_fields)
    mt = f.target
    X = node(i, j, k, grid, mt.location...)
    @inbounds ϕ = model_fields[_field_index(mt)][i, j, k]
    ϕᵣ = evaluate_target(mt.target, X, clock.time)
    rate = evaluate_rate(f.rate, i, j, k, X, model_fields, mt.location)
    return rate * f.mask(X...) * (ϕᵣ - ϕ)
end

"""
$(TYPEDSIGNATURES)

Wrap a `Relaxation` with a [`FlowDependentRate`](@ref) into a materialized form carrying
simulation-side location and an integer field index, so the kernel can evaluate the
inflow/outflow-dependent rate and read `ϕ` from `model_fields` directly (bypassing
`ContinuousForcing`, which has no hook for a rate that needs `model_fields`).
"""
function materialize_forcing(forcing::Relaxation{<:FlowDependentRate}, field, field_name, model_field_names)
    index = findfirst(==(field_name), model_field_names)
    target = MaterializedRelaxationTarget(instantiated_location(field), forcing.target, index)
    return Relaxation(forcing.rate, forcing.mask, target)
end

function Base.show(io::IO, rate::FlowDependentRate)
    FT = typeof(rate.west_edge)
    print(io, "FlowDependentRate{$FT}")
    rows = ["west_edge" => string(rate.west_edge), "east_edge" => string(rate.east_edge),
            "south_edge" => string(rate.south_edge), "north_edge" => string(rate.north_edge),
            "width" => string(rate.width), "rate_in" => string(rate.rate_in), "rate_out" => string(rate.rate_out)]
    width = maximum(length(first(r)) for r in rows)
    for (i, (key, value)) in enumerate(rows)
        prefix = i == length(rows) ? "└── " : "├── "
        print(io, "\n", prefix, lpad(key, width), ": ", value)
    end
end

Base.summary(rate::FlowDependentRate) =
    "FlowDependentRate(rate_in=$(rate.rate_in), rate_out=$(rate.rate_out), width=$(rate.width))"

#####
##### Sponge layer functions
#####

"""
    GaussianMask{D}(center, width)

Callable object that returns a Gaussian masking function centered on
`center`, with `width`, and varying along direction `D`, i.e.,

```
exp(-(D - center)^2 / (2 * width^2))
```

Example
=======

Create a Gaussian mask centered on `z=0` with width `1` meter.

```jldoctest
julia> using Oceananigans

julia> mask = GaussianMask{:z}(center=0, width=1)
GaussianMask{:z, Int64}(0, 1)
```
"""
struct GaussianMask{D, T}
    center :: T
     width :: T

    function GaussianMask{D}(; center, width) where D
        T = promote_type(typeof(center), typeof(width))
        return new{D, T}(center, width)
    end
end

@inline (g::GaussianMask{:x})(x, y, z) = exp(-(x - g.center)^2 / (2 * g.width^2))
@inline (g::GaussianMask{:y})(x, y, z) = exp(-(y - g.center)^2 / (2 * g.width^2))
@inline (g::GaussianMask{:z})(x, y, z) = exp(-(z - g.center)^2 / (2 * g.width^2))

show_exp_arg(D, c) = c == 0 ? "$D^2" :
                     c > 0  ? "($D - $c)^2" :
                              "($D + $(-c))^2"

Base.summary(g::GaussianMask{D}) where D =
    "exp(-$(show_exp_arg(D, g.center)) / (2 * $(g.width)^2))"


"""
    PiecewiseLinearMask{D}(center, width)

Callable object that returns a piecewise linear masking function centered on
`center`, with `width`, and varying along direction `D`. The mask is:
- 0 when |D - center| > width
- 1 when D = center
- Linear interpolation between 0 and 1 when |D - center| ≤ width

Example
=======

Create a piecewise linear mask centered on `z=0` with width `1` meter.

```jldoctest
julia> using Oceananigans

julia> mask = PiecewiseLinearMask{:z}(center=0, width=1)
PiecewiseLinearMask{:z, Int64}(0, 1)

julia> mask(0, 0, 0) == 1
true

julia> mask(0, 0, 1) == mask(0, 0, -1) == 0
true
```
"""
struct PiecewiseLinearMask{D, T}
    center :: T
     width :: T

    function PiecewiseLinearMask{D}(; center, width) where D
        T = promote_type(typeof(center), typeof(width))
        return new{D, T}(center, width)
    end
end

@inline function (p::PiecewiseLinearMask{:x})(x, y, z)
    d = 1 - abs(x - p.center) / p.width
    return max(0, d)
end

@inline function (p::PiecewiseLinearMask{:y})(x, y, z)
    d = 1 - abs(y - p.center) / p.width
    return max(0, d)
end

@inline function (p::PiecewiseLinearMask{:z})(x, y, z)
    d = 1 - abs(z - p.center) / p.width
    return max(0, d)
end

Base.summary(p::PiecewiseLinearMask{D}) where D =
    "piecewise_linear($D, center=$(p.center), width=$(p.width))"


"""
    CosineRampMask{D}(start, stop)

Callable object that returns a half-cosine ramp masking function varying
between `0` at coordinate `start` and `1` at coordinate `stop`, along
direction `D`. Outside the interval the mask is clamped to its endpoint
values. Inside the interval the mask is

```
(1 - cos(π * (D - start) / (stop - start))) / 2
```

The sign of `stop - start` flips the ramp direction, so the same struct
covers upward (`start < stop`) and downward (`start > stop`) ramps —
useful for upper/lower sponge layers and Davies-style lateral nudging
zones.

Example
=======

Create a z-ramp that smoothly transitions from 0 at `z = 1500` to 1 at
`z = 2500`.

```jldoctest
julia> using Oceananigans

julia> mask = CosineRampMask{:z}(start=1500, stop=2500)
CosineRampMask{:z, Int64}(1500, 2500)
```
"""
struct CosineRampMask{D, T}
    start :: T
     stop :: T

    function CosineRampMask{D}(; start, stop) where D
        start == stop && throw(ArgumentError("CosineRampMask{$D}: start ≠ stop required"))
        T = promote_type(typeof(start), typeof(stop))
        return new{D, T}(start, stop)
    end
end

@inline function cosine_ramp(m::CosineRampMask, ξ)
    r = clamp((ξ - m.start) / (m.stop - m.start), 0, 1)
    return (1 - cos(π * r)) / 2
end

@inline (m::CosineRampMask{:x})(x, y, z) = cosine_ramp(m, x)
@inline (m::CosineRampMask{:y})(x, y, z) = cosine_ramp(m, y)
@inline (m::CosineRampMask{:z})(x, y, z) = cosine_ramp(m, z)

Base.summary(m::CosineRampMask{D}) where D =
    "cosine_ramp($D, start=$(m.start), stop=$(m.stop))"


#####
##### Linear target functions
#####

"""
    LinearTarget{D}(intercept, gradient)

Callable object that returns a Linear target function
with `intercept` and `gradient`, and varying along direction `D`, i.e.,

```
intercept + D * gradient
```

Example
=======

Create a linear target function varying in `z`, equal to `0` at
`z=0` and with gradient 10⁻⁶:

```julia
julia> target = LinearTarget{:z}(intercept=0, gradient=1e-6)
```
"""
struct LinearTarget{D, T}
    intercept :: T
     gradient :: T

    function LinearTarget{D}(; intercept, gradient) where D
        T = promote_type(typeof(gradient), typeof(intercept))
        return new{D, T}(intercept, gradient)
    end
end

@inline (p::LinearTarget{:x})(x, y, z, t) = p.intercept + p.gradient * x
@inline (p::LinearTarget{:y})(x, y, z, t) = p.intercept + p.gradient * y
@inline (p::LinearTarget{:z})(x, y, z, t) = p.intercept + p.gradient * z

Base.summary(l::LinearTarget{:x}) = "$(l.intercept) + $(l.gradient) * x"
Base.summary(l::LinearTarget{:y}) = "$(l.intercept) + $(l.gradient) * y"
Base.summary(l::LinearTarget{:z}) = "$(l.intercept) + $(l.gradient) * z"
