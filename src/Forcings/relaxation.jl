using Oceananigans: instantiated_location
using Oceananigans.Grids: node, xnodes, ynodes
using Oceananigans.Fields: AbstractField, Field, compute!, show_location
using Oceananigans.AbstractOperations: Average
using Oceananigans.OutputReaders: interpolate
using Oceananigans.Utils: prettysummary

@inline zerofunction(args...) = 0
@inline onefunction(args...) = 1

T_zerofunction = typeof(zerofunction)
T_onefunction = typeof(onefunction)

Base.summary(::T_zerofunction) = "0"
Base.summary(::T_onefunction) = "1"

"""
    struct Relaxation{R, F, M, T, L, Tr}

Callable object for restoring fields to a `target` at some `rate` and within a
`mask`ed region in `x, y, z`. The `F` parameter (the `relaxed` quantity) sits
second because the kernel dispatches on it. The trailing parameters `L` and
`Tr` carry the instantiated `location` and user-supplied `transform`. After
`materialize_forcing` runs, `relaxed` is the forced field (or, when `transform`
is set, a `Field` holding `transform(forced_field)`); `location` and
`transform` carry their materialized values. All three are `nothing` for a
freshly-constructed `Relaxation`.
"""
struct Relaxation{R, F, M, T, L, Tr}
         rate :: R
      relaxed :: F
         mask :: M
       target :: T
     location :: L
    transform :: Tr
end

"""
    Relaxation(; rate, mask=onefunction, target=zerofunction, transform=nothing)

Returns a `Forcing` that restores a field to `target(X..., t)`
at the specified `rate`, in the region `mask(X...)`.

The functions `onefunction` and `zerofunction` always return 1 and 0, respectively.
Thus the default `mask` leaves the whole domain uncovered, and the default `target` is zero.

`target` may be a `Number`, a function `target(x, y, z, t)`, an `AbstractField` sharing
the forced field's grid and location, or a `FieldTimeSeries` (interpolated in space and
time at each grid point).

`transform`, if provided, is applied to the forced field at materialize time to build
the `relaxed` quantity: the tendency becomes
`rate * mask(X) * (target(X, t) - transform(field)[i, j, k])`. The user-supplied `target`
is still used as the RHS of the relaxation. This is useful for nudging a reduction
of the forced field — e.g. its horizontal average — toward a target profile.
`transform` may be a callable returning a `Field`
(`transform = f -> Field(Average(f, dims=(1, 2)))`) or a `Symbol` shortcut. The only
supported `Symbol` is `:horizontal_average`, equivalent to
`f -> Field(Average(f, dims=(1, 2)))`. Transformed quantities are recomputed each
step via `compute_forcing!`. `transform` is not supported with a `FieldTimeSeries` target.

Example
=======

* Restore a field to zero on a timescale of "3600" (equal
  to one hour if the time units of the simulation are seconds).

```jldoctest relaxation
using Oceananigans

damping = Relaxation(rate = 1/3600)

# output
Relaxation{Float64}
├──   rate: 0.0002777777777777778
├──   mask: 1
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
Relaxation{Float64}
├──   rate: 0.016666666666666666
├──   mask: exp(-(z + 100.0)^2 / (2 * 25.0^2))
└── target: 20.0 + 0.001 * z
```
"""
Relaxation(; rate, mask=onefunction, target=zerofunction, transform=nothing) =
    Relaxation(rate, nothing, mask, target, nothing, transform)

apply_transform(t,          field) = t(field)
apply_transform(s::Symbol,  field) = apply_transform(Val(s), field)
apply_transform(::Val{:horizontal_average}, field) = Field(Average(field, dims=(1, 2)))
apply_transform(::Val{S}, field) where S =
  throw(ArgumentError("unknown transform :$S; supported: :horizontal_average"))

# Note `F<:AbstractArray` (not `F<:AbstractField`): `Adapt.adapt_structure(to, ::Field) = adapt(to, f.data)`
# unwraps Field to its underlying OffsetArray on GPU, so an `<:AbstractField` constraint here
# would prevent kernel dispatch on the device and the compiler would emit a MethodError.
@inline function (r::Relaxation{R, F})(i, j, k, grid, clock, model_fields) where {R, F<:AbstractArray}
    X  = node(i, j, k, grid, r.location...)
    ϕ  = @inbounds r.relaxed[i, j, k]
    ϕᵣ = evaluate_target(r.target, i, j, k, X, clock.time)
    return r.rate * r.mask(X...) * (ϕᵣ - ϕ)
end

"""
    InterpolatedFieldTarget(field, loc, grid)

Wraps an `AbstractField` target whose grid differs from the forced field's so that
`evaluate_target` interpolates from the wrapped field at each kernel call.
`loc` and `grid` are carried separately because `Adapt`ing a `Field` reduces it to
its underlying data array and discards the metadata `interpolate` needs.
"""
struct InterpolatedFieldTarget{F, L, G}
    field :: F
      loc :: L
     grid :: G
end

Adapt.adapt_structure(to, t::InterpolatedFieldTarget) =
    InterpolatedFieldTarget(Adapt.adapt(to, t.field), t.loc, Adapt.adapt(to, t.grid))

Base.summary(t::InterpolatedFieldTarget) = "interpolated " * summary(t.field)

"""
    FieldTimeSeriesTarget(field_time_series, grid)

Wraps a `FieldTimeSeries` together with its grid so that the relaxation kernel can
`interpolate(X, Time(t), fts, ..., grid)` on the device. `Adapt`ing a
`FieldTimeSeries` to a `GPUAdaptedFieldTimeSeries` drops the grid, so it is cached
separately on this wrapper at materialize time.
"""
struct FieldTimeSeriesTarget{F, G}
    field_time_series :: F
    grid              :: G
end

Adapt.adapt_structure(to, t::FieldTimeSeriesTarget) =
    FieldTimeSeriesTarget(Adapt.adapt(to, t.field_time_series), Adapt.adapt(to, t.grid))

Base.summary(t::FieldTimeSeriesTarget) = summary(t.field_time_series)

const FieldRelaxation           = Relaxation{<:Any, <:Any, <:Any, <:AbstractField}
const FieldTimeSeriesRelaxation = Relaxation{<:Any, <:Any, <:Any, <:Union{FlavorOfFTS, FieldTimeSeriesTarget}}

@inline evaluate_target(c::Number,                    i, j, k, X, t) = c
@inline evaluate_target(f,                            i, j, k, X, t) = f(X..., t)
@inline evaluate_target(f::AbstractArray,             i, j, k, X, t) = @inbounds f[i, j, k]
@inline evaluate_target(t::InterpolatedFieldTarget,   i, j, k, X, time) =
    interpolate(X, t.field, t.loc, t.grid)
@inline evaluate_target(t::FieldTimeSeriesTarget,     i, j, k, X, time) =
    interpolate(X, Time(time), t.field_time_series, instantiated_location(t.field_time_series), t.grid)

# Default: take user-supplied target as-is (Numbers, callables, plain arrays, …).
materialize_target(target, field) = target

# AbstractField target: keep direct indexing when the grid matches the forced field
# and either the locations match or the target is reduced (so `f[i, j, k]` broadcasts
# the reduced dimensions, e.g. a horizontally-averaged field). Otherwise wrap so we
# interpolate from the target's grid and location at each kernel call.
function materialize_target(target::AbstractField, field)
    target_loc = instantiated_location(target)
    field_loc  = instantiated_location(field)
    is_reduced = any(isnothing, target_loc)
    if target.grid === field.grid && (target_loc == field_loc || is_reduced)
        return target
    else
        return InterpolatedFieldTarget(target, target_loc, target.grid)
    end
end

function materialize_forcing(forcing::Relaxation, field, field_name, model_field_names)
    target = materialize_target(forcing.target, field)
    relaxed = isnothing(forcing.transform) ? field : apply_transform(forcing.transform, field)
    return Relaxation(forcing.rate, relaxed, forcing.mask, target,
                      instantiated_location(field), forcing.transform)
end

function materialize_forcing(forcing::Relaxation{<:Any, <:Any, <:Any, <:FlavorOfFTS}, field,
                             field_name, model_field_names)
    isnothing(forcing.transform) ||
        throw(ArgumentError("`transform` is not supported with a `FieldTimeSeries` target"))
    validate_fts_target_extent(forcing.target, field)
    target = FieldTimeSeriesTarget(forcing.target, forcing.target.grid)
    return Relaxation(forcing.rate, field, forcing.mask, target, instantiated_location(field), nothing)
end

function validate_fts_target_extent(fts, field)
    fts_grid = fts.grid
    sim_grid = field.grid
    fts_loc = instantiated_location(fts)
    sim_loc = instantiated_location(field)

    # The kernel queries `interpolate(X, …, fts, fts_loc, fts.grid)` with
    # `X = node(i, j, k, sim_grid, sim_loc...)`; if X falls outside the FTS
    # node range, trilinear interpolation reads from FTS halos (which
    # `set!(fts[n], …)` does not fill), producing silently wrong values near
    # the boundary.
    #
    # Require horizontal bracketing only. The vertical is intentionally NOT required to bracket: a
    # target may not span the model's full column (e.g. a limited-area child nested in ERA5
    # pressure-level data, which does not reach the surface), and interpolation clamps to the target's
    # edge value there on a clamping vertical (e.g. a geopotential-height `PressureLevelGrid`) rather
    # than reading halos.
    for (label, nodes_fn) in (("x", xnodes), ("y", ynodes))
        sim_lo, sim_hi = extrema(nodes_fn(sim_grid, sim_loc...))
        fts_lo, fts_hi = extrema(nodes_fn(fts_grid, fts_loc...))
        (fts_lo ≤ sim_lo && sim_hi ≤ fts_hi) ||
            throw(ArgumentError(
                "FieldTimeSeries target $label-extent [$fts_lo, $fts_hi] does not " *
                "bracket model grid $label-extent [$sim_lo, $sim_hi]"))
    end
    return nothing
end

# `transform` is host-side only (`compute_forcing!`, `show`) and may not be isbits
# (e.g. a `Symbol`), so the device copy drops it.
Adapt.adapt_structure(to, r::Relaxation) =
    Relaxation(Adapt.adapt(to, r.rate),
               Adapt.adapt(to, r.relaxed),
               Adapt.adapt(to, r.mask),
               Adapt.adapt(to, r.target),
               Adapt.adapt(to, r.location),
               nothing)

"""Show the innards of a `Relaxation` in the REPL."""
function Base.show(io::IO, relaxation::Relaxation)
    FT = typeof(relaxation.rate)
    print(io, "Relaxation{$FT}")
    isnothing(relaxation.location) || print(io, " at ", show_location(map(typeof, relaxation.location)...))
    rows = Pair{String, String}["rate" => string(relaxation.rate)]
    isnothing(relaxation.relaxed) || push!(rows, "relaxed" => prettysummary(relaxation.relaxed))
    push!(rows, "mask"   => summary(relaxation.mask))
    push!(rows, "target" => summary(relaxation.target))
    isnothing(relaxation.transform) || push!(rows, "transform" => string(relaxation.transform))
    width = maximum(length(first(r)) for r in rows)
    for (i, (key, value)) in enumerate(rows)
        prefix = i == length(rows) ? "└── " : "├── "
        print(io, "\n", prefix, lpad(key, width), ": ", value)
    end
end

function Base.summary(relaxation::Relaxation)
    parts = ["rate=$(relaxation.rate)",
             "mask=$(summary(relaxation.mask))",
             "target=$(summary(relaxation.target))"]
    isnothing(relaxation.transform) || push!(parts, "transform=$(relaxation.transform)")
    return "Relaxation(" * join(parts, ", ") * ")"
end

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

@inline (g::GaussianMask)(dim) = exp(-(=dim - g.center)^2 / (2 * g.width^2))

@inline (g::GaussianMask{:x})(x, dim) = g(x)
@inline (g::GaussianMask{:z})(dim, z) = g(z)

@inline (g::GaussianMask{:x})(x, y, z) = g(x)
@inline (g::GaussianMask{:y})(x, y, z) = g(y)
@inline (g::GaussianMask{:z})(x, y, z) = g(z)

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
