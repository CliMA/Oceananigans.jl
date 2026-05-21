using Oceananigans.Grids: node, xnodes, ynodes, znodes
using Oceananigans.Fields: AbstractField, Field, compute!
using Oceananigans.AbstractOperations: Average
using Oceananigans.OutputReaders: interpolate
using Oceananigans: instantiated_location

@inline zerofunction(args...) = 0
@inline onefunction(args...) = 1

T_zerofunction = typeof(zerofunction)
T_onefunction = typeof(onefunction)

Base.summary(::T_zerofunction) = "0"
Base.summary(::T_onefunction) = "1"

"""
    struct Relaxation{R, M, T, F, L, Tr}

Callable object for restoring fields to a `target` at some `rate` and within a
`mask`ed region in `x, y, z`. The trailing parameters `F`, `L`, and `Tr` carry
the forced `field`, its instantiated `location`, and the user-supplied
`transform` once `materialize_forcing` runs (`nothing` for a freshly-constructed
`Relaxation`).
"""
struct Relaxation{R, M, T, F, L, Tr}
         rate :: R
         mask :: M
       target :: T
        field :: F
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
the target — overriding `target`. It may be a callable (`transform = f -> Field(Average(f, dims=(1, 2)))`)
or a `Symbol` shortcut (`transform = :horizontal_average`). Transformed targets are
recomputed each step via `compute_forcing!`.

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
Relaxation(; rate, mask=onefunction, target=zerofunction, transform=nothing) =
    Relaxation(rate, mask, target, nothing, nothing, transform)

const FieldRelaxation{R, M, T<:AbstractField, F, L, Tr}            = Relaxation{R, M, T, F, L, Tr}
const FieldTimeSeriesRelaxation{R, M, T<:FlavorOfFTS, F, L, Tr}    = Relaxation{R, M, T, F, L, Tr}

apply_transform(t,          field) = t(field)
apply_transform(s::Symbol,  field) = apply_transform(Val(s), field)
apply_transform(::Val{:horizontal_average}, field) = Field(Average(field, dims=(1, 2)))
apply_transform(::Val{S}, field) where S =
  throw(ArgumentError("unknown transform :$S; supported: :horizontal_average"))

@inline function (r::Relaxation{R, M, T, F, L, Tr})(i, j, k, grid, clock, model_fields) where {R, M, T, F<:AbstractField, L, Tr}
    X  = node(i, j, k, grid, r.location...)
    ϕ  = @inbounds r.field[i, j, k]
    ϕᵣ = evaluate_target(r.target, i, j, k, X, clock.time)
    return r.rate * r.mask(X...) * (ϕᵣ - ϕ)
end

@inline evaluate_target(c::Number,          i, j, k, X, t) = c
@inline evaluate_target(f,                  i, j, k, X, t) = f(X..., t)
@inline evaluate_target(f::AbstractArray,   i, j, k, X, t) = @inbounds f[i, j, k]
@inline evaluate_target(fts::FlavorOfFTS,   i, j, k, X, t) =
    interpolate(X, Time(t), fts, instantiated_location(fts), fts.grid)

validate_target(target, field) = nothing

function validate_target(target::AbstractField, field)
    target.grid === field.grid ||
        throw(ArgumentError("AbstractField `Relaxation` target must share its grid with the forced field"))
    instantiated_location(target) == instantiated_location(field) ||
        throw(ArgumentError("AbstractField `Relaxation` target must share its location with the forced field"))
    return nothing
end

function materialize_forcing(forcing::Relaxation, field, field_name, model_field_names)
    target = if isnothing(forcing.transform)
        validate_target(forcing.target, field)
        forcing.target
    else
        apply_transform(forcing.transform, field)
    end
    return Relaxation(forcing.rate, forcing.mask, target, field, instantiated_location(field), forcing.transform)
end

function materialize_forcing(forcing::Relaxation{<:Any, <:Any, <:FlavorOfFTS}, field,
                             field_name, model_field_names)
    isnothing(forcing.transform) ||
        throw(ArgumentError("`transform` is not supported with a `FieldTimeSeries` target"))
    validate_fts_target_extent(forcing.target, field)
    return Relaxation(forcing.rate, forcing.mask, forcing.target, field, instantiated_location(field), nothing)
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

Adapt.adapt_structure(to, r::Relaxation) =
    Relaxation(Adapt.adapt(to, r.rate),
               Adapt.adapt(to, r.mask),
               Adapt.adapt(to, r.target),
               Adapt.adapt(to, r.field),
               Adapt.adapt(to, r.location),
               Adapt.adapt(to, r.transform))

"""Show the innards of a `Relaxation` in the REPL."""
Base.show(io::IO, relaxation::Relaxation{R, M, T}) where {R, M, T} =
    print(io, "Relaxation{$R, $M, $T}", "\n",
        "├── rate: $(relaxation.rate)", "\n",
        "├── mask: $(summary(relaxation.mask))", "\n",
        "└── target: $(summary(relaxation.target))")

Base.summary(relaxation::Relaxation) =
    "Relaxation(rate=$(relaxation.rate), mask=$(summary(relaxation.mask)), target=$(summary(relaxation.target)))"

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
