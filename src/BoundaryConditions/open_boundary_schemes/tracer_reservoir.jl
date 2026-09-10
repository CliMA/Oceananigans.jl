#####
##### TracerReservoir open boundary scheme (after MOM6's OBC tracer reservoirs)
#####

"""
    TracerReservoir(; inflow_length_scale = 0, outflow_length_scale = 0)

A tracer open boundary condition with memory, following the tracer reservoirs of MOM6
(`MOM_open_boundary.F90`).

A prescribed-value open boundary has no memory: the instant the flow reverses, water
re-entering the domain carries the exterior value `cᵉˣᵗ`, no matter what just left. Across a
tidal cycle, an eddy brushing the boundary, or any oscillating flow, that manufactures a
spurious tracer flux — the domain exports its own water and imports someone else's.

A reservoir fixes this by carrying one extra value per boundary point, `cʳ`, which is the
concentration of the water sitting just outside the boundary. It is not relaxed in time but
in *distance advected*: each step the flow moves a distance `d = |uₙ| Δt` across the
boundary face, and `cʳ` is relaxed over a length scale `L` toward whichever reservoir the
flow is filling,

    cʳ ← (cʳ + a c★) / (1 + a) ,    a = d / L

taken as a backward-Euler step so it is unconditionally stable for any `Δt`. The target
`c★` and the length scale `L` depend on the direction of the flow:

  - **Outflow** (`uₙ` leaving the domain): `c★ = cᴵ`, the adjacent interior value, and
    `L = outflow_length_scale`. The reservoir fills with the water the domain is exporting.
  - **Inflow** (`uₙ` entering): `c★ = cᵉˣᵗ`, the prescribed exterior value, and
    `L = inflow_length_scale`. The reservoir is flushed toward the external state.

The boundary halo value is then set to `cʳ`. So on inflow the domain re-imports the water it
recently exported, and only converges to `cᵉˣᵗ` after a length `inflow_length_scale` of
sustained inflow.

Each length scale has three regimes, matching MOM6:

  - `L = 0` — **instant**: `cʳ = c★` every step. On outflow this is a zero-gradient
    (upwind) outflow condition; on inflow it snaps to `cᵉˣᵗ`. This is the memoryless
    behaviour of a plain `ValueBoundaryCondition`.
  - `0 < L < Inf` — **finite**: relaxation over the distance `L`, as above.
  - `L = Inf` — **frozen**: `cʳ` never changes in that flow direction.

Both length scales default to `0`, MOM6's own default, so that `TracerReservoir()` reproduces
the memoryless behaviour exactly and the reservoir is opt-in. There is no universal value to
default to: the useful `inflow_length_scale` is a property of the flow at the boundary, not of
the scheme.

Choosing `inflow_length_scale`. Note first that the benefit is not monotone — freezing the
reservoir is not the best choice, and can be much worse than a finite length scale. A reservoir
holds *one number* per boundary point. During outflow it tracks the boundary concentration; at
the reversal it holds whatever value was there at that instant, and then feeds that *constant*
back in. The water that actually returns has a declining profile, being the far tail of what
left, so a frozen reservoir over-feeds. Relaxing toward `cᵉˣᵗ` during inflow mimics that decline,
and the length scale at which it matches is set by the distance water travels across the
boundary during a reversal — the parcel excursion `D` (for an oscillation of amplitude `U` and
frequency `ω`, `D = 2U/ω`). Scanning a translating-patch problem over two patch widths and two
excursions puts the optimum at

    inflow_length_scale ≈ 0.3 D

with the error there some 8× smaller than memoryless. The scaling is with `D` and not with the
width of the tracer structure, as it must be for a scheme that relaxes over a distance advected.
It degrades when nearly all of the tracer leaves the domain (`D` ≳ 4× the structure width),
where no single-valued reservoir can help and the optimum is ill-defined.

`outflow_length_scale = 0` is almost always right: the domain should record the water it exports
immediately, and this makes the outflow condition zero-gradient, the standard choice.

`TracerReservoir` is a `Value` scheme for `Center`-located fields — tracers. It carries no
condition on the velocities, so it is used alongside a scheme for the boundary-normal
velocity ([`NormalRadiation`](@ref), [`ObliqueRadiation`](@ref) or
[`PerturbationAdvection`](@ref)) and the barotropic pair.

The reservoir slab is allocated automatically during boundary condition regularization; it is
one two-dimensional array over the boundary face, per tracer, per boundary. It is initialized
to `cᵉˣᵗ` on the first halo fill.

References
==========
* Adcroft, A. et al. (2019). "The GFDL global ocean and sea ice model OM4.0."
  Journal of Advances in Modeling Earth Systems, 11(10), 3167-3211.

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: TracerReservoir

TracerReservoir(inflow_length_scale = 20000)

# output
TracerReservoir{Float64}
├── inflow_length_scale: 20000.0
└── outflow_length_scale: 0.0
```
"""
struct TracerReservoir{FT, S}
    inflow_length_scale  :: FT
    outflow_length_scale :: FT
    cʳ  :: S  # anchor reservoir value (2D array or nothing)
    cʳˡ :: S  # latest reservoir value (2D array or nothing)
end

function TracerReservoir(FT = defaults.FloatType;
                         inflow_length_scale = 0,
                         outflow_length_scale = 0)

    inflow_length_scale  = convert(FT, inflow_length_scale)
    outflow_length_scale = convert(FT, outflow_length_scale)

    inflow_length_scale  >= 0 || throw(ArgumentError("inflow_length_scale must be non-negative"))
    outflow_length_scale >= 0 || throw(ArgumentError("outflow_length_scale must be non-negative"))

    return TracerReservoir(inflow_length_scale, outflow_length_scale, nothing, nothing)
end

Adapt.adapt_structure(to, r::TracerReservoir) =
    TracerReservoir(adapt(to, r.inflow_length_scale),
                    adapt(to, r.outflow_length_scale),
                    adapt(to, r.cʳ),
                    adapt(to, r.cʳˡ))

Base.summary(::TracerReservoir{FT}) where FT = "TracerReservoir{$FT}"

function Base.show(io::IO, r::TracerReservoir)
    print(io, summary(r), '\n')
    print(io, "├── inflow_length_scale: ",  prettysummary(r.inflow_length_scale), '\n')
    print(io, "└── outflow_length_scale: ", prettysummary(r.outflow_length_scale))
end

# The reservoir is a tracer condition: Value classification, Center-located fields.
const TRVBC = BoundaryCondition{<:Value{<:TracerReservoir}}

#####
##### Storage allocation during BC regularization
#####

function materialize_radiation_storage(reservoir::TracerReservoir, grid, loc, dim)
    FT = eltype(grid)
    Sx, Sy, Sz = size(grid, loc)
    arch = architecture(grid)

    tangential_size = dim == 1 ? (Sy, Sz) :
                      dim == 2 ? (Sx, Sz) :
                                 (Sx, Sy)

    cʳ  = on_architecture(arch, zeros(FT, tangential_size...))
    cʳˡ = on_architecture(arch, zeros(FT, tangential_size...))

    return TracerReservoir(reservoir.inflow_length_scale,
                           reservoir.outflow_length_scale,
                           cʳ, cʳˡ)
end

function regularize_boundary_condition(bc::TRVBC, grid, loc, dim, args...)
    regularized_condition = regularize_boundary_condition(bc.condition, grid, loc, dim, args...)
    reservoir = bc.classification.scheme
    materialized_reservoir = materialize_radiation_storage(reservoir, grid, loc, dim)
    classification = rebuild_classification(bc.classification, materialized_reservoir)
    return BoundaryCondition(classification, regularized_condition)
end

#####
##### The reservoir update
#####

# Backward-Euler relaxation of the reservoir toward the upstream value, over a distance.
#
#   cʳ ← (cʳ + a c★) / (1 + a),   a = |uₙ| Δt / L
#
# with (c★, L) = (cᴵ, L_out) on outflow and (cᵉˣᵗ, L_in) on inflow. `L = 0` is the instant
# limit `cʳ = c★`, handled explicitly so that no Inf or NaN is ever formed; `L = Inf` gives
# `a = 0` and leaves the reservoir frozen, which needs no special case.
#
# This is the length-scale form of MOM6's `update_segment_tracer_reservoirs`, written with a
# single branch instead of the mask-and-sentinel arithmetic that Fortran needs. The two
# implementations agree term by term: MOM6's `L_out` and `-L_in` are both `a` here, and its
# `a_out`/`a_in` switches are the `L == 0` branch.
@inline function reservoir_update(cʳ, cᴵ, cᵉˣᵗ, d, outflow, reservoir)
    L  = ifelse(outflow, reservoir.outflow_length_scale, reservoir.inflow_length_scale)
    c★ = ifelse(outflow, cᴵ, cᵉˣᵗ)

    instant = L == 0
    Lˢ = ifelse(instant, one(L), L)  # avoid dividing by zero in the unselected branch
    a  = d / Lˢ

    cʳ⁺ = (cʳ + a * c★) / (1 + a)

    return ifelse(instant, c★, cʳ⁺)
end

# The reservoir is advanced once per timestep from an anchor, following the same
# anchor/latest convention as NormalRadiation: at an anchored fill (stage ≤ 1) the value from
# the end of the previous step is promoted to the anchor; later stages of a multi-stage
# stepper re-step from that same anchor rather than compounding.
#
# On the very first fill there is no clock (or no completed stage), so Δt is infinite; the
# reservoir is then initialized to the exterior value, matching MOM6's `tres = t`.
@inline function reservoir_halo!(cᵇ, cᴵ, l, m, grid, c, bc, uₙ, outflow, closed, clock, model_fields)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    reservoir = bc.classification.scheme

    @inbounds begin
        cᵉˣᵗ = getbc(bc, l, m, grid, clock, model_fields)

        cʳᵃ = ifelse(anchored, reservoir.cʳˡ[l, m], reservoir.cʳ[l, m])
        cʳⁿ = ifelse(first_call, cᵉˣᵗ, cʳᵃ)

        d = abs(uₙ) * Δt
        cʳⁿ⁺¹ = reservoir_update(cʳⁿ, c[cᴵ...], cᵉˣᵗ, d, outflow, reservoir)

        c[cᵇ...] = ifelse(closed, zero(grid), cʳⁿ⁺¹)
        reservoir.cʳ[l, m]  = cʳⁿ   # anchor for later stages
        reservoir.cʳˡ[l, m] = cʳⁿ⁺¹ # latest, promoted at the next anchored fill
    end

    return nothing
end

#####
##### Halo filling — Center-located fields on the six boundaries
#####

# The advecting velocity is taken at the boundary FACE, not one cell into the interior: it is
# the flux through that face that carries tracer across the boundary, and it is what MOM6's
# reservoir uses (`uhr` at the segment face).

@inline function _fill_east_halo!(j, k, grid, c, bc::TRVBC, loc::CAA, clock, model_fields)
    ℓx, ℓy, ℓz = loc
    i = grid.Nx + 1
    uₙ = @inbounds model_fields.u[i, j, k]
    closed = immersed_peripheral_node(grid.Nx, j, k, grid, Center(), ℓy, ℓz)
    return reservoir_halo!((i, j, k), (i-1, j, k), j, k, grid, c, bc, uₙ, uₙ >= 0, closed, clock, model_fields)
end

@inline function _fill_west_halo!(j, k, grid, c, bc::TRVBC, loc::CAA, clock, model_fields)
    ℓx, ℓy, ℓz = loc
    uₙ = @inbounds model_fields.u[1, j, k]
    closed = immersed_peripheral_node(1, j, k, grid, Center(), ℓy, ℓz)
    return reservoir_halo!((0, j, k), (1, j, k), j, k, grid, c, bc, uₙ, uₙ <= 0, closed, clock, model_fields)
end

@inline function _fill_north_halo!(i, k, grid, c, bc::TRVBC, loc::ACA, clock, model_fields)
    ℓx, ℓy, ℓz = loc
    j = grid.Ny + 1
    uₙ = @inbounds model_fields.v[i, j, k]
    closed = immersed_peripheral_node(i, grid.Ny, k, grid, ℓx, Center(), ℓz)
    return reservoir_halo!((i, j, k), (i, j-1, k), i, k, grid, c, bc, uₙ, uₙ >= 0, closed, clock, model_fields)
end

@inline function _fill_south_halo!(i, k, grid, c, bc::TRVBC, loc::ACA, clock, model_fields)
    ℓx, ℓy, ℓz = loc
    uₙ = @inbounds model_fields.v[i, 1, k]
    closed = immersed_peripheral_node(i, 1, k, grid, ℓx, Center(), ℓz)
    return reservoir_halo!((i, 0, k), (i, 1, k), i, k, grid, c, bc, uₙ, uₙ <= 0, closed, clock, model_fields)
end

@inline function _fill_top_halo!(i, j, grid, c, bc::TRVBC, loc::AAC, clock, model_fields)
    ℓx, ℓy, ℓz = loc
    k = grid.Nz + 1
    uₙ = @inbounds model_fields.w[i, j, k]
    closed = immersed_peripheral_node(i, j, grid.Nz, grid, ℓx, ℓy, Center())
    return reservoir_halo!((i, j, k), (i, j, k-1), i, j, grid, c, bc, uₙ, uₙ >= 0, closed, clock, model_fields)
end

@inline function _fill_bottom_halo!(i, j, grid, c, bc::TRVBC, loc::AAC, clock, model_fields)
    ℓx, ℓy, ℓz = loc
    uₙ = @inbounds model_fields.w[i, j, 1]
    closed = immersed_peripheral_node(i, j, 1, grid, ℓx, ℓy, Center())
    return reservoir_halo!((i, j, 0), (i, j, 1), i, j, grid, c, bc, uₙ, uₙ <= 0, closed, clock, model_fields)
end
