#####
##### TracerReservoir open boundary scheme
#####

"""
    TracerReservoir(; inflow_length_scale = 0, outflow_length_scale = 0)

Open boundary condition for tracers that carries a reservoir value `cʳ` just outside each
boundary point, after the tracer reservoirs of MOM6. Each step the flow advects a distance
`d = |uₙ| Δt` across the boundary face, and the reservoir relaxes toward the upstream value
over a length scale `L`:

    cʳ ← (cʳ + a c★) / (1 + a),    a = d / L

On outflow `c★` is the adjacent interior value and `L = outflow_length_scale`; on inflow `c★`
is the exterior value `cᵉˣᵗ` and `L = inflow_length_scale`. The boundary halo value is set to
`cʳ`, the advecting velocity is taken at the boundary face, and the reservoir starts at `cᵉˣᵗ`.

`L = 0` sets `cʳ = c★` every step, `L = Inf` freezes the reservoir, and a finite `L` relaxes
it. With both length scales `0`, the default, the condition is a memoryless
`ValueBoundaryCondition`: zero-gradient on outflow and `cᵉˣᵗ` on inflow.

`TracerReservoir` is a `Value` scheme for `Center`-located fields.

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

# Backward-Euler relaxation of the reservoir toward the upstream value c★ over the length scale L.
@inline function reservoir_update(cʳ, cᴵ, cᵉˣᵗ, d, outflow, reservoir)
    L  = ifelse(outflow, reservoir.outflow_length_scale, reservoir.inflow_length_scale)
    c★ = ifelse(outflow, cᴵ, cᵉˣᵗ)
    a  = d / L
    return ifelse(L == 0, c★, (cʳ + a * c★) / (1 + a))
end

# The reservoir is advanced once per time step from an anchor: an anchored fill (stage ≤ 1)
# promotes the latest value to the anchor, and later stages re-step from it. The first fill
# (Δt = Inf) starts the reservoir at the exterior value.
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
