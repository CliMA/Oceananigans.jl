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
struct TracerReservoir{FT, S} <: AbstractRadiationScheme{FT}
    inflow_length_scale  :: FT
    outflow_length_scale :: FT
    φᵇ :: S  # reservoir value at the base of the stage (2D array or nothing)
    φ₁ :: S  # interior value at the base of the stage (2D array or nothing)
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
                    adapt(to, r.φᵇ),
                    adapt(to, r.φ₁))

Base.summary(::TracerReservoir{FT}) where FT = "TracerReservoir{$FT}"

function Base.show(io::IO, r::TracerReservoir)
    print(io, summary(r), '\n')
    print(io, "├── inflow_length_scale: ",  prettysummary(r.inflow_length_scale), '\n')
    print(io, "└── outflow_length_scale: ", prettysummary(r.outflow_length_scale))
end

const TRVBC = BoundaryCondition{<:Value{<:TracerReservoir}}

radiation_storage(reservoir::TracerReservoir, φᵇ, φ₁) =
    TracerReservoir(reservoir.inflow_length_scale, reservoir.outflow_length_scale, φᵇ, φ₁)

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

# The flow advects `d = |uₙ| Δt` across the boundary face, with the normal velocity taken at the face
@inline function step_exterior_state!(side, l, m, grid, c, bc::TRVBC, loc, clock, model_fields, Δt)
    iᵇ, jᵇ, kᵇ = boundary_point(side, loc, grid, 0, l, m)
    i₁, j₁, k₁ = boundary_point(side, loc, grid, 1, l, m)
    reservoir = bc.classification.scheme

    @inbounds begin
        cᵉˣᵗ = getbc(bc, l, m, grid, clock, model_fields)
        uₙ = normal_velocity(side, model_fields, grid, 0, l, m)
        d = abs(uₙ) * Δt
        cʳ = reservoir_update(reservoir.φᵇ[l, m], c[i₁, j₁, k₁], cᵉˣᵗ, d, outward(side, uₙ) >= 0, reservoir)
        c[iᵇ, jᵇ, kᵇ] = ifelse(immersed_boundary_point(side, loc, grid, l, m), zero(grid), cʳ)
    end

    return nothing
end

# The reservoir starts at the exterior value
@inline function initialize_exterior_state!(side, l, m, grid, c, bc::TRVBC, loc, clock, model_fields)
    iᵇ, jᵇ, kᵇ = boundary_point(side, loc, grid, 0, l, m)
    @inbounds c[iᵇ, jᵇ, kᵇ] = getbc(bc, l, m, grid, clock, model_fields)
    return nothing
end
