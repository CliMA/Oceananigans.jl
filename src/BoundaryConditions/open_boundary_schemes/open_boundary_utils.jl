using Oceananigans: defaults
using Oceananigans.Grids: column_depthᶠᶜᵃ, column_depthᶜᶠᵃ, column_depthᶜᶜᵃ, immersed_peripheral_node
using Oceananigans.Utils: unparameterized_type

#####
##### Shared utilities for the open boundary schemes below
#####

# Location type aliases used to dispatch halo filling on the field's staggering.
const FAA = Tuple{Face,   Any, Any}
const CAA = Tuple{Center, Any, Any}
const AFA = Tuple{Any, Face,   Any}
const ACA = Tuple{Any, Center, Any}
const AAF = Tuple{Any, Any, Face, }
const AAC = Tuple{Any, Any, Center}

# A fill without a clock (e.g. during initialization or state reconciliation) behaves
# as a first call: Δt = 0 and zero-gradient initialization of the boundary value.
@inline stage_Δt(clock) = clock.last_stage_Δt
@inline stage_Δt(::Nothing) = Inf

@inline anchored_fill(clock) = clock.stage ≤ 1
@inline anchored_fill(::Nothing) = true

# A `target_transport` is `nothing`, a fixed transport, or a callable of the grid (kept as is).
convert_target_transport(FT, ::Nothing) = nothing
convert_target_transport(FT, target_transport::Number) = convert(FT, target_transport)
convert_target_transport(FT, target_transport) = target_transport

"""
$(TYPEDEF)

A `target_transport` that varies in time: `values` [m³ s⁻¹] at `times` [s], interpolated linearly in the model time
and held constant outside `times[1]` and `times[end]`. Evaluated on the host once per barotropic step, so device
kernels never see it.

$(TYPEDFIELDS)

```jldoctest
julia> using Oceananigans.BoundaryConditions: TransportTimeSeries

julia> target = TransportTimeSeries([0, 86400], [1e6, 3e6]);

julia> target(43200)
2.0e6
```
"""
struct TransportTimeSeries{T, V}
    times  :: T
    values :: V
end

convert_target_transport(FT, s::TransportTimeSeries) = TransportTimeSeries(convert.(FT, s.times), convert.(FT, s.values))

function (s::TransportTimeSeries)(t)
    n = clamp(searchsortedlast(s.times, t), 1, length(s.times) - 1)
    t₁, t₂ = s.times[n], s.times[n+1]
    w = clamp((t - t₁) / (t₂ - t₁), 0, 1)
    return (1 - w) * s.values[n] + w * s.values[n+1]
end

Adapt.adapt_structure(to, ::TransportTimeSeries) = nothing

Base.summary(s::TransportTimeSeries) = string("TransportTimeSeries with ", length(s.times), " times")
