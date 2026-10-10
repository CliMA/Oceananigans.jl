using Oceananigans: defaults
using Oceananigans.Grids: column_depthᶠᶜᵃ, column_depthᶜᶠᵃ, column_depthᶜᶜᵃ, immersed_peripheral_node
using Oceananigans.Utils: getnamewrapper

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

# A scheme anchors at most once per iteration at each boundary point: a second fill in the same iteration
# recomputes the step from the anchored values instead of advancing it again.
@inline anchored_fill(clock, anchors, t, k) = anchored_fill(clock) & (@inbounds anchors[t, k] != clock.iteration)
@inline anchored_fill(::Nothing, anchors, t, k) = true

# A first call initializes the boundary value without taking a step, so it does not count as an anchor.
@inline record_anchor!(anchors, t, k, clock, anchored, first_call) =
    @inbounds anchors[t, k] = ifelse(anchored & !first_call, clock.iteration, anchors[t, k])

@inline record_anchor!(anchors, t, k, ::Nothing, anchored, first_call) = nothing

# A `target_transport` is `nothing`, a fixed transport, or a callable of the grid (kept as is).
convert_target_transport(FT, ::Nothing) = nothing
convert_target_transport(FT, target_transport::Number) = convert(FT, target_transport)
convert_target_transport(FT, target_transport) = target_transport
