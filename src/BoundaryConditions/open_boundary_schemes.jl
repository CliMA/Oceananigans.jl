using Oceananigans: defaults
using Oceananigans.Grids: column_depthᶠᶜᵃ, column_depthᶜᶠᵃ, column_depthᶜᶜᵃ, immersed_peripheral_node

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

include("open_boundary_schemes/perturbation_advection.jl")
include("open_boundary_schemes/gravity_wave_schemes.jl")
include("open_boundary_schemes/normal_radiation_scheme.jl")
