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

# Whether a neighbouring rank, or the other end of a periodic domain, lies to the left or the right along a direction
neighbour_on_left(::Type{<:Union{Grids.Periodic, LeftConnected, FullyConnected}}) = true
neighbour_on_left(T) = false
neighbour_on_right(::Type{<:Union{Grids.Periodic, RightConnected, FullyConnected}}) = true
neighbour_on_right(T) = false

"""
    boundary_state_field(grid, loc, dim)

A field on `grid` at the location `loc` of a boundary-conditioned field, reduced in the direction `dim` normal to the
boundary, to hold state that an open boundary scheme keeps along the boundary.
"""
function boundary_state_field end

# The values of a field reduced in direction `dim`, indexed along the boundary and in the other direction
along_boundary(f, dim) = dim == 1 ? view(f.data, 1, :, :) :
                         dim == 2 ? view(f.data, :, 1, :) :
                                    view(f.data, :, :, 1)

# A `target_transport` is `nothing`, a fixed transport, or a callable of the grid (kept as is).
convert_target_transport(FT, ::Nothing) = nothing
convert_target_transport(FT, target_transport::Number) = convert(FT, target_transport)
convert_target_transport(FT, target_transport) = target_transport
