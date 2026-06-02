using ..Models: open_boundary_inflow_transport,
                correct_left_boundary_transport!,
                correct_right_boundary_transport!

enforce_net_zero_transport!(velocities, ::Nothing) = nothing

"""
    enforce_net_zero_transport!(velocities, boundary_transport)

Correct boundary values in `velocities` so that the combined net transport through all
`NormalFlowBoundaryCondition` boundaries with a radiation scheme vanishes — the
solvability condition for the incompressible-pressure Poisson problem solved
by `NonhydrostaticModel` without a free surface.

`velocities` is a `NamedTuple` of three face-normal fields, e.g. `(; u, v, w)`.
`boundary_transport` is the container returned by `initialize_boundary_transport`,
or `nothing` when no open boundaries require correction (in which case this is
a no-op).
"""
function enforce_net_zero_transport!(velocities, boundary_transport)
    u, v, w = velocities

    ∮udA = open_boundary_inflow_transport(boundary_transport, velocities)
    A = boundary_transport.total_area_scheme_boundaries

    A⁻¹_∮udA = ∮udA / A

    correct_left_boundary_transport!(u, u.boundary_conditions.west,   Val(:west),   A⁻¹_∮udA)
    correct_left_boundary_transport!(v, v.boundary_conditions.south,  Val(:south),  A⁻¹_∮udA)
    correct_left_boundary_transport!(w, w.boundary_conditions.bottom, Val(:bottom), A⁻¹_∮udA)

    correct_right_boundary_transport!(u, u.boundary_conditions.east,  Val(:east),  A⁻¹_∮udA)
    correct_right_boundary_transport!(v, v.boundary_conditions.north, Val(:north), A⁻¹_∮udA)
    correct_right_boundary_transport!(w, w.boundary_conditions.top,   Val(:top),   A⁻¹_∮udA)

    return nothing
end
