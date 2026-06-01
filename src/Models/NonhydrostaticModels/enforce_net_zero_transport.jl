using ..Models: open_boundary_inflow_transport,
                update_open_boundary_transport!,
                apply_targeted_left_boundary_correction!,
                apply_targeted_right_boundary_correction!,
                correct_left_boundary_transport!,
                correct_right_boundary_transport!

enforce_net_zero_transport!(velocities, ::Nothing) = nothing

"""
    enforce_net_zero_transport!(velocities, boundary_transport)

Correct boundary values in `velocities` so that the combined net transport through all
`OpenBoundaryCondition` boundaries with a radiation scheme vanishes — the
solvability condition for the incompressible-pressure Poisson problem solved
by `NonhydrostaticModel` without a free surface.

`velocities` is a `NamedTuple` of three face-normal fields, e.g. `(; u, v, w)`.
`boundary_transport` is the container returned by `initialize_boundary_transport`,
or `nothing` when no open boundaries require correction (in which case this is
a no-op).

Boundaries whose scheme carries a `target_transport` are corrected first by a
per-boundary shift so that their net transport matches the target exactly; the
remaining net imbalance is then distributed uniformly over the pool boundaries
(those without a `target_transport`).
"""
function enforce_net_zero_transport!(velocities, boundary_transport)
    u, v, w = velocities

    # Step 1: compute current fluxes before any corrections.
    update_open_boundary_transport!(boundary_transport)

    # Step 2: per-boundary corrections for targeted boundaries.
    apply_targeted_left_boundary_correction!(u, u.boundary_conditions.west,   Val(:west),   boundary_transport)
    apply_targeted_left_boundary_correction!(v, v.boundary_conditions.south,  Val(:south),  boundary_transport)
    apply_targeted_left_boundary_correction!(w, w.boundary_conditions.bottom, Val(:bottom), boundary_transport)
    apply_targeted_right_boundary_correction!(u, u.boundary_conditions.east,  Val(:east),   boundary_transport)
    apply_targeted_right_boundary_correction!(v, v.boundary_conditions.north, Val(:north),  boundary_transport)
    apply_targeted_right_boundary_correction!(w, w.boundary_conditions.top,   Val(:top),    boundary_transport)

    # Step 3: distribute the remaining imbalance over pool boundaries.
    A = boundary_transport.total_area_pool_boundaries
    ∮udA = open_boundary_inflow_transport(boundary_transport, velocities)

    if iszero(A)
        FT = eltype(u)
        net_zero_tolerance = sqrt(eps(FT)) * boundary_transport.total_area_scheme_boundaries
        if abs(∮udA) > net_zero_tolerance
            error("Every open boundary in this `NonhydrostaticModel` carries a `target_transport`, " *
                  "but the targets do not sum to a net-zero transport across the domain " *
                  "(net inflow = $∮udA). The pressure Poisson problem has no solution in this " *
                  "configuration. Make sure inflow targets balance outflow targets, or leave " *
                  "at least one open boundary without a `target_transport` so the global pool " *
                  "correction can absorb the residual.")
        end
        return nothing
    end

    A⁻¹_∮udA = ∮udA / A

    correct_left_boundary_transport!(u, u.boundary_conditions.west,   Val(:west),   A⁻¹_∮udA)
    correct_left_boundary_transport!(v, v.boundary_conditions.south,  Val(:south),  A⁻¹_∮udA)
    correct_left_boundary_transport!(w, w.boundary_conditions.bottom, Val(:bottom), A⁻¹_∮udA)

    correct_right_boundary_transport!(u, u.boundary_conditions.east,  Val(:east),  A⁻¹_∮udA)
    correct_right_boundary_transport!(v, v.boundary_conditions.north, Val(:north), A⁻¹_∮udA)
    correct_right_boundary_transport!(w, w.boundary_conditions.top,   Val(:top),   A⁻¹_∮udA)

    return nothing
end
