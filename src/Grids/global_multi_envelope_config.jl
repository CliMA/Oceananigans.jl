####
#### "ME100": a recommended multi-envelope vertical coordinate for global modelling
####

# Surface-refined reference faces: spacing fine near the surface (z = 0), coarsening with depth, via an
# exponential stretch z(s) = -depth (exp(θ s) - 1)/(exp(θ) - 1), s ∈ [0, 1]. Returned bottom-to-surface
# (ascending), as Oceananigans expects.
function surface_refined_reference_faces(FT, depth, Nz, θ)
    faces = map(0:Nz) do k
        s = k / Nz
        -depth * (exp(θ * s) - 1) / (exp(θ) - 1)
    end
    return FT.(sort(faces))
end

"""
    global_multi_envelope_z([FT=Float64];
                            reference_depth = 6000,
                            surface_levels = 40, mid_levels = 40, bottom_levels = 20,
                            surface_stretching = 6.0)

Build the recommended global multi-envelope ("ME100") vertical coordinate: a surface-refined reference
grid split into three sub-zones — a pycnocline/shelf-following surface zone, a geopotential-like mid zone,
and a bathymetry-following bottom zone — totalling `surface_levels + mid_levels + bottom_levels` levels
(100 by default). Pass the returned discretization as the `z` argument to `RectilinearGrid` /
`LatitudeLongitudeGrid`, then call [`materialize_envelopes!`](@ref) with three envelope-depth functions
(shallowest → deepest, e.g. a pycnocline dome, a mid envelope, and the bathymetry), each clipped so they
stay strictly increasing.

This is the level *structure*; the envelope depths are problem-specific. See
`docs/plans/2026-06-16-multi-envelope-vertical-coordinate.md` for the design rationale.
"""
function global_multi_envelope_z(FT=Float64;
                                 reference_depth = 6000,
                                 surface_levels = 40,
                                 mid_levels = 40,
                                 bottom_levels = 20,
                                 surface_stretching = 6.0)

    Nz = surface_levels + mid_levels + bottom_levels
    r_faces = surface_refined_reference_faces(FT, reference_depth, Nz, surface_stretching)
    level_counts = (surface_levels, mid_levels, bottom_levels)

    return MultiEnvelopeVerticalDiscretization(r_faces; formulation = MultiEnvelope(; level_counts))
end
