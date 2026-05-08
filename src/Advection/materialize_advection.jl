"""
    materialize_advection(advection, grid)

Return a fully materialized advection scheme appropriate for `grid`.
It exists to allow advection schemes to defer specialising settings until
additional information about the backend from grid is available.

For example it allows to set per-backend defaults for WENO weight computation
setting.
"""
materialize_advection(advection, grid) = advection
materialize_advection(::Nothing, grid) = nothing

materialize_advection(advection::FluxFormAdvection, grid) = FluxFormAdvection(
    materialize_advection(advection.x, grid),
    materialize_advection(advection.y, grid),
    materialize_advection(advection.z, grid),
)

# Upwinding treatments hold a cross_scheme that may contain a deferred WENO weight computation
materialize_advection(u::OnlySelfUpwinding, grid) =
    OnlySelfUpwinding(materialize_advection(u.cross_scheme, grid),
                      u.δU_stencil, u.δV_stencil, u.δu²_stencil, u.δv²_stencil)

materialize_advection(u::CrossAndSelfUpwinding, grid) =
    CrossAndSelfUpwinding(materialize_advection(u.cross_scheme, grid),
                          u.divergence_stencil, u.δu²_stencil, u.δv²_stencil)

materialize_advection(u::VelocityUpwinding, grid) =
    VelocityUpwinding(materialize_advection(u.cross_scheme, grid))

# VectorInvariant wraps multiple sub-schemes; recurse into each
materialize_advection(vi::VectorInvariant{N, FT, M}, grid) where {N, FT, M} =
    VectorInvariant{N, FT, M}(materialize_advection(vi.vorticity_scheme, grid),
                              vi.vorticity_stencil,
                              materialize_advection(vi.vertical_advection_scheme, grid),
                              materialize_advection(vi.kinetic_energy_gradient_scheme, grid),
                              materialize_advection(vi.divergence_scheme, grid),
                              materialize_advection(vi.upwinding, grid))

# WENO with a concrete WCT just recurses into the inner advecting_velocity_scheme
materialize_advection(weno::WENO{N, FT, WCT, PP, SI, M}, grid) where {N, FT, WCT, PP, SI, M} =
    WENO{N, FT, WCT, M}(weno.bounds,
                        materialize_advection(weno.advecting_velocity_scheme, grid))

# WENO with deferred WCT picks the architecture-dependent default
materialize_advection(weno::WENO{N, FT, Nothing, PP, SI, M}, grid) where {N, FT, PP, SI, M} =
    WENO{N, FT, default_weno_weight_computation(architecture(grid)), M}(
        weno.bounds,
        materialize_advection(weno.advecting_velocity_scheme, grid))

materialize_advection(scheme::UpwindBiased{N, FT, SI, M}, grid) where {N, FT, SI, M} =
    UpwindBiased{N, FT, M}(materialize_advection(scheme.advecting_velocity_scheme, grid))

# Centered has no inner schemes on this branch; return as-is.
materialize_advection(scheme::Centered, grid) = scheme

# Default weight-computation strategy per architecture. The general default uses
# the backend-optimized division (`Base.FastMath.div_fast` on CPU; NVPTX `rcp.approx`
# on CUDA via OceananigansCUDAExt). Reactant and Distributed overrides live in
# their respective extension/source modules.
default_weno_weight_computation(arch) = BackendOptimizedDivision
