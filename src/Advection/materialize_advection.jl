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

# VectorInvariant wraps multiple sub-schemes; recurse into each
materialize_advection(vi::VectorInvariant{N,FT,M}, grid) where {N,FT,M} =
    VectorInvariant{N,FT,M}(
        materialize_advection(vi.vorticity_scheme, grid),
        vi.vorticity_stencil,
        materialize_advection(vi.vertical_advection_scheme, grid),
        materialize_advection(vi.kinetic_energy_gradient_scheme, grid),
        materialize_advection(vi.divergence_scheme, grid),
        vi.upwinding,
    )


materialize_advection(weno::WENO{N,FT,WCT}, grid) where {N,FT,WCT} = WENO{N,FT,WCT}(
    weno.bounds,
    materialize_advection(weno.buffer_scheme, grid),
    materialize_advection(weno.advecting_velocity_scheme, grid),
)

materialize_advection(weno::WENO{N,FT,Nothing}, grid) where {N,FT} =
    WENO{N,FT,Oceananigans.defaults.weno_weight_computation}(
        weno.bounds,
        materialize_advection(weno.buffer_scheme, grid),
        materialize_advection(weno.advecting_velocity_scheme, grid),
    )

materialize_advection(scheme::UpwindBiased{N,FT}, grid) where {N,FT} = UpwindBiased{N,FT}(
    materialize_advection(scheme.buffer_scheme, grid),
    materialize_advection(scheme.advecting_velocity_scheme, grid),
)

materialize_advection(scheme::Centered{N,FT}, grid) where {N,FT} =
    Centered{N,FT}(materialize_advection(scheme.buffer_scheme, grid))
