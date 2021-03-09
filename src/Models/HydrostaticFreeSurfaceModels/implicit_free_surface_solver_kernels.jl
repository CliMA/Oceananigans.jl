using Oceananigans.Architectures: device
using Oceananigans.Operators: ΔzC, Δzᵃᵃᶠ, Δzᵃᵃᶜ

"""
Compute the vertical integrated volume flux from the bottom to z=0 (i.e. linear free-surface)

    `U^{*} = ∫ [(u^{*})] dz`
    `V^{*} = ∫ [(v^{*})] dz`
"""
### Note - what we really want is RHS = divergence of the vertically integrated volume flux
###        we can optimize this a bit later to do this all in one go to save using intermediate variables.
function compute_vertically_integrated_volume_flux!(free_surface, model)

    event = launch!(model.architecture,
                    model.grid,
                    :xy,
                    _compute_vertically_integrated_volume_flux!,
                    model.velocities,
                    model.grid,
                    free_surface.barotropic_volume_flux,
                    dependencies=Event(device(model.architecture)))

    return event
end
@kernel function _compute_vertically_integrated_volume_flux!(U, grid, barotropic_volume_flux )
    i, j = @index(Global, NTuple)
    # U.w[i, j, 1] = 0 is enforced via halo regions.
    barotropic_volume_flux.u[i, j, 1] = 0.
    barotropic_volume_flux.v[i, j, 1] = 0.
    @unroll for k in 1:grid.Nz
        #### Not sure this will always be Δzᵃᵃᶜ. When we have step bathymetry then it may be
        #### Δzᶠᶜᶜ and Δzᶜᶠᶜ. Not entirely sure the notation works perfectly. For the X direction
        #### volume flux we locate the z direction length at the intersection of a YZ plane that is
        #### on an x face and a XZ plane that passes through a y center. For the Y direction
        #### volume flux we locate the z direction lenght at the intersection of the XZ plane 
        #### that is on a y face and the YZ plane that passes through the x center.
        @inbounds barotropic_volume_flux.u[i, j, 1] += U.u[i, j, k]*Δyᶠᶜᵃ(i, j, k, grid)*Δzᵃᵃᶜ(i, j, k, grid)
        @inbounds barotropic_volume_flux.v[i, j, 1] += U.v[i, j, k]*Δxᶜᶠᵃ(i, j, k, grid)*Δzᵃᵃᶜ(i, j, k, grid)
     end
end


"""
Compute volume integrated divergence (need to change name)

 `D = sum.... `
"""
function compute_volume_scaled_divergence!(free_surface, model)
    event = launch!(model.architecture,
                    model.grid,
                    :xy,
                    _compute_volume_scaled_divergence!,
                    model.grid,
                    free_surface.barotropic_volume_flux.u,
                    free_surface.barotropic_volume_flux.v,
                    free_surface.implicit_step_solver.solver.settings.RHS,
                    dependencies=Event(device(model.architecture)))
    return event
end
@kernel function _compute_volume_scaled_divergence!(grid, ut, vt, div)
    # Here we use a integral form that has been multiplied through by volumes to be 
    # consistent with the symmetric "A" matrix.
    # The quantities differenced here are transports i.e. normal velocity vectors
    # integrated over an area.
    #
    i, j = @index(Global, NTuple)
    @inbounds div[i, j, 1] = δxᶜᵃᵃ(i, j, 1, grid, ut) + δyᵃᶜᵃ(i, j, 1, grid, vt)
end

function add_previous_free_surface_contribution(free_surface, model, Δt )
   g = model.free_surface.gravitational_acceleration
   event = launch!(model.architecture,
                   model.grid,
                   :xy,
                   _add_previous_free_surface_contribution!,
                   model.grid,
                   free_surface.η.data,
                   free_surface.implicit_step_solver.solver.settings.RHS,
                   g,
                   Δt,
                   dependencies=Event(device(model.architecture)))
    return event
end
@kernel function _add_previous_free_surface_contribution!(grid,η,RHS,g,Δt)
    i, j = @index(Global, NTuple)
    @inbounds RHS[i,j,1] -= Azᶜᶜᵃ(i, j, 1, grid)*η[i,j, 1]/(g*Δt^2)
end

