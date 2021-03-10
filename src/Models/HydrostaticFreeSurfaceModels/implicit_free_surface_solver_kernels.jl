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


function ∇²_baro_operator( HAx, HAy )
    ## Some of this should probably end up in some operator and grid generic auxilliaries
    Ax_baro = HAx
    Ay_baro = HAy
    @inline Ax_∂xᶠᶜᵃ_baro(i, j, k, grid, c) = Ax_baro[i, j, 1] * ∂xᶠᶜᵃ(i, j, 1, grid, c)
    @inline Ay_∂yᶜᶠᵃ_baro(i, j, k, grid, c) = Ay_baro[i, j, 1] * ∂yᶜᶠᵃ(i, j, 1, grid, c)
    @inline function ∇²_baro(i, j, k, grid, c)
       return  δxᶜᵃᵃ(i, j, 1, grid, Ax_∂xᶠᶜᵃ_baro, c) +
               δyᵃᶜᵃ(i, j, 1, grid, Ay_∂yᶜᶠᵃ_baro, c)
    end
    return  ∇²_baro
end

@kernel function implicit_η!(∇²_baro, Δt, g, grid, f, implicit_η_f)
        i, j = @index(Global, NTuple)
        ### Not sure what to call this function
        ### it is for left hand side operator in
        ### ( ∇ʰ⋅H∇ʰ - 1/gΔt² ) ηⁿ⁺¹ = 1/(gΔt) ∇ʰH U̅ˢᵗᵃʳ - 1/(gΔt²) ηⁿ
        ### written in a discrete finite volume form in which the equation
        ### is arranged to ensure a symmtric form
        ### e.g.
        ### 
        ### δⁱÂʷ∂ˣηⁿ⁺¹ + δʲÂˢ∂ʸηⁿ⁺¹ - 1/gΔt² Aᶻηⁿ⁺¹ =
        ###  1/(gΔt)(δⁱÂʷu̅ˢᵗᵃʳ + δʲÂˢv̅ˢᵗᵃʳ) - 1/gΔt² Aᶻηⁿ
        ###
        ### where  ̂ indicates a vertical integral, and
        ###        ̅ indicates a vertical average
        ###
        i, j = @index(Global, NTuple)
        @inbounds implicit_η_f[i, j, 1] =  ∇²_baro(i, j, 1, grid, f) - Azᶜᶜᵃ(i, j, 1, grid)*f[i,j, 1]/(g*Δt^2)
end


