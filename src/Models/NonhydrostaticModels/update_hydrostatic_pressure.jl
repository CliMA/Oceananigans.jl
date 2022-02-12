using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ

"""
Update the hydrostatic pressure perturbation pHY′. This is done by integrating
the `buoyancy_perturbation` downwards:

    `pHY′ = ∫ buoyancy_perturbation dz` from `z=0` down to `z=-Lz`
"""
@kernel function _update_hydrostatic_pressure!(pHY′, grid, buoyancy, C)
    i, j = @index(Global, NTuple)

    @inbounds pHY′[i, j, grid.Nz] = - ℑzᵃᵃᶠ(i, j, grid.Nz+1, grid, z_dot_g_b, buoyancy, C) * Δzᶜᶜᶠ(i, j, grid.Nz+1, grid)

    @unroll for k in grid.Nz-1 : -1 : 1
        @inbounds pHY′[i, j, k] = pHY′[i, j, k+1] - ℑzᵃᵃᶠ(i, j, k+1, grid, z_dot_g_b, buoyancy, C) * Δzᶜᶜᶠ(i, j, k+1, grid)
    end
end

update_hydrostatic_pressure!(model) = update_hydrostatic_pressure!(model.grid, model)

update_hydrostatic_pressure!(::AbstractGrid{<:Any, <:Any, <:Any, <:Flat}, model) = nothing

update_hydrostatic_pressure!(grid, model) = update_hydrostatic_pressure!(model.pressures.pHY′, model.architecture, model.grid, model.buoyancy, model.tracers)

function update_hydrostatic_pressure!(pHY′, arch, grid, buoyancy, tracers)
    pressure_calculation = launch!(arch, grid, :xy, _update_hydrostatic_pressure!,
                                   pHY′, grid, buoyancy, tracers,
                                   dependencies = Event(device(arch)))

    # Fill halo regions for pressure
    wait(device(arch), pressure_calculation)

    fill_halo_regions!(pHY′, arch)
    return nothing
end
