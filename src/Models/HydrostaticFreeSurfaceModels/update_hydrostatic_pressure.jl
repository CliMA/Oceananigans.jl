using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ
using Oceananigans.ImmersedBoundaries: PartialCellBottom, ImmersedBoundaryGrid

"""
Update the hydrostatic pressure perturbation pHY′. This is done by integrating
the `buoyancy_perturbationᶜᶜᶜ` downwards:

    `pHY′ = ∫ buoyancy_perturbationᶜᶜᶜ dz` from `z=0` down to `z=-Lz`
"""
@kernel function _update_hydrostatic_pressure!(pHY′, grid, buoyancy, C)
    i, j = @index(Global, NTuple)

    @inbounds pHY′[i, j, grid.Nz] = - z_dot_g_bᶜᶜᶠ(i, j, grid.Nz+1, grid, buoyancy, C) * Δzᶜᶜᶠ(i, j, grid.Nz+1, grid)

    @unroll for k in grid.Nz-1 : -1 : 1
        @inbounds pHY′[i, j, k] = pHY′[i, j, k+1] - z_dot_g_bᶜᶜᶠ(i, j, k+1, grid, buoyancy, C) * Δzᶜᶜᶠ(i, j, k+1, grid)
    end
end

update_hydrostatic_pressure!(model) = update_hydrostatic_pressure!(model.grid, model)
update_hydrostatic_pressure!(::AbstractGrid{<:Any, <:Any, <:Any, <:Flat}, model) = nothing
update_hydrostatic_pressure!(grid, model) = update_hydrostatic_pressure!(model.pressures.pHY′, model.architecture, model.grid, model.buoyancy, model.tracers)

# Partial cell "algorithm"
const PCB = PartialCellBottom
const PCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:PCB}

update_hydrostatic_pressure!(pHY′, arch, ibg::PCBIBG, buoyancy, tracers) =
    update_hydrostatic_pressure!(pHY′, arch, ibg.underlying_grid, buoyancy, tracers)

update_hydrostatic_pressure!(pHY′, arch, grid, buoyancy, tracers) =  
    launch!(arch, grid, :xy, _update_hydrostatic_pressure!, pHY′, grid, buoyancy, tracers)
