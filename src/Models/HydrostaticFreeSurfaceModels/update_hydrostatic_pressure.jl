using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ
using Oceananigans.ImmersedBoundaries: PartialCellBottom, ImmersedBoundaryGrid

"""
Update the hydrostatic pressure′. This is done by integrating
the `buoyancy_perturbationᶜᶜᶜ` downwards:

    `pressure = ∫ buoyancy_perturbationᶜᶜᶜ dz` from `z=0` down to `z=-Lz`
"""
@kernel function _update_hydrostatic_pressure!(pressure, grid, buoyancy, C)
    i, j = @index(Global, NTuple)

    @inbounds pressure[i, j, grid.Nz] = - z_dot_g_bᶜᶜᶠ(i, j, grid.Nz+1, grid, buoyancy, C) * Δzᶜᶜᶠ(i, j, grid.Nz+1, grid)

    @unroll for k in grid.Nz-1 : -1 : 1
        @inbounds pressure[i, j, k] = pressure[i, j, k+1] - z_dot_g_bᶜᶜᶠ(i, j, k+1, grid, buoyancy, C) * Δzᶜᶜᶠ(i, j, k+1, grid)
    end
end

update_hydrostatic_pressure!(model) = update_hydrostatic_pressure!(model.grid, model)
update_hydrostatic_pressure!(::AbstractGrid{<:Any, <:Any, <:Any, <:Flat}, model) = nothing
update_hydrostatic_pressure!(grid, model) = update_hydrostatic_pressure!(model.pressure, model.architecture, model.grid, model.buoyancy, model.tracers)

# Partial cell "algorithm"
const PCB = PartialCellBottom
const PCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:PCB}

update_hydrostatic_pressure!(pressure, arch, ibg::PCBIBG, buoyancy, tracers) =
    update_hydrostatic_pressure!(pressure, arch, ibg.underlying_grid, buoyancy, tracers)

update_hydrostatic_pressure!(pressure, arch, grid, buoyancy, tracers) =  
    launch!(arch, grid, :xy, _update_hydrostatic_pressure!, pressure, grid, buoyancy, tracers)
