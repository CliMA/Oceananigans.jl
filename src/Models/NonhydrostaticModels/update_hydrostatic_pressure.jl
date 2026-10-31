using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ
using Oceananigans.ImmersedBoundaries: PartialCellVerticalBoundaries, ImmersedBoundaryGrid
using Oceananigans.Grids: topology
using Oceananigans.Fields: instantiated_location
using Oceananigans.BoundaryConditions: _fill_bottom_and_top_halo!

"""
Update the hydrostatic pressure perturbation pHY′. This is done by integrating
the `buoyancy_perturbationᶜᶜᶜ` downwards:

    `pHY′ = ∫ buoyancy_perturbationᶜᶜᶜ dz` from `z=0` down to `z=-Lz`
"""
@kernel function _update_hydrostatic_pressure!(pHY′, grid, buoyancy, C)
    i, j = @index(Global, NTuple)

    pᵏ = - z_dot_g_bᶜᶜᶠ(i, j, grid.Nz+1, grid, buoyancy, C) * Δzᶜᶜᶠ(i, j, grid.Nz+1, grid)
    @inbounds pHY′[i, j, grid.Nz] = pᵏ

    for k in grid.Nz-1 : -1 : 1
        pᵏ -= z_dot_g_bᶜᶜᶠ(i, j, k+1, grid, buoyancy, C) * Δzᶜᶜᶠ(i, j, k+1, grid)
        @inbounds pHY′[i, j, k] = pᵏ
    end
end

update_hydrostatic_pressure!(model; kwargs...) = update_hydrostatic_pressure!(model.grid, model; kwargs...)
update_hydrostatic_pressure!(::AbstractGrid{<:Any, <:Any, <:Any, <:Flat}, model; kwargs...) = nothing
update_hydrostatic_pressure!(grid, model; kwargs...) =
    update_hydrostatic_pressure!(model.pressures.pHY′, model.architecture, model.grid, model.buoyancy, model.tracers; kwargs...)

# Partial cell "algorithm"
const PCB = PartialCellVerticalBoundaries
const PCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:PCB}

update_hydrostatic_pressure!(pHY′, arch, ibg::PCBIBG, buoyancy, tracers; parameters = surface_kernel_parameters(ibg.underlying_grid)) =
    update_hydrostatic_pressure!(pHY′, arch, ibg.underlying_grid, buoyancy, tracers; parameters)

# The vertical halos are filled over the same columns, which include the horizontal halos
function update_hydrostatic_pressure!(pHY′, arch, grid, buoyancy, tracers; parameters = surface_kernel_parameters(grid))
    launch!(arch, grid, parameters, _update_hydrostatic_pressure!, pHY′, grid, buoyancy, tracers)

    bcs = pHY′.boundary_conditions
    launch!(arch, grid, parameters, _fill_bottom_and_top_halo!, pHY′, bcs.bottom, bcs.top, instantiated_location(pHY′), grid, ())

    return nothing
end

update_hydrostatic_pressure!(::Nothing, arch, grid, args...; kw...) = nothing
update_hydrostatic_pressure!(::Nothing, arch, ::PCBIBG, args...; kw...) = nothing
