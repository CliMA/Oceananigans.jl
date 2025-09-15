using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ
using Oceananigans.ImmersedBoundaries: PartialCellBottom, ImmersedBoundaryGrid
using Oceananigans.Grids: topology
using Oceananigans.Grids: XFlatGrid, YFlatGrid

@kernel function _update_hydrostatic_pressure!(pHY′, grid, buoyancy, tracers, coriolis, velocities)
    i, j = @index(Global, NTuple)

    dpdz⁺ = z_dot_g_bᶜᶜᶠ(i, j, grid.Nz+1, grid, buoyancy, tracers) - z_f_cross_U(i, j, grid.Nz+1, grid, coriolis, velocities)
    @inbounds pHY′[i, j, grid.Nz] = - dpdz⁺ * Δzᶜᶜᶠ(i, j, grid.Nz+1, grid)

    for k in grid.Nz-1 : -1 : 1
        dpdz⁺ = z_dot_g_bᶜᶜᶠ(i, j, k+1, grid, buoyancy, tracers) - z_f_cross_U(i, j, k+1, grid, coriolis, velocities)

        # Using dpdz = (p⁺ - pᵏ) / Δzᶜᶜᶠ
        @inbounds pHY′[i, j, k] = pHY′[i, j, k+1] - dpdz⁺ * Δzᶜᶜᶠ(i, j, k+1, grid)
    end
end

"""
    update_hydrostatic_pressure!(pHY′, grid, buoyancy, tracers, coriolis, velocities; parameters=:xy)

Update the hydrostatic pressure perturbation pHY′. This is done by integrating
the `buoyancy_perturbationᶜᶜᶜ` downwards:

```math
pHY′ = ∫ [ b - ẑ ⋅ (f × u) ] dz′
```

from ``z′=0`` down to ``z′=z``.
"""
function update_hydrostatic_pressure!(pHY′, grid, buoyancy, tracers,
                                      coriolis=nothing, velocities=nothing; parameters=:xy)
    arch = grid.architecture
    launch!(arch, grid, parameters,
            _update_hydrostatic_pressure!, pHY′, grid, buoyancy, tracers, coriolis, velocities)
    return nothing
end

# Catch some special cases
update_hydrostatic_pressure!(pHY′, ::AbstractGrid{<:Any, <:Any, <:Any, <:Flat}, args...; kwargs...) = nothing

# Partial cell "algorithm"
const PCB = PartialCellBottom
const PCBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:PCB}

update_hydrostatic_pressure!(pHY′, ibg::PCBIBG, args...; kwargs...) =
    update_hydrostatic_pressure!(pHY′, ibg.underlying_grid, args...; kwargs...)

update_hydrostatic_pressure!(::Nothing, grid, args...; kw...) = nothing
update_hydrostatic_pressure!(::Nothing, ::PCBIBG, args...; kw...) = nothing

# extend p kernel to compute also the boundaries
@inline function p_kernel_parameters(grid)
    Nx, Ny, _ = size(grid)
    TX, TY, _ = topology(grid)

    ii = ifelse(TX == Flat, 1:Nx, 0:Nx+1)
    jj = ifelse(TY == Flat, 1:Ny, 0:Ny+1)

    return KernelParameters(ii, jj)
end

