using Oceananigans.Grids: AbstractGrid, Flat
using Oceananigans.Fields: Center
using Oceananigans.AbstractOperations: MultiaryOperation, Δx, Δy, Δz
using Oceananigans.Operators: identity1
using Oceananigans.TurbulenceClosures: cell_diffusion_timescale
using Oceananigans.Models: AbstractModel
using Oceananigans.Models.ShallowWaterModels: ShallowWaterModel
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

"""
    CFL{D, S}

An object for computing the Courant-Freidrichs-Lewy (CFL) number.
"""
struct CFL{D, S}
    Δt :: D
    timescale :: S
end

"""
    CFL(Δt [, timescale=Oceananigans.cell_advection_timescale])

Returns an object for computing the Courant-Freidrichs-Lewy (CFL) number
associated with time step or `TimeStepWizard` `Δt` and `timescale`.

See also `AdvectiveCFL` and `DiffusiveCFL`.
"""
CFL(Δt) = CFL(Δt, cell_advection_timescale)

(c::CFL)(model) = c.Δt / c.timescale(model)

"""
    AdvectiveCFL(Δt)

Returns an object for computing the Courant-Freidrichs-Lewy (CFL) number
associated with time step or `TimeStepWizard` `Δt` and the time scale
for advection across a cell.

Example
=======
```julia
julia> model = NonhydrostaticModel(grid=RectilinearGrid(size=(16, 16, 16), length=(8, 8, 8)));

julia> cfl = AdvectiveCFL(1.0);

julia> data(model.velocities.u) .= π;

julia> cfl(model)
6.283185307179586
```
"""
AdvectiveCFL(Δt) = CFL(Δt, cell_advection_timescale)

"""
    DiffusiveCFL(Δt)

Returns an object for computing the diffusive Courant-Freidrichs-Lewy (CFL) number
associated with time step or `TimeStepWizard` `Δt` and the time scale for diffusion
across a cell associated with `model.closure`.

The maximum diffusive CFL number among viscosity and all tracer diffusivities is
returned.

Example
=======
```julia
julia> model = NonhydrostaticModel(grid=RectilinearGrid(size=(16, 16, 16), length=(1, 1, 1)));

julia> dcfl = DiffusiveCFL(0.1);

julia> dcfl(model)
2.688e-5
```
"""
DiffusiveCFL(Δt) = CFL(Δt, cell_diffusion_timescale)

#####
##### Cell advection time-scale calculation
#####

function cell_advection_timescale(grid, velocities)
    u, v, w = velocities 
    terms = (abs(u) / Δx, abs(v) / Δy, abs(w) / Δz)
    # Manually construct non-interpolating
    #
    #       net_frequency = abs(u) / Δx + abs(v) / Δy + abs(w) / Δz
    #
    net_frequency = MultiaryOperation{Center, Center, Center}(+, terms, identity1, grid)
    min_timescale = 1 / maximum(net_frequency)
    return min_timescale
end

#####
##### Various translations
#####

cell_advection_timescale(model::AbstractModel) = cell_advection_timescale(model.grid, model.velocities)
cell_advection_timescale(ibg::ImmersedBoundaryGrid, velocities) = cell_advection_timescale(ibg.grid, velocities)

function cell_advection_timescale(model::ShallowWaterModel)
    uh, vh, h = model.solution
    u = uh / h
    v = vh / h
    w = 0
    return cell_advection_timescale(model.grid, (; u, v, w))
end

#####
##### Flattened grids (just treating 2D cases for now...)
#####

const XFlatGrid = AbstractGrid{FT, Flat} where FT
const YFlatGrid = AbstractGrid{FT, TX, Flat} where {FT, TX}
const ZFlatGrid = AbstractGrid{FT, TX, TY, Flat} where {FT, TX, TY}

cell_advection_timescale(grid::XFlatGrid, U) = cell_advection_timescale(grid, (; u=0, v=U.v, w=U.w))
cell_advection_timescale(grid::YFlatGrid, U) = cell_advection_timescale(grid, (; u=U.u, v=0, w=U.w))
cell_advection_timescale(grid::ZFlatGrid, U) = cell_advection_timescale(grid, (; u=U.u, v=U.v, w=0))

