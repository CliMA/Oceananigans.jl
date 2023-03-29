using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.TurbulenceClosures: cell_diffusion_timescale

"""
    struct CFL{D, S}

An object for computing the Courant-Freidrichs-Lewy (CFL) number.
"""
struct CFL{D, S}
           Δt :: D
    timescale :: S
end

"""
    CFL(Δt [, timescale = Oceananigans.cell_advection_timescale])

Return an object for computing the Courant-Freidrichs-Lewy (CFL) number
associated with time step `Δt` or `TimeStepWizard` and `timescale`.

See also [`AdvectiveCFL`](@ref Oceananigans.Diagnostics.AdvectiveCFL)
and [`DiffusiveCFL`](Oceananigans.Diagnostics.DiffusiveCFL).
"""
CFL(Δt) = CFL(Δt, cell_advection_timescale)

(c::CFL)(model) = c.Δt / c.timescale(model)

"""
    AdvectiveCFL(Δt)

Return an object for computing the Courant-Freidrichs-Lewy (CFL) number
associated with time step `Δt` or `TimeStepWizard` and the time scale
for advection across a cell. The advective CFL is, e.g., ``U Δt / Δx``.

Example
=======
```jldoctest
julia> using Oceananigans

julia> model = NonhydrostaticModel(grid = RectilinearGrid(size=(16, 16, 16), extent=(8, 8, 8)));

julia> Δt = 1.0;

julia> cfl = AdvectiveCFL(Δt);

julia> model.velocities.u .= π;

julia> cfl(model)
6.283185307179586
```
"""
AdvectiveCFL(Δt) = CFL(Δt, cell_advection_timescale)

"""
    DiffusiveCFL(Δt)

Returns an object for computing the diffusive Courant-Freidrichs-Lewy (CFL) number
associated with time step `Δt` or `TimeStepWizard` and the time scale for diffusion
across a cell associated with `model.closure`.  The diffusive CFL, e.g., for viscosity
is ``ν Δt / Δx²``.

The maximum diffusive CFL number among viscosity and all tracer diffusivities is
returned.

Example
=======
```jldoctest
julia> using Oceananigans

julia> model = NonhydrostaticModel(grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1)),
                                   closure = ScalarDiffusivity(; ν = 1e-2));

julia> Δt = 0.1;

julia> dcfl = DiffusiveCFL(Δt);

julia> dcfl(model)
0.256
```
"""
DiffusiveCFL(Δt) = CFL(Δt, cell_diffusion_timescale)

#####
##### Accurate CFL via reduction
#####

using Oceananigans.AbstractOperations: Δx, Δy, Δz, GridMetricOperation

@inline advection_timescale(u, v, w, dx, dy, dz) = 1 / ((abs(u) / dx + abs(v) / dy + abs(w) / dz))

function accurate_cell_advection_timescale(model)
    
    grid = model.grid
    velocities = model.velocities

    Nx, Ny, Nz = size(grid)

    u = interior(velocities.u, 1:Nx, 1:Ny, 1:Nz)
    v = interior(velocities.v, 1:Nx, 1:Ny, 1:Nz)
    w = interior(velocities.w, 1:Nx, 1:Ny, 1:Nz)

    dx = compute!(Field(GridMetricOperation((Face, Center, Center), Δx, grid)))
    dy = compute!(Field(GridMetricOperation((Center, Face, Center), Δy, grid)))
    dz = compute!(Field(GridMetricOperation((Center, Center, Face), Δz, grid)))

    dx = interior(dx, 1:Nx, 1:Ny, 1:Nz)
    dy = interior(dy, 1:Nx, 1:Ny, 1:Nz)
    dz = interior(dz, 1:Nx, 1:Ny, 1:Nz)

    timescale = mapreduce(advection_timescale, min, u, v, w, dx, dy, dz)

    return timescale
end
