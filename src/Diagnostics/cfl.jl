using Oceananigans: cell_advection_timescale
using Oceananigans.TurbulenceClosures: cell_diffusion_timescale

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

See also `AdvectiveCFL` and `DiffusiveCFL`
"""
CFL(Δt) = CFL(Δt, cell_advection_timescale)

(c::CFL{<:Number})(model) = c.Δt / c.timescale(model)
(c::CFL{<:TimeStepWizard})(model) = c.Δt.Δt / c.timescale(model)

"""
    AdvectiveCFL(Δt)

Returns an object for computing the Courant-Freidrichs-Lewy (CFL) number
associated with time step or `TimeStepWizard` `Δt` and the time scale
for advection across a cell.

Example
=======
```julia
julia> model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(8, 8, 8)));

julia> cfl = AdvectiveCFL(1.0);

julia> data(model.velocities.u) .= π;

julia> cfl(model)
6.283185307179586
```
"""
AdvectiveCFL(Δt) = CFL(Δt, cell_advection_timescale)

"""
    DiffusiveCFL(Δt)

Returns an object for computing the Courant-Freidrichs-Lewy (CFL) number
associated with time step or `TimeStepWizard` `Δt` and the time scale
for diffusion across a cell associated with `model.closure`.

Example
=======
```julia
julia> model = Model(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)));

julia> cfl = DiffusiveCFL(0.1);

julia> cfl(model)
2.688e-5
```
"""
DiffusiveCFL(Δt) = CFL(Δt, cell_diffusion_timescale)
