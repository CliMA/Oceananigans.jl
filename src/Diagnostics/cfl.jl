using Oceananigans.Utils: cell_advection_timescale
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
julia> model = NonhydrostaticModel(grid=RegularRectilinearGrid(size=(16, 16, 16), length=(8, 8, 8)));

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
julia> model = NonhydrostaticModel(grid=RegularRectilinearGrid(size=(16, 16, 16), length=(1, 1, 1)));

julia> dcfl = DiffusiveCFL(0.1);

julia> dcfl(model)
2.688e-5
```
"""
DiffusiveCFL(Δt) = CFL(Δt, cell_diffusion_timescale)

#####
##### Accurate CFL via reduction
#####

using CUDA, CUDAKernels, KernelAbstractions, Tullio

using Oceananigans.Models
using Oceananigans.Grids: halo_size
using Oceananigans.Operators: Δxᶠᶜᵃ, Δyᶜᶠᵃ, Δzᵃᵃᶠ

accurate_cell_advection_timescale(model) = accurate_cell_advection_timescale(model.grid, model.velocities)

function accurate_cell_advection_timescale(grid, velocities)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    is = 1+Hx:Nx+Hx
    js = 1+Hy:Ny+Hy
    ks = 1+Hz:Nz+Hz

    u = view(velocities.u.data.parent, is, js, ks)
    v = view(velocities.v.data.parent, is, js, ks)
    w = view(velocities.w.data.parent, is, js, ks)

    min_timescale = minimum(
        @tullio (min) timescale[k] := 1 / (  abs(u[i, j, k]) / Δxᶠᶜᵃ(i, j, k, grid)
                                           + abs(v[i, j, k]) / Δyᶜᶠᵃ(i, j, k, grid)
                                           + abs(w[i, j, k]) / Δzᵃᵃᶠ(i, j, k, grid))
    )

    return min_timescale
end
