using Oceananigans.Utils: cell_advection_timescale
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

using CUDA, CUDAKernels, KernelAbstractions, Tullio

using Oceananigans.Models
using Oceananigans.Grids: halo_size
using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶠ

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
        @tullio (min) timescale[k] := 1 / (  abs(u[i, j, k]) / Δxᶠᶜᶜ(i, j, k, grid)
                                           + abs(v[i, j, k]) / Δyᶜᶠᶜ(i, j, k, grid)
                                           + abs(w[i, j, k]) / Δzᶜᶜᶠ(i, j, k, grid))
    )

    return min_timescale
end
