using KernelAbstractions: @kernel, @index

using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.BuoyancyFormulations: materialize_buoyancy
using Oceananigans.Fields: CenterField, Field, set!
using Oceananigans.Grids: Center, inactive_node, rnode, topology, c, f
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, CavityIBG, CavityLoad, mask_immersed_field!

import Oceananigans.ImmersedBoundaries: materialize_ice_load
using Oceananigans.Utils: launch!

using .NonhydrostaticModels: integrate_cavity_hydrostatic_pressure!, topmost_active_index

"""
$(TYPEDSIGNATURES)

Return a two-dimensional `Field{Center, Center, Nothing}` containing the static ice-load
potential `Φ = pʳ - pᵐ` for a grid with a [`GridFittedCavity`](@ref) or
[`PartialCellCavity`](@ref) immersed boundary, evaluated at the topmost wet cell of each column.

Here `pʳ` is the hydrostatic pressure anomaly of `reference_tracers` integrated from the top of
the domain as if there were no ice, and `pᵐ` is the hydrostatic pressure anomaly the model
computes for the same tracers, which excludes the ice-covered levels. `Φ` is the weight of the
ice expressed as the water it displaces, and vanishes in columns without ice or without wet cells.

Passing `Φ` as the `ice_load` of the cavity adds it to the hydrostatic pressure of every column,
which keeps a cavity whose stratification equals `reference_tracers` at rest.
[`CavityLoad`](@ref) does this when the grid is built.

Arguments
=========

- `grid`: an `ImmersedBoundaryGrid` with a `GridFittedCavity` or `PartialCellCavity`.
- `buoyancy`: the model's buoyancy formulation, e.g. `BuoyancyTracer()` or `SeawaterBuoyancy()`.
- `reference_tracers`: a `NamedTuple` of the tracers required by `buoyancy`, whose values
  may be anything accepted by `set!`. Typically the initial condition.

Example
=======

```jldoctest
julia> using Oceananigans

julia> underlying_grid = RectilinearGrid(size=(1, 4, 10), x=(0, 1), y=(0, 4), z=(-100, 0), topology=(Periodic, Bounded, Bounded));

julia> ceiling_height(x, y) = y < 2 ? -40 : 0;

julia> grid = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(-100, ceiling_height));

julia> Φ = cavity_load_potential(grid, BuoyancyTracer(), (; b = (x, y, z) -> 1e-5 * z));

julia> Array(interior(Φ))[1, :, 1]
4-element Vector{Float64}:
 0.008
 0.008
 0.0
 0.0

julia> grid = ImmersedBoundaryGrid(underlying_grid, GridFittedCavity(-100, ceiling_height; ice_load=Φ));

julia> grid.immersed_boundary.ice_load[1, 1:4, 1]
4-element Vector{Float64}:
 0.008
 0.008
 0.0
 0.0
```
"""
function cavity_load_potential(grid::CavityIBG, buoyancy, reference_tracers::NamedTuple)
    buoyancy = materialize_buoyancy(buoyancy, grid)
    arch = architecture(grid)
    names = keys(reference_tracers)
    underlying_grid = grid.underlying_grid

    ice_free_tracers = NamedTuple{names}(ntuple(n -> CenterField(underlying_grid), length(names)))
    masked_tracers   = NamedTuple{names}(ntuple(n -> CenterField(grid), length(names)))

    for name in names
        set!(ice_free_tracers[name], reference_tracers[name])
        fill_halo_regions!(ice_free_tracers[name])

        set!(masked_tracers[name], reference_tracers[name])
        mask_immersed_field!(masked_tracers[name])
        fill_halo_regions!(masked_tracers[name])
    end

    ice_free_pressure = CenterField(underlying_grid)
    masked_pressure   = CenterField(grid)
    launch!(arch, grid, :xy, _compute_ice_free_hydrostatic_pressure!, ice_free_pressure, grid, buoyancy, ice_free_tracers)
    launch!(arch, grid, :xy, _compute_unloaded_hydrostatic_pressure!, masked_pressure, grid, buoyancy, masked_tracers)

    Φ = Field{Center, Center, Nothing}(grid)
    launch!(arch, grid, :xy, _compute_cavity_load_potential!, Φ, grid, ice_free_pressure, masked_pressure)
    fill_halo_regions!(Φ)

    return Φ
end

function materialize_ice_load(grid, unloaded_ib, load::CavityLoad)
    TX, TY, TZ = topology(grid)
    unloaded_grid = ImmersedBoundaryGrid{TX, TY, TZ}(grid, unloaded_ib, nothing, nothing)
    Φ = cavity_load_potential(unloaded_grid, load.buoyancy, load.reference_tracers)
    return Φ.data
end

@kernel function _compute_ice_free_hydrostatic_pressure!(p, grid, buoyancy, C)
    i, j = @index(Global, NTuple)
    rᵗ = rnode(i, j, grid.Nz + 1, grid.underlying_grid, c, c, f)
    integrate_cavity_hydrostatic_pressure!(p, i, j, grid, grid.Nz, rᵗ, zero(grid), buoyancy, C)
end

# The model's pressure without any `ice_load` already attached to `grid`
@kernel function _compute_unloaded_hydrostatic_pressure!(p, grid, buoyancy, C)
    i, j = @index(Global, NTuple)
    kᵗ = topmost_active_index(i, j, grid)
    rᵈ = @inbounds grid.immersed_boundary.ceiling_height[i, j, 1]
    integrate_cavity_hydrostatic_pressure!(p, i, j, grid, kᵗ, rᵈ, zero(grid), buoyancy, C)
end

@kernel function _compute_cavity_load_potential!(Φ, grid, pʳ, pᵐ)
    i, j = @index(Global, NTuple)
    Φᵢⱼ = zero(grid)
    for k in 1:grid.Nz
        wet = !inactive_node(i, j, k, grid, c, c, c)
        Φᵢⱼ = @inbounds ifelse(wet, pʳ[i, j, k] - pᵐ[i, j, k], Φᵢⱼ)
    end
    @inbounds Φ[i, j, 1] = Φᵢⱼ
end
