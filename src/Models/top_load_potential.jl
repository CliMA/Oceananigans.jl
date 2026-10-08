using KernelAbstractions: @kernel, @index

using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.BuoyancyFormulations: materialize_buoyancy
using Oceananigans.Fields: CenterField, set!
using Oceananigans.Grids: Center, inactive_node, rnode, topology, c, f
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, ImmersedTopIBG, TopLoad, mask_immersed_field!
using Oceananigans.Utils: launch!

using .NonhydrostaticModels: integrate_immersed_top_hydrostatic_pressure!, topmost_active_index

import Oceananigans.ImmersedBoundaries: materialize_top_load

"""
$(TYPEDSIGNATURES)

Return a two-dimensional `Field{Center, Center, Nothing}` containing the static top-load
potential `Φ = pʳ - pᵐ` for a grid with an immersed top, evaluated at the topmost wet cell of each column.

Here `pʳ` is the hydrostatic pressure anomaly of `reference_tracers` integrated from the top of
the domain as if there were no immersed top, and `pᵐ` is the hydrostatic pressure anomaly the model
computes for the same tracers, which excludes the top-covered levels. `Φ` is the weight of the
solid top expressed as the water it displaces, and vanishes in open columns or columns without wet cells.

Passing `Φ` as the `top_load` of the immersed boundary adds it to the hydrostatic pressure of every column,
which keeps a fluid whose stratification equals `reference_tracers` at rest.
[`TopLoad`](@ref) does this when the grid is built.

Arguments
=========

- `grid`: an `ImmersedBoundaryGrid` with a `GridFittedBottom` or `PartialCellBottom` that has a `top_height`.
- `buoyancy`: the model's buoyancy formulation, e.g. `BuoyancyTracer()` or `SeawaterBuoyancy()`.
- `reference_tracers`: a `NamedTuple` of the tracers required by `buoyancy`, whose values
  may be anything accepted by `set!`. Typically the initial condition.

Example
=======

```jldoctest
julia> using Oceananigans

julia> underlying_grid = RectilinearGrid(size=(1, 4, 10), x=(0, 1), y=(0, 4), z=(-100, 0), topology=(Periodic, Bounded, Bounded));

julia> top_height(x, y) = y < 2 ? -40 : 0;

julia> grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(-100; top_height));

julia> Φ = top_load_potential(grid, BuoyancyTracer(), (; b = (x, y, z) -> 1e-5 * z));

julia> Φ[1, 1:4, 1]
4-element Vector{Float64}:
 0.008
 0.008
 0.0
 0.0
```
"""
function top_load_potential(grid::ImmersedTopIBG, buoyancy, reference_tracers::NamedTuple)
    buoyancy = materialize_buoyancy(buoyancy, grid)
    arch = architecture(grid)
    names = keys(reference_tracers)
    underlying_grid = grid.underlying_grid

    reference_tracer_fields = NamedTuple{names}(ntuple(n -> CenterField(underlying_grid), length(names)))
    masked_tracers          = NamedTuple{names}(ntuple(n -> CenterField(grid), length(names)))

    for name in names
        set!(reference_tracer_fields[name], reference_tracers[name])
        fill_halo_regions!(reference_tracer_fields[name])

        set!(masked_tracers[name], reference_tracers[name])
        mask_immersed_field!(masked_tracers[name])
        fill_halo_regions!(masked_tracers[name])
    end

    reference_pressure = CenterField(underlying_grid)
    masked_pressure    = CenterField(grid)
    launch!(arch, grid, :xy, _compute_reference_hydrostatic_pressure!, reference_pressure, grid, buoyancy, reference_tracer_fields)
    launch!(arch, grid, :xy, _compute_unloaded_hydrostatic_pressure!, masked_pressure, grid, buoyancy, masked_tracers)

    Φ = Field{Center, Center, Nothing}(grid)
    launch!(arch, grid, :xy, _compute_top_load_potential!, Φ, grid, reference_pressure, masked_pressure)
    fill_halo_regions!(Φ)

    return Φ
end

function materialize_top_load(grid, unloaded_ib, load::TopLoad)
    TX, TY, TZ = topology(grid)
    unloaded_grid = ImmersedBoundaryGrid{TX, TY, TZ}(grid, unloaded_ib, nothing, nothing)
    Φ = top_load_potential(unloaded_grid, load.buoyancy, load.reference_tracers)
    return Φ.data
end

@kernel function _compute_reference_hydrostatic_pressure!(p, grid, buoyancy, C)
    i, j = @index(Global, NTuple)
    rᵗ = rnode(i, j, grid.Nz + 1, grid.underlying_grid, c, c, f)
    integrate_immersed_top_hydrostatic_pressure!(p, i, j, grid, grid.Nz, rᵗ, zero(grid), buoyancy, C)
end

# The model's pressure without any `top_load` already attached to `grid`
@kernel function _compute_unloaded_hydrostatic_pressure!(p, grid, buoyancy, C)
    i, j = @index(Global, NTuple)
    kᵗ = topmost_active_index(i, j, grid)
    rᵗ = @inbounds grid.immersed_boundary.top_height[i, j, 1]
    integrate_immersed_top_hydrostatic_pressure!(p, i, j, grid, kᵗ, rᵗ, zero(grid), buoyancy, C)
end

@kernel function _compute_top_load_potential!(Φ, grid, pʳ, pᵐ)
    i, j = @index(Global, NTuple)
    Φᵢⱼ = zero(grid)
    for k in 1:grid.Nz
        wet = !inactive_node(i, j, k, grid, c, c, c)
        Φᵢⱼ = @inbounds ifelse(wet, pʳ[i, j, k] - pᵐ[i, j, k], Φᵢⱼ)
    end
    @inbounds Φ[i, j, 1] = Φᵢⱼ
end
