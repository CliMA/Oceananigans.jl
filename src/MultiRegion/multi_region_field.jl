using Oceananigans.BoundaryConditions: default_auxiliary_bc
using Oceananigans.Fields: FunctionField, data_summary
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Operators: assumed_field_location
using Oceananigans.OutputWriters: output_indices

import Oceananigans.Fields: set!, compute!, compute_at!, validate_field_data, validate_boundary_conditions
import Oceananigans.Fields: validate_indices, FieldBoundaryBuffers
import Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions
import Base: fill!, axes
import Oceananigans.Simulations: hasnan

# Field and FunctionField (both fields with "grids attached")
const MultiRegionField{LX, LY, LZ, O} = Field{LX, LY, LZ, O, <:MultiRegionGrid} where {LX, LY, LZ, O}
const MultiRegionComputedField{LX, LY, LZ, O} = Field{LX, LY, LZ, <:AbstractOperation, <:MultiRegionGrid} where {LX, LY, LZ}
const MultiRegionFunctionField{LX, LY, LZ, C, P, F} = FunctionField{LX, LY, LZ, C, P, F, <:MultiRegionGrid} where {LX, LY, LZ, C, P, F}

const GriddedMultiRegionField = Union{MultiRegionField, MultiRegionFunctionField}
const GriddedMultiRegionFieldTuple{N, T} = NTuple{N, T} where {N, T<:GriddedMultiRegionField}
const GriddedMultiRegionFieldNamedTuple{S, N} = NamedTuple{S, N} where {S, N<:GriddedMultiRegionFieldTuple}

# Utils
Base.size(f::GriddedMultiRegionField) = size(getregion(f.grid, 1))

@inline isregional(f::GriddedMultiRegionField) = true
@inline devices(f::GriddedMultiRegionField)    = devices(f.grid)
@inline sync_all_devices!(f::GriddedMultiRegionField)  = sync_all_devices!(devices(f.grid))

@inline switch_device!(f::GriddedMultiRegionField, d) = switch_device!(f.grid, d)
@inline getdevice(f::GriddedMultiRegionField, d)      = getdevice(f.grid, d)

@inline getregion(f::MultiRegionFunctionField{LX, LY, LZ}, r) where {LX, LY, LZ} =
    FunctionField{LX, LY, LZ}(_getregion(f.func, r),
                              _getregion(f.grid, r),
                              clock = _getregion(f.clock, r),
                              parameters = _getregion(f.parameters, r))

@inline getregion(f::MultiRegionField{LX, LY, LZ}, r) where {LX, LY, LZ} =
    Field{LX, LY, LZ}(_getregion(f.grid, r),
                      _getregion(f.data, r),
                      _getregion(f.boundary_conditions, r),
                      _getregion(f.indices, r),
                      _getregion(f.operand, r),
                      _getregion(f.status, r),
                      _getregion(f.boundary_buffers, r))

@inline _getregion(f::MultiRegionFunctionField{LX, LY, LZ}, r) where {LX, LY, LZ} =
    FunctionField{LX, LY, LZ}(getregion(f.func, r),
                              getregion(f.grid, r),
                              clock = getregion(f.clock, r),
                              parameters = getregion(f.parameters, r))

@inline _getregion(f::MultiRegionField{LX, LY, LZ}, r) where {LX, LY, LZ} =
    Field{LX, LY, LZ}(getregion(f.grid, r),
                      getregion(f.data, r),
                      getregion(f.boundary_conditions, r),
                      getregion(f.indices, r),
                      getregion(f.operand, r),
                      getregion(f.status, r),
                      getregion(f.boundary_buffers, r))

"""
    reconstruct_global_field(mrf)

Reconstruct a global field from `mrf::MultiRegionField` on the `CPU`.
"""
function reconstruct_global_field(mrf::MultiRegionField)
    global_grid  = on_architecture(CPU(), reconstruct_global_grid(mrf.grid))
    indices      = reconstruct_global_indices(mrf.indices, mrf.grid.partition, size(global_grid))
    global_field = Field(location(mrf), global_grid; indices)

    data = construct_regionally(interior, mrf)
    data = construct_regionally(Array, data)
    compact_data!(global_field, global_grid, data, mrf.grid.partition)
    
    fill_halo_regions!(global_field)
    return global_field
end

# Fallback!
@inline reconstruct_global_field(f::AbstractField) = f

function reconstruct_global_indices(indices, p::XPartition, N)
    idx1 = getregion(indices, 1)[1]
    idxl = getregion(indices, length(p))[1]

    if idx1 == Colon() && idxl == Colon()
        idx_x = Colon()
    else
        idx_x = UnitRange(idx1 == Colon() ? 1 : first(idx1), idxl == Colon() ? N[1] : last(idxl))
    end

    idx_y = getregion(indices, 1)[2]
    idx_z = getregion(indices, 1)[3]

    return (idx_x, idx_y, idx_z)
end

function reconstruct_global_indices(indices, p::YPartition, N)
    idx1 = getregion(indices, 1)[2]
    idxl = getregion(indices, length(p))[2]

    if idx1 == Colon() && idxl == Colon()
        idx_y = Colon()
    else
        idx_y = UnitRange(ix1 == Colon() ? 1 : first(idx1), idxl == Colon() ? N[2] : last(idxl))
    end

    idx_x = getregion(indices, 1)[1]
    idx_z = getregion(indices, 1)[3]

    return (idx_x, idx_y, idx_z)
end

## Functions applied regionally
set!(mrf::MultiRegionField, v)  = apply_regionally!(set!,  mrf, v)
fill!(mrf::MultiRegionField, v) = apply_regionally!(fill!, mrf, v)

set!(mrf::MultiRegionField, f::Function)  = apply_regionally!(set!, mrf, f)

compute_at!(mrf::GriddedMultiRegionField, time)  = apply_regionally!(compute_at!, mrf, time)
compute_at!(mrf::MultiRegionComputedField, time) = apply_regionally!(compute_at!, mrf, time)

@inline hasnan(field::MultiRegionField) = (&)(construct_regionally(hasnan, field).regional_objects...)

validate_indices(indices, loc, mrg::MultiRegionGrid) = 
    construct_regionally(validate_indices, indices, loc, mrg.region_grids)

FieldBoundaryBuffers(grid::MultiRegionGrid, args...; kwargs...) = 
    construct_regionally(FieldBoundaryBuffers, grid, args...; kwargs...)

FieldBoundaryConditions(mrg::MultiRegionGrid, loc, indices; kwargs...) =
  construct_regionally(inject_regional_bcs, mrg, Iterate(1:length(mrg)), Reference(mrg.partition), Reference(loc), indices; kwargs...)

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                              mrg::MultiRegionGrid,
                                              field_name::Symbol,
                                              prognostic_field_name=nothing)

  reg_bcs = regularize_field_boundary_conditions(bcs, mrg.region_grids[1], field_name, prognostic_field_name)
  loc = assumed_field_location(field_name)

  return FieldBoundaryConditions(mrg, loc; west = reg_bcs.west,
                                           east = reg_bcs.east,
                                           south = reg_bcs.south,
                                           north = reg_bcs.north,
                                           bottom = reg_bcs.bottom,
                                           top = reg_bcs.top,
                                           immersed = reg_bcs.immersed)
end

function inject_regional_bcs(grid, region, partition, loc, indices;   
                              west = default_auxiliary_bc(topology(grid, 1)(), loc[1]()),
                              east = default_auxiliary_bc(topology(grid, 1)(), loc[1]()),
                             south = default_auxiliary_bc(topology(grid, 2)(), loc[2]()),
                             north = default_auxiliary_bc(topology(grid, 2)(), loc[2]()),
                            bottom = default_auxiliary_bc(topology(grid, 3)(), loc[3]()),
                               top = default_auxiliary_bc(topology(grid, 3)(), loc[3]()),
                          immersed = NoFluxBoundaryCondition())


  west  = inject_west_boundary(region, partition, west)
  east  = inject_east_boundary(region, partition, east)
  south = inject_south_boundary(region, partition, south)
  north = inject_north_boundary(region, partition, north)
  return FieldBoundaryConditions(indices, west, east, south, north, bottom, top, immersed)
end

function Base.show(io::IO, field::MultiRegionField)

  bcs = field.boundary_conditions

  prefix =
      string("$(summary(field))\n",
             "├── grid: ", summary(field.grid), "\n",
             "├── boundary conditions: ", summary(bcs), "\n")
  middle = isnothing(field.operand) ? "" :
      string("├── operand: ", summary(field.operand), "\n",
             "├── status: ", summary(field.status), "\n")

  suffix = string("└── data: ", summary(field.data), "\n",
                  "    └── ", data_summary(field))

  print(io, prefix, middle, suffix)
end


