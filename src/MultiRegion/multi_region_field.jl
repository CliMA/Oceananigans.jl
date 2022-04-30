using Oceananigans.BoundaryConditions: default_auxiliary_bc
using Oceananigans.Fields: FunctionField, data_summary
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Operators: assumed_field_location
using Oceananigans.OutputWriters: output_indices

import Oceananigans.Fields: set!, compute!, compute_at!, validate_field_data, validate_boundary_conditions
import Oceananigans.Fields: validate_indices, FieldBoundaryBuffers
import Oceananigans.BoundaryConditions: FieldBoundaryConditions, regularize_field_boundary_conditions
import Oceananigans.Grids: new_data
import Base: fill!
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
sync_all_devices!(f::GriddedMultiRegionField)  = sync_all_devices!(devices(f.grid))

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

@inline reconstruct_global_field(f::AbstractField) = f

function reconstruct_global_field(mrf::MultiRegionField)
    global_grid  = on_architecture(CPU(), reconstruct_global_grid(mrf.grid))
    global_field = Field(location(mrf), global_grid)
  
    data = construct_regionally(interior, mrf)
    data = construct_regionally(Array, data)
    compact_data!(global_field, global_grid, data, mrf.grid.partition)
    
    fill_halo_regions!(global_field)
    return global_field
end

## Functions applied regionally
set!(mrf::MultiRegionField, v) = apply_regionally!(set!, mrf, v)

compute_at!(mrf::GriddedMultiRegionField, time)  = apply_regionally!(compute_at!, mrf, time)
compute_at!(mrf::MultiRegionComputedField, time) = apply_regionally!(compute_at!, mrf, time)

new_data(FT::DataType, mrg::MultiRegionGrid, args...) = construct_regionally(new_data, FT, mrg, args...)
@inline hasnan(field::MultiRegionField) = (&)(hasnan.(construct_regionally(parent, field).regions)...)

validate_indices(indices, loc, mrg::MultiRegionGrid, args...) = 
    construct_regionally(validate_indices, indices, loc, mrg.region_grids, args...)

FieldBoundaryBuffers(grid::MultiRegionGrid, args...; kwargs...) = 
    construct_regionally(FieldBoundaryBuffers, grid, args...; kwargs...)

FieldBoundaryConditions(mrg::MultiRegionGrid, loc, args...; kwargs...) =
  construct_regionally(inject_regional_bcs, mrg, Iterate(1:length(mrg)), Reference(mrg.partition), Reference(loc), args...; kwargs...)

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

function inject_regional_bcs(grid, region, partition, loc, args...;   
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
  return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

function Base.show(io::IO, field::MultiRegionField)

  bcs = field.boundary_conditions

  prefix =
      string("$(summary(field))\n",
             "├── grid: ", summary(field.grid), '\n',
             "├── boundary conditions: ", summary(bcs), '\n')
  middle = isnothing(field.operand) ? "" :
      string("├── operand: ", summary(field.operand), '\n',
             "├── status: ", summary(field.status), '\n')

  suffix = string("└── data: ", summary(field.data), '\n',
                  "    └── ", data_summary(field))

  print(io, prefix, middle, suffix)
end


