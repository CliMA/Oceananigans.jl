using Oceananigans.BoundaryConditions: default_auxiliary_field_boundary_condition
using Oceananigans.Fields: FunctionField

import Oceananigans.Fields: set!, validate_field_data, validate_boundary_conditions
import Oceananigans.BoundaryConditions: FieldBoundaryConditions
import Oceananigans.Grids: new_data
import Base: fill!

const MultiRegionField{LX, LY, LZ, O}                  = Field{LX, LY, LZ, O, <:MultiRegionGrid} where {LX, LY, LZ, O}
const MultiRegionFunctionField{LX, LY, LZ, C, P, F, G} = FunctionField{LX, LY, LZ, C, P, F, <:MultiRegionGrid} where {LX, LY, LZ, C, P, F}

const MultiRegionFields = Union{MultiRegionField, MultiRegionFunctionField}

isregional(f::MultiRegionFields) = true

devices(f::MultiRegionFields) = devices(f.grid)
switch_device!(f::MultiRegionFields, i) = switch_device!(f.grid, i)
getdevice(f::MultiRegionFields, i) = getdevice(f.grid, i)

regions(f::MultiRegionField) = 1:length(f.data)

getregion(f::MultiRegionFunctionField{LX, LY, LZ}, i) where {LX, LY, LZ} =
  FunctionField{LX, LY, LZ}(
        getregion(f.func, i),
        getregion(f.grid, i),
        clock = getregion(f.clock, i),
        parameters = getregion(f.parameters, i))

getregion(f::MultiRegionField{LX, LY, LZ}, i) where {LX, LY, LZ} =
  Field{LX, LY, LZ}(
        getregion(f.grid, i),
        getregion(f.data, i),
        getregion(f.boundary_conditions, i),
        getregion(f.operand, i),
        getregion(f.status, i))

new_data(FT, mrg::MultiRegionGrid, loc) = construct_regionally(new_data, FT, mrg.region_grids, loc)
# set!(f::MultiRegionField, func::Function) = apply_regionally!(set!, f, func)

fill!(f::MultiRegionField, val) = apply_regionally!(fill!, f, val)

validate_field_data(loc, data, mrg::MultiRegionGrid) = apply_regionally!(validate_field_data, loc, data, mrg.region_grids)
validate_boundary_conditions(loc, mrg::MultiRegionGrid, bcs) = apply_regionally!(validate_boundary_conditions, loc, mrg.region_grids, bcs)

FieldBoundaryConditions(mrg::MultiRegionGrid, loc; kwargs...) =
  construct_regionally(inject_regional_bcs, mrg, Iterate(regions(mrg.region_grids)), Reference(mrg.partition), Reference(loc); kwargs...)

function inject_regional_bcs(grid, region, partition, loc;   
                              west = default_auxiliary_field_boundary_condition(topology(grid, 1)(), loc[1]()),
                              east = default_auxiliary_field_boundary_condition(topology(grid, 1)(), loc[1]()),
                             south = default_auxiliary_field_boundary_condition(topology(grid, 2)(), loc[2]()),
                             north = default_auxiliary_field_boundary_condition(topology(grid, 2)(), loc[2]()),
                            bottom = default_auxiliary_field_boundary_condition(topology(grid, 3)(), loc[3]()),
                               top = default_auxiliary_field_boundary_condition(topology(grid, 3)(), loc[3]()),
                          immersed = NoFluxBoundaryCondition())

  west  = inject_west_boundary(region, partition, west)
  east  = inject_east_boundary(region, partition, east)
  south = inject_south_boundary(region, partition, south)
  north = inject_north_boundary(region, partition, north)
  return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end



Base.size(f::MultiRegionField) = size(getregion(f.grid, 1))
