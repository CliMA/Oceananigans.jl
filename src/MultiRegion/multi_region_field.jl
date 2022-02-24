using Oceananigans.BoundaryConditions: default_auxiliary_field_boundary_condition

import Oceananigans.Fields: set!, validate_field_data, validate_boundary_conditions
import Oceananigans.BoundaryConditions: FieldBoundaryConditions
import Oceananigans.Grids: new_data

const MultiRegionField{LX, LY, LZ, O} = Field{LX, LY, LZ, O, <:MultiRegionGrid} where {LX, LY, LZ, O}

isregional(f::MultiRegionField) = true

devices(f::MultiRegionField)  = devices(f.grid)
regions(f::MultiRegionField)  = 1:length(f.data)

switch_device!(f::MultiRegionField, i) = switch_device!(f.grid, i)

getdevice(f::MultiRegionField, i) = getdevice(f.grid, i)

getregion(f::MultiRegionField, i) =
  Field(location(f),
        getregion(f.grid, i),
        getregion(f.data, i),
        getregion(f.boundary_conditions, i),
        getregion(f.operand, i),
        getregion(f.status, i))

new_data(FT, mrg::MultiRegionGrid, loc) = construct_regionally(new_data, FT, mrg.region_grids, loc)
set!(f::MultiRegionField, func::Function) = apply_regionally!(set!, f, func)

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
