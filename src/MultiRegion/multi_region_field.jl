using Oceananigans.BoundaryConditions: default_auxiliary_bc
using Oceananigans.Fields: FunctionField, data_summary

import Oceananigans.Fields: set!, validate_field_data, validate_boundary_conditions, validate_indices, FieldBoundaryBuffers
import Oceananigans.BoundaryConditions: FieldBoundaryConditions
import Oceananigans.Grids: new_data
import Base: fill!

import Oceananigans.Simulations: hasnan

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
        getregion(f.indices, i),
        getregion(f.operand, i),
        getregion(f.status, i),
        getregion(f.boundary_buffers, i))

new_data(FT::DataType, mrg::MultiRegionGrid, args...) = construct_regionally(new_data, FT, mrg.region_grids, args...)

fill!(f::MultiRegionField, val) = apply_regionally!(fill!, f, val)
hasnan(field::MultiRegionField) = (&)(hasnan.(construct_regionally(parent, field).regions)...)

validate_field_data(loc, data, mrg::MultiRegionGrid, args...) = apply_regionally!(validate_field_data, loc, data, mrg.region_grids, args...)
validate_boundary_conditions(loc, mrg::MultiRegionGrid, bcs, args...) = apply_regionally!(validate_boundary_conditions, loc, mrg.region_grids, bcs, args...)
validate_indices(indices, loc, mrg::MultiRegionGrid, args...) = construct_regionally(validate_indices, indices, loc, mrg.region_grids, args...)

FieldBoundaryConditions(mrg::MultiRegionGrid, loc, args...; kwargs...) =
  construct_regionally(inject_regional_bcs, mrg, Iterate(1:length(mrg)), Reference(mrg.partition), Reference(loc), args...; kwargs...)

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions, mrg::MultiRegionGrid, field_name::Symbol, prognostic_field_names=nothing)
    loc = assumed_field_location(field_name)
    return FieldBoundaryConditions(mrg, loc)
end

FieldBoundaryBuffers(grid::MultiRegionGrid, args...) = construct_regionally(FieldBoundaryBuffers, grid, args...)

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

Base.size(f::MultiRegionField) = size(getregion(f.grid, 1))

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


