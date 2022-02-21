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

new_data(FT, mrg::MultiRegionGrid, loc) = apply_regionally(new_data, FT, mrg.region_grids, loc)
set!(f::MultiRegionField, func::Function) = apply_regionally!(set!, f, func)

validate_field_data(loc, data, mrg::MultiRegionGrid) = apply_regionally!(validate_field_data, loc, data, grids(mrg))
validate_boundary_conditions(loc, mrg::MultiRegionGrid, bcs) = apply_regionally!(validate_boundary_conditions, loc, grids(mrg), bcs)

FieldBoundaryConditions(mrg::MultiRegionGrid, loc) =
  apply_regionally(inject_regional_bcs, MultiRegionObject(Tuple(1:length(partition(mrg))), devices(mrg)), mrg.partition, apply_regionally(FieldBoundaryConditions, grids(mrg), loc))

Base.show(io::IO, field::MultiRegionField) = print(io, "MultiRegionField")
