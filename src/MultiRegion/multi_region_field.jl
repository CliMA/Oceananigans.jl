import Oceananigans.Fields: set!, validate_field_data
import Oceananigans.Grids: new_data

const MultiRegionField{LX, LY, LZ, O} = Field{LX, LY, LZ, O, <:MultiRegionGrid} where {LX, LY, LZ, O}

isregional(f::MultiRegionField) = true
devices(f::MultiRegionField) = devices(f.grid)
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

set!(f::MultiRegionField, func::Function)       = apply_regionally!(set!, f, func)
set!(f::MultiRegionField, g::MultiRegionObject) = apply_regionally!(set!, f, g)

validate_field_data(loc, data, grid::MultiRegionGrid) = apply_regionally!(validate_field_data, loc, data, grid.region_grids)

Base.show(io::IO, field::MultiRegionField) = print(io, "MultiRegionField")