const PolarBoundaryCondition{V} = BoundaryCondition{<:Value, <:PolarValue}

struct PolarValue{C}
    c :: C
end

PolarBoundaryCondition(field) = 
    ValueBoundaryCondition(PolarValue(field))

update_boundary_condition!(bcs::PolarBoundaryCondition, ::Val{:south}, field, model) = 
    bcs.value.c .= mean(interior(field, :, 1, :), dims = 1)

update_boundary_condition!(bcs::PolarBoundaryCondition, ::Val{:north}, field, model) = 
    bcs.value.c .= mean(interior(field, :, 1, :), dims = 1)

# North - South flux boundary conditions are not valid on a Latitude-Longitude grid if the last / first rows represent the poles
function regularize_north_boundary_condition(bc::DefaultBoundaryCondition, grid::LatitudeLongitudeGrid, loc, dim, args...) where C
    φmax = φnode(grid.Ny+1, grid, Face()) 
    
    if φmax == 90 
        _, LY, LZ = loc
        field = Field{Nothing, LY, LZ}(grid)
        return PolarBoundaryCondition(field)
    else
        return regularize_boundary_condition(bc, grid, args...)
    end
end

# North - South flux boundary conditions are not valid on a Latitude-Longitude grid if the last / first rows represent the poles
function regularize_south_boundary_condition(bc::DefaultBoundaryCondition, grid::LatitudeLongitudeGrid, loc, dim, args...) 
    φmin = φnode(1, grid, Face()) 

    if φmin == - 90 
        _, LY, LZ = loc
        field = Field{Nothing, LY, LZ}(grid)
        return PolarBoundaryCondition(field)
    else
        return regularize_boundary_condition(bc, grid, args...)
    end
end