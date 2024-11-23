condition_operand(data, grid, Loc, condition, mask) = data

struct PolarValue{C}
    c :: C
end

const PolarBoundaryCondition{V} = BoundaryCondition{<:Value, <:PolarValue}

PolarBoundaryCondition(field) = 
    ValueBoundaryCondition(PolarValue(field))

@inline getbc(pv::PolarValue, args...) = getbc(pv.c, args...)

# TODO: vectors should have a different treatment since vector components should account for the frame of reference
# North - South flux boundary conditions are not valid on a Latitude-Longitude grid if the last / first rows represent the poles
function regularize_north_boundary_condition(bc::DefaultBoundaryCondition, grid::LatitudeLongitudeGrid, loc, args...) 
    φmax = φnode(grid.Ny+1, grid, Face()) 
    
    if φmax == 90 
        _, LY, LZ = loc
        field = Field{Nothing, LY, LZ}(grid; indices = (:, grid.Ny, :))
        return PolarBoundaryCondition(field)
    else
        return regularize_boundary_condition(bc, grid, loc, args...)
    end
end

# North - South flux boundary conditions are not valid on a Latitude-Longitude grid if the last / first rows represent the poles
function regularize_south_boundary_condition(bc::DefaultBoundaryCondition, grid::LatitudeLongitudeGrid, loc, args...) 
    φmin = φnode(1, grid, Face()) 

    if φmin == - 90 
        _, LY, LZ = loc
        field = Field{Nothing, LY, LZ}(grid; indices = (:, 1, :))
        return PolarBoundaryCondition(field)
    else
        return regularize_boundary_condition(bc, grid, loc, args...)
    end
end

function update_pole_value!(bc::PolarBoundaryCondition, c, grid, loc) 
    operand = condition_operand(c, grid, loc, nothing, 0)
    mean!(bc.condition.c, operand)
    return nothing
end

function fill_south_halo!(c, bc::PolarBoundaryCondition, size, offset, loc, arch, grid, args...; only_local_halos = false, kwargs...) 
    update_pole_value!(bc, c, grid, loc)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_only_south_halo!, c, bc, loc, grid, Tuple(args); kwargs...)
end

function fill_north_halo!(c, bc::PolarBoundaryCondition, size, offset, loc, arch, grid, args...; only_local_halos = false, kwargs...) 
    update_pole_value!(bc, c, grid, loc)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_only_north_halo!, c, bc, loc, grid, Tuple(args); kwargs...)
end

function fill_south_and_north_halo!(c, south_bc::PolarBoundaryCondition, north_bc, size, offset, loc, arch, grid, args...; only_local_halos = false, kwargs...)
    update_pole_value!(south_bc, c, grid, loc)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_south_and_north_halo!, c, south_bc, north_bc, loc, grid, Tuple(args); kwargs...)
end

function fill_south_and_north_halo!(c, south_bc, north_bc::PolarBoundaryCondition, size, offset, loc, arch, grid, args...; only_local_halos = false, kwargs...)
    update_pole_value!(north_bc, c, grid, loc)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_south_and_north_halo!, c, south_bc, north_bc, loc, grid, Tuple(args); kwargs...)
end

function fill_south_and_north_halo!(c, south_bc::PolarBoundaryCondition, north_bc::PolarBoundaryCondition, size, offset, loc, arch, grid, args...; only_local_halos = false, kwargs...)
    update_pole_value!(south_bc, c, grid, loc)
    update_pole_value!(north_bc, c, grid, loc)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_south_and_north_halo!, c, south_bc, north_bc, loc, grid, Tuple(args); kwargs...)
end
