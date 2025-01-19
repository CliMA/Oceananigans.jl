using Oceananigans.Grids: inactive_node, new_data, YFlatGrid
using CUDA: @allowscalar

struct PolarValue{D, S}
    data :: D
    side :: S
end

Adapt.adapt_structure(to, pv::PolarValue) = PolarValue(Adapt.adapt(to, pv.data), nothing)

const PolarBoundaryCondition{V} = BoundaryCondition{<:Value, <:PolarValue}

function PolarBoundaryCondition(grid, side, zloc)
    FT   = eltype(grid)
    loc  = (Nothing, Nothing, zloc)
    data = new_data(FT, grid, loc)
    return ValueBoundaryCondition(PolarValue(data, side))
end

# Just a column
@inline getbc(pv::BC{<:Value, <:PolarValue}, i, k, args...) = @inbounds pv.condition.data[1, 1, k]

# YFlat grids do not have boundary conditions!
latitude_north_auxiliary_bc(::YFlatGrid, args...) = nothing
latitude_south_auxiliary_bc(::YFlatGrid, args...) = nothing

# TODO: vectors should have a different treatment since vector components should account for the frame of reference.
# For the moment, the `PolarBoundaryConditions` is implemented only for fields that have `loc[1] == loc[2] == Center()`, which
# we assume are not components of horizontal vectors that would require rotation. (The `w` velocity if not a tracer, but it does
# not require rotation since it is a scalar field.)
# North - South flux boundary conditions are not valid on a Latitude-Longitude grid if the last / first rows represent the poles
function latitude_north_auxiliary_bc(grid, loc, default_bc=DefaultBoundaryCondition()) 
    # Check if the halo lies beyond the north pole
    φmax = @allowscalar φnode(grid.Ny+1, grid, Center()) 
    
    # Assumption: fields at `Center`s in x and y are not vector components
    rotated_field = loc[1] != Center || loc[2] != Center

    # No problem!
    if φmax < 90 || rotated_field
        return default_bc
    end

    return PolarBoundaryCondition(grid, :north, loc[3])
end

# North - South flux boundary conditions are not valid on a Latitude-Longitude grid if the last / first rows represent the poles
function latitude_south_auxiliary_bc(grid, loc, default_bc=DefaultBoundaryCondition()) 
    # Check if the halo lies beyond the south pole
    φmin = @allowscalar φnode(0, grid, Face()) 

    # Assumption: fields at `Center`s in x and y are not vector components
    rotated_field = loc[1] != Center || loc[2] != Center

    # No problem!
    if φmin > -90 || rotated_field
        return default_bc
    end

    return PolarBoundaryCondition(grid, :south, loc[3])
end

regularize_north_boundary_condition(bc::DefaultBoundaryCondition, grid::LatitudeLongitudeGrid, loc, args...) = 
    regularize_boundary_condition(latitude_north_auxiliary_bc(grid, loc, bc), grid, loc, args...)

regularize_south_boundary_condition(bc::DefaultBoundaryCondition, grid::LatitudeLongitudeGrid, loc, args...) = 
    regularize_boundary_condition(latitude_south_auxiliary_bc(grid, loc, bc), grid, loc, args...)

@kernel function _average_pole_value!(data, c, j, grid, loc)
    i′, j′, k = @index(Global, NTuple)
    c̄ = zero(grid)
    n = 0
    @inbounds for i in 1:grid.Nx
        inactive = inactive_node(i, j, k, grid, loc...)
        c̄ += ifelse(inactive, 0, c[i, j, k])
        n += ifelse(inactive, 0, 1)
    end
    @inbounds data[i′, j′, k] = ifelse(n == 0,  0,  c̄ / n)
end

function update_pole_value!(bc::PolarValue, c, grid, loc) 
    j = bc.side == :north ? grid.Ny : 1
    Nz = size(c, 3)
    Oz = c.offsets[3]
    params = KernelParameters(1:1, 1:1, 1+Oz:Nz+Oz)
    launch!(architecture(grid), grid, params, _average_pole_value!, bc.data, c, j, grid, loc)
    return nothing
end

function fill_south_halo!(c, bc::PolarBoundaryCondition, size, offset, loc, arch, grid, args...; only_local_halos = false, kwargs...) 
    update_pole_value!(bc.condition, c, grid, loc)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_only_south_halo!, c, bc, loc, grid, Tuple(args); kwargs...)
end

function fill_north_halo!(c, bc::PolarBoundaryCondition, size, offset, loc, arch, grid, args...; only_local_halos = false, kwargs...) 
    update_pole_value!(bc.condition, c, grid, loc)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_only_north_halo!, c, bc, loc, grid, Tuple(args); kwargs...)
end

function fill_south_and_north_halo!(c, south_bc::PolarBoundaryCondition, north_bc, size, offset, loc, arch, grid, args...; only_local_halos = false, kwargs...)
    update_pole_value!(south_bc.condition, c, grid, loc)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_south_and_north_halo!, c, south_bc, north_bc, loc, grid, Tuple(args); kwargs...)
end

function fill_south_and_north_halo!(c, south_bc, north_bc::PolarBoundaryCondition, size, offset, loc, arch, grid, args...; only_local_halos = false, kwargs...)
    update_pole_value!(north_bc.condition, c, grid, loc)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_south_and_north_halo!, c, south_bc, north_bc, loc, grid, Tuple(args); kwargs...)
end

function fill_south_and_north_halo!(c, south_bc::PolarBoundaryCondition, north_bc::PolarBoundaryCondition, size, offset, loc, arch, grid, args...; only_local_halos = false, kwargs...)
    update_pole_value!(south_bc.condition, c, grid, loc)
    update_pole_value!(north_bc.condition, c, grid, loc)
    return launch!(arch, grid, KernelParameters(size, offset),
                   _fill_south_and_north_halo!, c, south_bc, north_bc, loc, grid, Tuple(args); kwargs...)
end

# If it is a LatitudeLongitudeGrid, we include the PolarBoundaryConditions
function FieldBoundaryConditions(grid::LatitudeLongitudeGrid, location, indices=(:, :, :);
                                 west     = default_auxiliary_bc(topology(grid, 1)(), location[1]()),
                                 east     = default_auxiliary_bc(topology(grid, 1)(), location[1]()),
                                 south    = default_auxiliary_bc(topology(grid, 2)(), location[2]()),
                                 north    = default_auxiliary_bc(topology(grid, 2)(), location[2]()),
                                 bottom   = default_auxiliary_bc(topology(grid, 3)(), location[3]()),
                                 top      = default_auxiliary_bc(topology(grid, 3)(), location[3]()),
                                 immersed = NoFluxBoundaryCondition())

    north = latitude_north_auxiliary_bc(grid, location, north)
    south = latitude_south_auxiliary_bc(grid, location, south)

    return FieldBoundaryConditions(indices, west, east, south, north, bottom, top, immersed)
end