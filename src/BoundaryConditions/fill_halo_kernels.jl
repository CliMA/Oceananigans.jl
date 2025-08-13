using Oceananigans.Utils: configure_kernel

"""
    construct_boundary_conditions_kernels(bcs::FieldBoundaryConditions,
                                          data::OffsetArray,
                                          grid::AbstractGrid,
                                          loc,
                                          indices)

Constructs and attaches preconfigured boundary condition kernels for a given data array and grid
to the provided `FieldBoundaryConditions` object.
"""
function construct_boundary_conditions_kernels(bcs::FieldBoundaryConditions,
                                               data::OffsetArray,
                                               grid::AbstractGrid,
                                               loc,
                                               indices)

    arch = architecture(grid)

    sides, bcs = permute_boundary_conditions(boundary_conditions)
    sides = tuple(fill_halos!...)

    kernels! = []

    for task in 1:length(fill_halos!)
        side = sides[task]
        bc   = bcs[task]

        size    = fill_halo_size(data, side, indices, bc[1], loc, grid)
        offset  = fill_halo_offset(size, side, indices)
        kernel! = fill_halo_kernel!(side, bc[1], grid, size, offset, data)

        push!(kernels!, kernel!)
    end

    kernels! = tuple(kernels!...)
    regularized_bcs = FieldBoundaryConditions(bcs.west, bcs.east, bcs.south, bcs.north,
                                              bcs.bottom, bcs.top, bcs.immersed,
                                              kernels!, ordered_bcs)
    return regularized_bcs
end

@inline fix_halo_offsets(o, co) = co > 0 ? o - co : o # Windowed fields have only positive offsets to correct

@inline periodic_size_and_offset(c, dim1, dim2, size, offset)     = (size, fix_halo_offsets.(offset, c.offsets[[dim1, dim2]]))
@inline periodic_size_and_offset(c, dim1, dim2, ::Symbol, offset) = (size(parent(c))[[dim1, dim2]], (0, 0))

####
#### Fill halo configured kernels
####

@inline nothing_function(args...) = nothing

const NoBC = Union{Nothing, Missing}

fill_halo_kernel!(value, bc::NoBC, args...) = nothing_function(args...)

#####
##### Two-sided fill halo kernels
#####

fill_halo_kernel!(::WestAndEast, bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_west_and_east_halo!)[1]

fill_halo_kernel!(::SouthAndNorth, bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_south_and_north_halo!)[1]

fill_halo_kernel!(::BottomAndTop, bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_bottom_and_top_halo!)[1]

#####
##### One-sided fill halo kernels
#####

fill_halo_kernel!(::West, bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_only_west_halo!)[1]

fill_halo_kernel!(::East, bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_only_east_halo!)[1]

fill_halo_kernel!(::South, bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_only_south_halo!)[1]

fill_halo_kernel!(::North, bc::BoundaryCondition, grid, size, offset, data) =
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_only_north_halo!)[1]

fill_halo_kernel!(::Bottom, bc::BoundaryCondition, grid, size, offset, data) =
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_only_bottom_halo!)[1]

fill_halo_kernel!(::Top, bc::BoundaryCondition, grid, size, offset, data) =
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_only_top_halo!)[1]

#####
##### Periodic fill halo kernels (Always two-sided)
#####

function fill_halo_kernel!(::WestAndEast, bc::PBC, grid, size, offset, data) 
    yz_size, offset = periodic_size_and_offset(data, 2, 3, size, offset)
    return configure_kernel(architecture(grid), grid, KernelParameters(yz_size, offset), _fill_periodic_west_and_east_halo!)[1]
end

function fill_halo_kernel!(::SouthAndNorth, bc::PBC, grid, size, offset, data) 
    xz_size, offset = periodic_size_and_offset(data, 1, 3, size, offset)
    return configure_kernel(architecture(grid), grid, KernelParameters(xz_size, offset), _fill_periodic_south_and_north_halo!)[1]
end

function fill_halo_kernel!(::BottomAndTop, bc::PBC, grid, size, offset, data) 
    xy_size, offset = periodic_size_and_offset(data, 1, 2, size, offset)
    return configure_kernel(architecture(grid), grid, KernelParameters(xy_size, offset), _fill_periodic_bottom_and_top_halo!)[1]
end

#####
##### Distributed boundary conditions?
#####

# A struct to hold the side of the fill_halo kernel
# These are defined in `src/DistributedComputations/halo_communication.jl`
struct DistributedFillHalo{S}
    side :: S
end

for Side in (:WestAndEast, :SouthAndNorth, :BottomAndTop, :West, :East, :South, :North, :Bottom, :Top)
    @eval fill_halo_kernel!(::$Side, bc::DCBC, grid, size, offset, data) =  DistributedFillHalo($Side())
end