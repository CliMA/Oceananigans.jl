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
                                               loc, indices)

    kernels!, ordered_bcs = fill_halo_kernels(bcs, data, grid, loc, indices)
    regularized_bcs = FieldBoundaryConditions(bcs.west, bcs.east, bcs.south, bcs.north,
                                              bcs.bottom, bcs.top, bcs.immersed,
                                              kernels!, ordered_bcs)
    return regularized_bcs
end

@inline function fill_halo_kernels(bcs::FieldBoundaryConditions,
                                   data::OffsetArray,
                                   grid::AbstractGrid,
                                   loc, indices)

    arch = architecture(grid)
    sides, ordered_bcs = permute_boundary_conditions(bcs)
    sides = tuple(sides...)
    reduced_dimensions = findall(x -> x isa Nothing, loc)
    reduced_dimensions = tuple(reduced_dimensions...)
    kernels! = []

    for task in 1:length(sides)
        side = sides[task]
        bc   = ordered_bcs[task]

        size    = fill_halo_size(data, side, indices, bc[1], loc, grid)
        offset  = fill_halo_offset(size, side, indices)
        kernel! = fill_halo_kernel!(side, bc[1], grid, size, offset, data, reduced_dimensions)

        push!(kernels!, kernel!)
    end

    kernels! = tuple(kernels!...)

    return kernels!, ordered_bcs
end

@inline get_boundary_kernels(bcs::NoKernelFBC, data, grid, loc, indices) = fill_halo_kernels(bcs, data, grid, loc, indices)
@inline get_boundary_kernels(bcs::FieldBoundaryConditions, args...)      = bcs.kernels!, bcs.ordered_bcs

@inline fix_halo_offsets(o, co) = co > 0 ? o - co : o # Windowed fields have only positive offsets to correct

@inline periodic_size_and_offset(c, dim1, dim2, size, offset)     = (size, fix_halo_offsets.(offset, c.offsets[[dim1, dim2]]))
@inline periodic_size_and_offset(c, dim1, dim2, ::Symbol, offset) = (size(parent(c))[[dim1, dim2]], (0, 0))

####
#### Fill halo configured kernels
####

const NoBC = Union{Nothing, Missing}

fill_halo_kernel!(value, bc::NoBC, args...) = nothing

@inline kernel_parameters(size, offset) = KernelParameters(size, offset)
@inline kernel_parameters(size::Symbol, offset) = size

#####
##### Two-sided fill halo kernels
#####

fill_halo_kernel!(::WestAndEast, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) = 
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_west_and_east_halo!; reduced_dimensions)[1]

fill_halo_kernel!(::SouthAndNorth, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_south_and_north_halo!; reduced_dimensions)[1]

fill_halo_kernel!(::BottomAndTop, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) = 
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_bottom_and_top_halo!; reduced_dimensions)[1]

#####
##### One-sided fill halo kernels
#####

fill_halo_kernel!(::West, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) = 
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_west_halo!; reduced_dimensions)[1]

fill_halo_kernel!(::East, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) = 
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_east_halo!; reduced_dimensions)[1]

fill_halo_kernel!(::South, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) = 
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_south_halo!; reduced_dimensions)[1]

fill_halo_kernel!(::North, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_north_halo!; reduced_dimensions)[1]

fill_halo_kernel!(::Bottom, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_bottom_halo!; reduced_dimensions)[1]

fill_halo_kernel!(::Top, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_top_halo!; reduced_dimensions)[1]

#####
##### Periodic fill halo kernels (Always two-sided)
#####

function fill_halo_kernel!(::WestAndEast, bc::PBC, grid, size, offset, data, reduced_dimensions) 
    yz_size, offset = periodic_size_and_offset(data, 2, 3, size, offset)
    return configure_kernel(architecture(grid), grid, kernel_parameters(yz_size, offset), _fill_periodic_west_and_east_halo!)[1]
end

function fill_halo_kernel!(::SouthAndNorth, bc::PBC, grid, size, offset, data, reduced_dimensions) 
    xz_size, offset = periodic_size_and_offset(data, 1, 3, size, offset)
    return configure_kernel(architecture(grid), grid, kernel_parameters(xz_size, offset), _fill_periodic_south_and_north_halo!)[1]
end

function fill_halo_kernel!(::BottomAndTop, bc::PBC, grid, size, offset, data, reduced_dimensions) 
    xy_size, offset = periodic_size_and_offset(data, 1, 2, size, offset)
    return configure_kernel(architecture(grid), grid, kernel_parameters(xy_size, offset), _fill_periodic_bottom_and_top_halo!)[1]
end

#####
##### Distributed Boundary Conditions
#####

# A struct to hold the side of the fill_halo kernel
# These are defined in `src/DistributedComputations/halo_communication.jl`
struct DistributedFillHalo{S}
    side :: S
end

for Side in (:WestAndEast, :SouthAndNorth, :BottomAndTop, :West, :East, :South, :North, :Bottom, :Top)
    @eval fill_halo_kernel!(::$Side, bc::DCBC, grid, size, offset, data, reduced_dimensions) =  DistributedFillHalo($Side())
end

#####
##### TODO: MultiRegion Boundary Conditions 
#####

# A struct to hold the side of the fill_halo kernel
# These are defined in `src/MultiRegion/multi_region_boundary_conditions.jl`
struct MultiRegionFillHalo{S}
    side :: S
end

for Side in (:WestAndEast, :SouthAndNorth, :BottomAndTop, :West, :East, :South, :North, :Bottom, :Top)
    @eval fill_halo_kernel!(::$Side, bc::MCBC, grid, size, offset, data, reduced_dimensions) =  MultiRegionFillHalo($Side())
end