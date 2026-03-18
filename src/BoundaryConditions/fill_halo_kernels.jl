using Oceananigans.Utils: configure_kernel

"""
    construct_boundary_conditions_kernels(bcs::FieldBoundaryConditions,
                                          data::OffsetArray,
                                          grid::AbstractGrid,
                                          loc, indices)

Construct preconfigured boundary condition kernels for a given `data` array, `grid`,
and the provided `bcs` (a FieldBoundaryConditions` object).
Return a new `FieldBoundaryConditions` object with the preconfigured kernels and
ordered boundary conditions.
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

# If the bcs are nothing or missing... they remain nothing or missing
construct_boundary_conditions_kernels(::Nothing, data, grid, loc, indices) = nothing
construct_boundary_conditions_kernels(::Missing, data, grid, loc, indices) = missing

# Select the valid BC out of a tuple to configure the kernel
@inline select_bc(bcs::Tuple) = @inbounds bcs[1]
@inline select_bc(bcs::Tuple{<:Nothing, <:BoundaryCondition}) = @inbounds bcs[2]
@inline select_bc(bcs::Tuple{<:BoundaryCondition, <:Nothing}) = @inbounds bcs[1]
@inline select_bc(bcs::BoundaryCondition) = bcs

@inline function fill_halo_kernels(bcs::FieldBoundaryConditions, data::OffsetArray, grid::AbstractGrid, loc, indices)
    sides, ordered_bcs = permute_boundary_conditions(bcs)
    reduced_dimensions = findall(x -> x isa Nothing, loc)
    reduced_dimensions = tuple(reduced_dimensions...)
    names = Tuple(side_name(side) for side in sides)
    kernels! = []

    for task in 1:length(sides)
        side = sides[task]
        bc   = select_bc(ordered_bcs[task])

        size    = fill_halo_size(data, side, indices, bc, loc, grid)
        offset  = fill_halo_offset(size, side, indices)
        kernel! = fill_halo_kernel(side, bc, grid, size, offset, data, reduced_dimensions)

        push!(kernels!, kernel!)
    end

    kernels! = tuple(kernels!...)

    return NamedTuple{names}(kernels!), NamedTuple{names}(ordered_bcs)
end

@inline get_boundary_kernels(bcs::NoKernelFBC, data, grid, loc, indices) = fill_halo_kernels(bcs, data, grid, loc, indices)
@inline get_boundary_kernels(bcs, args...) = bcs.kernels, bcs.ordered_bcs

@inline periodic_size(c, dim1, dim2, size) = size

@inline function periodic_size(c, dim1, dim2, ::Symbol)
    parent_size = size(parent(c))
    return (parent_size[dim1], parent_size[dim2])
end

@inline function periodic_offset(c, dim1, dim2, kernel_offset)
    field_offsets = (c.offsets[dim1], c.offsets[dim2])

    # Windowed fields (from `view(field, indices...)`) have positive OffsetArray offsets                                                                                                                                         
    # in windowed dimensions. Subtract these to avoid double-counting the kernel launch offset.
    offset1 = kernel_offset[1] - max(0, field_offsets[1])
    offset2 = kernel_offset[2] - max(0, field_offsets[2])
    
    return (offset1, offset2)
end

@inline periodic_offset(c, dim1, dim2, ::Symbol) = (0, 0)

####
#### Fill halo configured kernels
####

const NoBC = Union{Nothing, Missing}

fill_halo_kernel(value, bc::NoBC, args...) = nothing

@inline kernel_parameters(size, offset) = KernelParameters(size, offset)
@inline kernel_parameters(size::Symbol, offset) = size

#####
##### Two-sided fill halo kernels
#####

fill_halo_kernel(::WestAndEast, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_west_and_east_halo!; reduced_dimensions)[1]

fill_halo_kernel(::SouthAndNorth, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_south_and_north_halo!; reduced_dimensions)[1]

fill_halo_kernel(::BottomAndTop, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_bottom_and_top_halo!; reduced_dimensions)[1]

#####
##### One-sided fill halo kernels
#####

fill_halo_kernel(::West, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_west_halo!; reduced_dimensions)[1]

fill_halo_kernel(::East, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_east_halo!; reduced_dimensions)[1]

fill_halo_kernel(::South, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_south_halo!; reduced_dimensions)[1]

fill_halo_kernel(::North, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_north_halo!; reduced_dimensions)[1]

fill_halo_kernel(::Bottom, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_bottom_halo!; reduced_dimensions)[1]

fill_halo_kernel(::Top, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_top_halo!; reduced_dimensions)[1]

#####
##### Periodic fill halo kernels (Always two-sided)
#####

struct PeriodicFillHalo{K, N, H}
    kernel :: K
    PeriodicFillHalo(kernel, ::Val{N}, ::Val{H}) where {N, H} = new{typeof(kernel), N, H}(kernel)
end

function fill_halo_kernel(::WestAndEast, bc::PBC, grid, size, offset, data, reduced_dimensions)
    yz_size  = periodic_size(data, 2, 3, size)
    yz_offset = periodic_offset(data, 2, 3, offset)
    kernel = configure_kernel(architecture(grid), grid, kernel_parameters(yz_size, yz_offset), _fill_periodic_west_and_east_halo!)[1]
    return PeriodicFillHalo(kernel, Val(grid.Nx), Val(grid.Hx))
end

function fill_halo_kernel(::SouthAndNorth, bc::PBC, grid, size, offset, data, reduced_dimensions)
    xz_size   = periodic_size(data, 1, 3, size)
    xz_offset = periodic_offset(data, 1, 3, offset)
    kernel = configure_kernel(architecture(grid), grid, kernel_parameters(xz_size, xz_offset), _fill_periodic_south_and_north_halo!)[1]
    return PeriodicFillHalo(kernel, Val(grid.Ny), Val(grid.Hy))
end

function fill_halo_kernel(::BottomAndTop, bc::PBC, grid, size, offset, data, reduced_dimensions)
    xy_size   = periodic_size(data, 1, 2, size)
    xy_offset = periodic_offset(data, 1, 2, offset)
    kernel = configure_kernel(architecture(grid), grid, kernel_parameters(xy_size, xy_offset), _fill_periodic_bottom_and_top_halo!)[1]
    return PeriodicFillHalo(kernel, Val(grid.Nz), Val(grid.Hz))
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
    @eval fill_halo_kernel(::$Side, bc::DCBC, grid, size, offset, data, reduced_dimensions) = DistributedFillHalo($Side())
end

#####
##### MultiRegion Boundary Conditions
#####

# A struct to hold the side of the fill_halo kernel
# These are defined in `src/MultiRegion/multi_region_boundary_conditions.jl`
struct MultiRegionFillHalo{S}
    side :: S
end

for Side in (:WestAndEast, :SouthAndNorth, :BottomAndTop, :West, :East, :South, :North, :Bottom, :Top)
    @eval fill_halo_kernel(::$Side, bc::MCBC, grid, size, offset, data, reduced_dimensions) =  MultiRegionFillHalo($Side())
end

#####
##### PeriodicFillHalo dispatch
#####

@inline fill_halo_event!(c, pfh::PeriodicFillHalo{K, N, H}, bcs::Tuple{Any, Any}, loc, grid, args...; kwargs...) where {K, N, H} =
    pfh.kernel(c, Val(N), Val(H))

@inline fill_halo_event!(c, pfh::PeriodicFillHalo{K, N, H}, bcs::Tuple{Any}, loc, grid, args...; kwargs...) where {K, N, H} =
    pfh.kernel(c, Val(N), Val(H))
end
