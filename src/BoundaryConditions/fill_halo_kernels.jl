using Oceananigans.Utils: configure_kernel

"""
$(TYPEDSIGNATURES)

Construct preconfigured boundary condition kernels for a given `data` array, `grid`,
and the provided `bcs` (a FieldBoundaryConditions` object).
Return a new `FieldBoundaryConditions` object with the preconfigured kernels and
ordered boundary conditions.
"""
Base.@constprop :aggressive function construct_boundary_conditions_kernels(bcs::FieldBoundaryConditions,
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
    reduced_dims = reduced_dimensions(loc)
    names = map(side_name, sides)

    kernels! = map(sides, ordered_bcs) do side, side_bcs
        bc      = select_bc(side_bcs)
        size    = fill_halo_size(data, side, indices, bc, loc, grid)
        offset  = fill_halo_offset(size, side, indices)
        fill_halo_kernel(side, bc, grid, size, offset, data, reduced_dims)
    end

    return NamedTuple{names}(kernels!), NamedTuple{names}(ordered_bcs)
end

@inline reduced_dimension(::Nothing, dim) = (dim,)
@inline reduced_dimension(loc, dim) = ()
@inline reduced_dimensions(loc) = (reduced_dimension(loc[1], 1)..., reduced_dimension(loc[2], 2)..., reduced_dimension(loc[3], 3)...)

@inline get_boundary_kernels(bcs::NoKernelFBC, data, grid, loc, indices) = fill_halo_kernels(bcs, data, grid, loc, indices)
@inline get_boundary_kernels(bcs, args...) = bcs.kernels, bcs.ordered_bcs

@inline periodic_size(c, dim1, dim2, size) = size

@inline function periodic_size(c, dim1, dim2, ::Symbol)
    parent_size = size(parent(c))
    return (parent_size[dim1], parent_size[dim2])
end

# Windowed fields (from `view(field, indices...)`) have OffsetArray offsets equal to the kernel
# launch offset in windowed dimensions, and negative ones elsewhere; subtracting the positive part
# avoids double-counting the launch offset and leaves only windows starting at a non-positive index.
@inline periodic_offset(c, dim1, dim2, kernel_offset) = (min(kernel_offset[1], 0), min(kernel_offset[2], 0))

@inline periodic_offset(c, dim1, dim2, ::Symbol) = (0, 0)

####
#### Fill halo configured kernels
####

const NoBC = Union{Nothing, Missing}

@inline fill_halo_kernel(value, bc::NoBC, args...) = nothing

@inline kernel_parameters(size, offset) = KernelParameters(size, offset)
@inline kernel_parameters(size::Symbol, offset) = size

#####
##### Two-sided fill halo kernels
#####

@inline fill_halo_kernel(::WestAndEast, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_west_and_east_halo!; reduced_dimensions)[1]

@inline fill_halo_kernel(::SouthAndNorth, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_south_and_north_halo!; reduced_dimensions)[1]

@inline fill_halo_kernel(::BottomAndTop, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_bottom_and_top_halo!; reduced_dimensions)[1]

#####
##### One-sided fill halo kernels
#####

@inline fill_halo_kernel(::West, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_west_halo!; reduced_dimensions)[1]

@inline fill_halo_kernel(::East, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_east_halo!; reduced_dimensions)[1]

@inline fill_halo_kernel(::South, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_south_halo!; reduced_dimensions)[1]

@inline fill_halo_kernel(::North, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_north_halo!; reduced_dimensions)[1]

@inline fill_halo_kernel(::Bottom, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_bottom_halo!; reduced_dimensions)[1]

@inline fill_halo_kernel(::Top, bc::BoundaryCondition, grid, size, offset, data, reduced_dimensions) =
    configure_kernel(architecture(grid), grid, kernel_parameters(size, offset), _fill_only_top_halo!; reduced_dimensions)[1]

#####
##### Periodic fill halo kernels (Always two-sided)
#####

struct PeriodicFillHalo{K, N, H}
    kernel :: K
    PeriodicFillHalo(kernel, ::Val{N}, ::Val{H}) where {N, H} = new{typeof(kernel), N, H}(kernel)
end

@inline function fill_halo_kernel(::WestAndEast, bc::PBC, grid, size, offset, data, reduced_dimensions)
    yz_size  = periodic_size(data, 2, 3, size)
    yz_offset = periodic_offset(data, 2, 3, offset)
    kernel = configure_kernel(architecture(grid), grid, kernel_parameters(yz_size, yz_offset), _fill_periodic_west_and_east_halo!)[1]
    return PeriodicFillHalo(kernel, Val(Base.size(grid, 1)), Val(halo_size(grid, 1)))
end

@inline function fill_halo_kernel(::SouthAndNorth, bc::PBC, grid, size, offset, data, reduced_dimensions)
    xz_size   = periodic_size(data, 1, 3, size)
    xz_offset = periodic_offset(data, 1, 3, offset)
    kernel = configure_kernel(architecture(grid), grid, kernel_parameters(xz_size, xz_offset), _fill_periodic_south_and_north_halo!)[1]
    return PeriodicFillHalo(kernel, Val(Base.size(grid, 2)), Val(halo_size(grid, 2)))
end

@inline function fill_halo_kernel(::BottomAndTop, bc::PBC, grid, size, offset, data, reduced_dimensions)
    xy_size   = periodic_size(data, 1, 2, size)
    xy_offset = periodic_offset(data, 1, 2, offset)
    kernel = configure_kernel(architecture(grid), grid, kernel_parameters(xy_size, xy_offset), _fill_periodic_bottom_and_top_halo!)[1]
    return PeriodicFillHalo(kernel, Val(Base.size(grid, 3)), Val(halo_size(grid, 3)))
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
