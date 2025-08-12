using Oceananigans.Utils: configure_kernel

function construct_boundary_conditions_kernels(bcs::FieldBoundaryConditions,
                                               data::OffsetArray,
                                               grid::AbstractGrid,
                                               loc,
                                               indices)

    kernels!, ordered_bcs = fill_halo_kernels!(data, grid, loc, indices, bcs)
    regularized_bcs = FieldBoundaryConditions(bcs.west, bcs.east, bcs.south, bcs.north,
                                              bcs.bottom, bcs.top, bcs.immersed,
                                              kernels!, ordered_bcs)
    return regularized_bcs
end

@inline fix_halo_offsets(o, co) = co > 0 ? o - co : o # Windowed fields have only positive offsets to correct

@inline periodic_size_and_offset(c, dim1, dim2, size, offset)     = (size, fix_halo_offsets.(offset, c.offsets[[dim1, dim2]]))
@inline periodic_size_and_offset(c, dim1, dim2, ::Symbol, offset) = (size(parent(c))[[dim1, dim2]], (0, 0))

function fill_halo_kernels!(data, grid, loc, indices, boundary_conditions::FieldBoundaryConditions)

    arch = architecture(grid)

    fill_halos!, bcs = permute_boundary_conditions(boundary_conditions)
    fill_halos! = tuple(fill_halos!...)

    kernels! = []

    for task in 1:length(fill_halos!)
        fill_halo! = fill_halos![task]
        bc         = bcs[task]

        size    = fill_halo_size(data, fill_halo!, indices, bc[1], loc, grid)
        offset  = fill_halo_offset(size, fill_halo!, indices)
        kernel! = fill_halo_kernel!(fill_halo!, bc[1], grid, size, offset, data)

        push!(kernels!, kernel!)
    end

    return tuple(kernels!...), bcs
end

####
#### Fill halo configured kernels
####

nothing_function(args...) = nothing

const NoBC = Union{Nothing, Missing}

fill_halo_kernel!(value, bc::NoBC, args...) = nothing_function(args...)

#####
##### Two-sided fill halo kernels
#####

fill_halo_kernel!(::typeof(fill_west_and_east_halo!), bc::BoundaryCondition, grid, size, offset) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_west_and_east_halo!)[1]

fill_halo_kernel!(::typeof(fill_south_and_north_halo!), bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_south_and_north_halo!)[1]

fill_halo_kernel!(::typeof(fill_bottom_and_top_halo!), bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_bottom_and_top_halo!)[1]

#####
##### One-sided fill halo kernels
#####

fill_halo_kernel!(::typeof(fill_west_halo!), bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_only_west_halo!)[1]

fill_halo_kernel!(::typeof(fill_east_halo!), bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_only_east_halo!)[1]

fill_halo_kernel!(::typeof(fill_south_halo!), bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_only_south_halo!)[1]

fill_halo_kernel!(::typeof(fill_north_halo!), bc::BoundaryCondition, grid, size, offset, data) =
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_only_north_halo!)[1]

fill_halo_kernel!(::typeof(fill_bottom_halo!), bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_only_bottom_halo!)[1]

fill_halo_kernel!(::typeof(fill_top_halo!), bc::BoundaryCondition, grid, size, offset, data) = 
    configure_kernel(architecture(grid), grid, KernelParameters(size, offset), _fill_only_top_halo!)[1]

#####
##### Periodic fill halo kernels (Always two-sided)
#####

function fill_halo_kernel!(::typeof(fill_west_and_east_halo!), bc::PBC, grid, size, offset, data) 
    yz_size, offset = periodic_size_and_offset(data, 2, 3, size, offset)
    return configure_kernel(architecture(grid), grid, KernelParameters(yz_size, offset), _fill_west_and_east_halo!)[1]
end

function fill_halo_kernel!(::typeof(fill_south_and_north_halo!), bc::PBC, grid, size, offset, data) 
    xz_size, offset = periodic_size_and_offset(data, 1, 3, size, offset)
    return configure_kernel(architecture(grid), grid, KernelParameters(xz_size, offset), _fill_south_and_north_halo!)[1]
end

function fill_halo_kernel!(::typeof(fill_bottom_and_top_halo!), bc::PBC, grid, size, offset, data) 
    xy_size, offset = periodic_size_and_offset(data, 1, 2, size, offset)
    return configure_kernel(architecture(grid), grid, KernelParameters(xy_size, offset), _fill_bottom_and_top_halo!)[1]
end

#####
##### Distributed boundary conditions?
#####