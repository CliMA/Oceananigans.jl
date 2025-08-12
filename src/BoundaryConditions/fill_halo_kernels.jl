
function fill_halo_kernels!(data, grid, loc, indices, boundary_conditions::FieldBoundaryConditions)

    arch = architecture(grid)

    fill_halos!, bcs = permute_boundary_conditions(bcs)
    fill_halos! = tuple(fill_halos!...)

    kernels! = []

    for task in 1:length(fill_halos!)
        fill_halo! = fill_halos![task]
        bc = bcs[task]
        size = fill_halo_size(data, fill_halo!, indices, bc[1], location, grid)

        size   = fill_halo_size(data, fill_halo!, indices, bc[1], loc, grid)
        offset = fill_halo_offset(size, fill_halo!, indices)

        kernel! = fill_halo_kernel!(Val(fill_halo!), bc[1], arch, size, offset, loc)

        push!(kernels!, kernel!)
    end

    return kernels!, bcs
end

####
#### Fill halo configured kernels
####

