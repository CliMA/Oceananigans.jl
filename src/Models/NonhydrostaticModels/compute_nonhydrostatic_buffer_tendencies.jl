import Oceananigans.Models: compute_buffer_tendencies!

using Oceananigans.Models: buffer_tendency_kernel_parameters,
                           buffer_p_kernel_parameters,
                           buffer_κ_kernel_parameters,
                           buffer_parameters

using Oceananigans.TurbulenceClosures: required_halo_size_x, required_halo_size_y
using Oceananigans.Grids: XFlatGrid, YFlatGrid

# TODO: the code in this file is difficult to understand.
# Rewriting it may be helpful.

# We assume here that top/bottom BC are always synched (no partitioning in z)
function compute_buffer_tendencies!(model::NonhydrostaticModel)
    grid = model.grid
    arch = architecture(grid)

    p_parameters = buffer_p_kernel_parameters(grid, arch)
    κ_parameters = buffer_κ_kernel_parameters(grid, model.closure, arch)

    # We need new values for `p` and `κ`
    compute_auxiliaries!(model; p_parameters, κ_parameters)

    # parameters for communicating North / South / East / West side
    kernel_parameters = buffer_tendency_kernel_parameters(grid, arch)
    compute_interior_tendency_contributions!(model, kernel_parameters)

    return nothing
end

