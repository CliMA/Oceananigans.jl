import Oceananigans.Utils: _launch!

function _launch!(arch::Distributed, args...; kwargs...)
    child_arch = child_architecture(arch)
    return _launch!(child_arch, args...; kwargs...)
end

# Launch kernels over conditioned cell maps
@inline function _launch!(arch::Distributed, grid, workspec, kernel!,
                          first_kernel_arg, second_kernel_arg, other_kernel_args::InteriorBoundarySet;
                          active_cells_map::InteriorBoundarySet, kwargs...)
    child_arch = child_architecture(arch)
    _launch!(child_arch, grid, workspec, kernel!, first_kernel_arg, second_kernel_arg, other_kernel_args;
          active_cells_map=active_cells_map, kwargs...)
end
