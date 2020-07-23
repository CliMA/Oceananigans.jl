#####
##### Utilities for launching kernels
#####

using CUDA
using KernelAbstractions
using Oceananigans.Architectures

const MAX_THREADS_PER_BLOCK = 256

"""
    work_layout(grid, dims)

Returns the `workgroup` and `worksize` for launching a kernel over `dims`
on `grid`. The `workgroup` is a tuple specifying the threads per block in each dimension.
The `worksize` specifies the range of the loop in each dimension.

For more information, see: https://github.com/climate-machine/Oceananigans.jl/pull/308
"""
function work_layout(grid, dims)

    workgroup = grid.Nx == 1 && grid.Ny == 1 ?

                    # One-dimensional column models:
                    (1, 1) :

                grid.Nx == 1 ?

                    # Two-dimensional y-z slice models:
                    (1, min(MAX_THREADS_PER_BLOCK, grid.Ny)) :

                grid.Ny == 1 ?

                    # Two-dimensional x-z slice models:
                    (1, min(MAX_THREADS_PER_BLOCK, grid.Nx)) :

                    # Three-dimensional models use the default (16, 16)
                    (Int(√(MAX_THREADS_PER_BLOCK)), Int(√(MAX_THREADS_PER_BLOCK)))

    worksize = dims == :xyz ? (grid.Nx, grid.Ny, grid.Nz) :
               dims == :xy  ? (grid.Nx, grid.Ny) :
               dims == :xz  ? (grid.Nx, grid.Nz) :
               dims == :yz  ? (grid.Ny, grid.Nz) : error("Unsupported launch configuration: $dims")

    return workgroup, worksize
end

"""
    launch!(arch, grid, layout, kernel!, args...; dependencies=nothing, kwargs...)

Launches `kernel!`, with arguments `args` and keyword arguments `kwargs`,
over the `dims` of `grid` on the architecture `arch`.

Returns an `event` token associated with the `kernel!` launch.

The keyword argument `dependencies` is an `Event` or `MultiEvent` specifying prior kernels
that must complete before `kernel!` is launched.
"""
function launch!(arch, grid, dims, kernel!, args...; dependencies=nothing, kwargs...)

    workgroup, worksize = work_layout(grid, dims)

    loop! = kernel!(Oceananigans.Architectures.device(arch), workgroup, worksize)

    @debug "worksize and kernel:" worksize kernel!

    event = loop!(args...; dependencies=dependencies, kwargs...)

    return event
end
