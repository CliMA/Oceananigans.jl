#####
##### Utilities for launching kernels
#####

using KernelAbstractions
using Oceananigans.Architectures
using Oceananigans.Grids

flatten_reduced_dimensions(worksize, dims) = Tuple(i ∈ dims ? 1 : worksize[i] for i = 1:3)

const MAX_GPU_THREADS_PER_BLOCK = 256

function heuristic_workgroup(::GPU, Nx, Ny, Nz)
    # One-dimensional column models:
    Nx == 1 && Ny == 1 && return (1, 1)

    # Two-dimensional y-z slice models:
    Nx == 1 && return (1, min(MAX_GPU_THREADS_PER_BLOCK, Ny))

    # Two-dimensional x-z slice models:
    Ny == 1 && return (1, min(MAX_GPU_THREADS_PER_BLOCK, Nx))

    # Three-dimensional GPU models
    return (Int(√MAX_GPU_THREADS_PER_BLOCK), Int(√MAX_GPU_THREADS_PER_BLOCK))
end

function heuristic_workgroup(::CPU, Nx, Ny, Nz)
    z_group_size = 1 # full partition along outer dimension by default
    z_range = Nz === 1 ? 0 : Nz # collapse z range and fully partition along (x, y) if Nz=1

    # We devote Nz threads to the outer loop. If Nz < Nthreads,
    # we further divide the iteration amongst the second dimension (y).
    # Examples:
    # ndrange = (Nx, Ny, 8) and nthreads = 4 implies group_size = (Nx, Ny, 1)
    # ndrange = (Nx, Ny, 2) and nthreads = 4 implies group_size = (Nx, Ny/2, 1)
    # ndrange = (Nx, Ny, 1) and nthreads = 4 implies group_size = (Nx, Ny/4, 1)
    y_residual_threads = max(1, Nthreads - z_range)
    y_group_size = max(1, floor(Int, Ny / y_residual_threads))
    y_range = Ny === 1 ? 0 : Ny # collapse y-partition partition the remainder of threads in x if Ny=1.

    # Further partition the problem along the x direction if the
    # direction is small.
    x_residual_threads = max(1, y_residual_threads - y_range)
    x_group_size = max(1, floor(Int, Nx / y_residual_threads))

    return (x_group_size, y_group_size) # padded with 1 by default
end

"""
    work_layout(arch, grid, dims; include_right_boundaries=false, location=nothing)

Returns the `workgroup` and `worksize` for launching a kernel over `dims`
on `grid` and `arch`itecture. The `workgroup` is a tuple specifying the threads per block in each
dimension. The `worksize` specifies the range of the loop in each dimension.

Specifying `include_right_boundaries=true` will ensure the work layout includes the
right face end points along bounded dimensions. This requires the field `location`
to be specified.

For more information, see: https://github.com/CliMA/Oceananigans.jl/pull/308
"""
function work_layout(arch, grid, dims; include_right_boundaries=false, location=nothing, reduced_dimensions=())

    Nx, Ny, Nz = size(grid)

    workgroup = heuristic_workgroup(arch, Nx, Ny, Nz)
   
    Nx′, Ny′, Nz′ = include_right_boundaries ? size(location, grid) : (Nx, Ny, Nz)

    Nx′, Ny′, Nz′ = flatten_reduced_dimensions((Nx′, Ny′, Nz′), reduced_dimensions)

    worksize = dims == :xyz ? (Nx′, Ny′, Nz′) :
               dims == :xy  ? (Nx′, Ny′) :
               dims == :xz  ? (Nx′, Nz′) :
               dims == :yz  ? (Ny′, Nz′) : throw(ArgumentError("Unsupported launch configuration: $dims"))

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
function launch!(arch, grid, dims, kernel!, args...;
                 dependencies = nothing,
                 include_right_boundaries = false,
                 reduced_dimensions = (),
                 location = nothing,
                 kwargs...)

    workgroup, worksize = work_layout(arch, grid, dims,
                                      include_right_boundaries = include_right_boundaries,
                                      reduced_dimensions = reduced_dimensions,
                                      location = location)

    loop! = kernel!(Architectures.device(arch), workgroup, worksize)

    @debug "Launching kernel $kernel! with worksize $worksize"

    event = loop!(args...; dependencies=dependencies, kwargs...)

    return event
end

# When dims::Val
@inline launch!(arch, grid, ::Val{dims}, args...; kwargs...) where dims = launch!(arch, grid, dims, args...; kwargs...)
