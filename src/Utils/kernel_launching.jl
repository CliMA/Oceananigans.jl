#####
##### Utilities for launching kernels
#####

using KernelAbstractions
using Oceananigans.Architectures
using Oceananigans.Grids

const MAX_THREADS_PER_BLOCK = 256

flatten_reduced_dimensions(worksize, dims) = Tuple(i ∈ dims ? 1 : worksize[i] for i = 1:3)

"""
    work_layout(grid, dims; include_right_boundaries=false, location=nothing)

Returns the `workgroup` and `worksize` for launching a kernel over `dims`
on `grid`. The `workgroup` is a tuple specifying the threads per block in each
dimension. The `worksize` specifies the range of the loop in each dimension.

Specifying `include_right_boundaries=true` will ensure the work layout includes the
right face end points along bounded dimensions. This requires the field `location`
to be specified.

For more information, see: https://github.com/CliMA/Oceananigans.jl/pull/308
"""
function work_layout(grid, dims; include_right_boundaries=false, location=nothing, reduced_dimensions=())

    Nx, Ny, Nz = size(grid)

    workgroup = Nx == 1 && Ny == 1 ?

                    # One-dimensional column models:
                    (1, 1) :

                Nx == 1 ?

                    # Two-dimensional y-z slice models:
                    (1, min(MAX_THREADS_PER_BLOCK, Ny)) :

                Ny == 1 ?

                    # Two-dimensional x-z slice models:
                    (1, min(MAX_THREADS_PER_BLOCK, Nx)) :

                    # Three-dimensional models
                    (Int(√MAX_THREADS_PER_BLOCK), Int(√MAX_THREADS_PER_BLOCK))

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
                 dependencies = device_event(arch),
                 include_right_boundaries = false,
                 reduced_dimensions = (),
                 location = nothing)

    workgroup, worksize = work_layout(grid, dims,
                                      include_right_boundaries = include_right_boundaries,
                                            reduced_dimensions = reduced_dimensions,
                                                      location = location)

    loop! = kernel!(Architectures.device(arch), workgroup, worksize)

    @debug "Launching kernel $kernel! with worksize $worksize"

    event = loop!(args...; dependencies=dependencies)

    wait(Architectures.device(arch), event)

    return event
end

# When dims::Val
@inline launch!(arch, grid, ::Val{dims}, args...; kwargs...) where dims = launch!(arch, grid, dims, args...; kwargs...)
