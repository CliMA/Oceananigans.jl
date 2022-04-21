#####
##### Utilities for launching kernels
#####

using KernelAbstractions
using Oceananigans.Architectures
using Oceananigans.Grids

flatten_reduced_dimensions(worksize, dims) = Tuple(i ∈ dims ? 1 : worksize[i] for i = 1:3)

function heuristic_workgroup(Wx, Wy, Wz=nothing)

    workgroup = Wx == 1 && Wy == 1 ?

                    # One-dimensional column models:
                    (1, 1) :

                Wx == 1 ?

                    # Two-dimensional y-z slice models:
                    (1, min(256, Wy)) :

                Wy == 1 ?

                    # Two-dimensional x-z slice models:
                    (1, min(256, Wx)) :

                    # Three-dimensional models
                    (16, 16)

    return workgroup
end

function work_layout(grid, worksize::Tuple; kwargs...)
    workgroup = heuristic_workgroup(worksize...)
    return workgroup, worksize
end

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
function work_layout(grid, workdims::Symbol; include_right_boundaries=false, location=nothing, reduced_dimensions=())

    Nx′, Ny′, Nz′ = include_right_boundaries ? size(location, grid) : size(grid)
    Nx′, Ny′, Nz′ = flatten_reduced_dimensions((Nx′, Ny′, Nz′), reduced_dimensions)

    workgroup = heuristic_workgroup(Nx′, Ny′, Nz′)

    # Drop omitted dimemsions
    worksize = workdims == :xyz ? (Nx′, Ny′, Nz′) :
               workdims == :xy  ? (Nx′, Ny′) :
               workdims == :xz  ? (Nx′, Nz′) :
               workdims == :yz  ? (Ny′, Nz′) : throw(ArgumentError("Unsupported launch configuration: $workdims"))

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
function launch!(arch, grid, workspec, kernel!, kernel_args...;
                 dependencies = nothing,
                 include_right_boundaries = false,
                 reduced_dimensions = (),
                 location = nothing,
                 kwargs...)

    workgroup, worksize = work_layout(grid, workspec,
                                      include_right_boundaries = include_right_boundaries,
                                      reduced_dimensions = reduced_dimensions,
                                      location = location)

    loop! = kernel!(Architectures.device(arch), workgroup, worksize)

    @debug "Launching kernel $kernel! with worksize $worksize"

    event = loop!(kernel_args...; dependencies=dependencies)

    return event
end

# When dims::Val
@inline launch!(arch, grid, ::Val{workspec}, args...; kwargs...) where workspec =
    launch!(arch, grid, workspec, args...; kwargs...)
