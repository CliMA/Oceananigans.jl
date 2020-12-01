#=
These functions return the transforms required to solve Poisson's equation with
periodic boundary conditions or staggered Neumann boundary.

Fast Fourier transforms (FFTs) are used in the periodic dimensions and
real-to-real discrete cosine transforms are used in the wall-bounded dimensions.
Note that the DCT-II is used for the DCT and the DCT-III for the IDCT
which correspond to REDFT10 and REDFT01 in FFTW.

They operatore on an array with the shape of `A`, which is needed to plan
efficient transforms. `A` will be mutated.
=#

function plan_forward_transform(A::Array, ::Periodic, dims, planner_flag=FFTW.PATIENT)
    length(dims) == 0 && return nothing
    return FFTW.plan_fft!(A, dims, flags=planner_flag)
end

function plan_forward_transform(A::Array, ::Bounded, dims, planner_flag=FFTW.PATIENT)
    length(dims) == 0 && return nothing
    return FFTW.plan_r2r!(A, FFTW.REDFT10, dims, flags=planner_flag)
end

function plan_backward_transform(A::Array, ::Periodic, dims, planner_flag=FFTW.PATIENT)
    length(dims) == 0 && return nothing
    return FFTW.plan_ifft!(A, dims, flags=planner_flag)
end

function plan_backward_transform(A::Array, ::Bounded, dims, planner_flag=FFTW.PATIENT)
    length(dims) == 0 && return nothing
    return FFTW.plan_r2r!(A, FFTW.REDFT01, dims, flags=planner_flag)
end

non_batched_topologies = ((Periodic, Bounded, Periodic),
                          (Periodic, Bounded, Bounded),
                          (Bounded, Periodic, Bounded),
                          (Bounded, Bounded, Bounded))

function plan_forward_transform(A::CuArray, topo, dims, planner_flag)
    length(dims) == 0 && return nothing
    return CUDA.CUFFT.plan_fft!(A, dims)
end

function plan_backward_transform(A::CuArray, topo, dims, planner_flag)
    length(dims) == 0 && return nothing
    return CUDA.CUFFT.plan_ifft!(A, dims)
end

function plan_transforms(arch, grid, storage, planner_flag)
    topo = topology(grid)
    periodic_dims = findall(t -> t == Periodic, topo)
    bounded_dims = findall(t -> t == Bounded, topo)

    if arch isa GPU && topo == (Periodic, Bounded, Bounded)
        forward_plan_x = plan_forward_transform(storage, Periodic(), [1], planner_flag)
        forward_plan_y = plan_forward_transform(storage, Bounded(),  [1], planner_flag)
        forward_plan_z = plan_forward_transform(storage, Bounded(),  [3], planner_flag)

        forward_transforms = (
            x = Transform(forward_plan_x, Forward(), arch, grid, [1]),
            y = Transform(forward_plan_y, Forward(), arch, grid, [2]),
            z = Transform(forward_plan_z, Forward(), arch, grid, [3])
        )

        backward_plan_x = plan_backward_transform(storage, Periodic(), [1], planner_flag)
        backward_plan_y = plan_backward_transform(storage, Bounded(),  [1], planner_flag)
        backward_plan_z = plan_backward_transform(storage, Bounded(),  [3], planner_flag)

        backward_transforms = (
            x = Transform(backward_plan_x, Backward(), arch, grid, [1]),
            y = Transform(backward_plan_y, Backward(), arch, grid, [2]),
            z = Transform(backward_plan_z, Backward(), arch, grid, [3])
        )

        buffer_needed = true
    else
        forward_periodic_plan = plan_forward_transform(storage, Periodic(), periodic_dims, planner_flag)
        forward_bounded_plan = plan_forward_transform(storage, Bounded(), bounded_dims, planner_flag)

        forward_transforms = (
            periodic = Transform(forward_periodic_plan, Forward(), arch, grid, periodic_dims),
            bounded = Transform(forward_bounded_plan, Forward(), arch, grid, bounded_dims)
        )

        backward_periodic_plan = plan_backward_transform(storage, Periodic(), periodic_dims, planner_flag)
        backward_bounded_plan = plan_backward_transform(storage, Bounded(), bounded_dims, planner_flag)

        backward_transforms = (
            periodic = Transform(backward_periodic_plan, Backward(), arch, grid, periodic_dims),
            bounded = Transform(backward_bounded_plan, Backward(), arch, grid, bounded_dims)
        )

        buffer_needed = false
    end

    transforms = (forward = forward_transforms, backward = backward_transforms)

    return transforms, buffer_needed
end