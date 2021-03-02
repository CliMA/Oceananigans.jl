#####
##### These functions return the transforms required to solve Poisson's equation with
##### periodic boundary conditions or staggered Neumann boundary conditions.
#####
##### Fast Fourier transforms (FFTs) are used in the periodic dimensions and
##### real-to-real discrete cosine transforms are used in the wall-bounded dimensions.
##### Note that the DCT-II is used for the DCT and the DCT-III for the IDCT
##### which correspond to REDFT10 and REDFT01 in FFTW.
#####
##### They operate on an array with the shape of `A`, which is needed to plan
##### efficient transforms. `A` will be mutated.
#####

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
                          (Bounded, Bounded, Periodic),
                          (Bounded, Bounded, Bounded))

function plan_forward_transform(A::CuArray, topo, dims, planner_flag)
    length(dims) == 0 && return nothing
    return CUDA.CUFFT.plan_fft!(A, dims)
end

function plan_backward_transform(A::CuArray, topo, dims, planner_flag)
    length(dims) == 0 && return nothing
    return CUDA.CUFFT.plan_ifft!(A, dims)
end

" Used by FFTBasedPoissonSolver "
function plan_transforms(arch, grid::RegularRectilinearGrid, storage, planner_flag)
    Nx, Ny, Nz = size(grid)
    topo = topology(grid)
    periodic_dims = findall(t -> t == Periodic, topo)
    bounded_dims = findall(t -> t == Bounded, topo)

    if arch isa GPU && topo in non_batched_topologies
        if topo == (Periodic, Bounded, Bounded)
            forward_plan_x = plan_forward_transform(storage, Periodic(), [1], planner_flag)
            forward_plan_y = plan_forward_transform(reshape(storage, (Ny, Nx, Nz)), Bounded(),  [1], planner_flag)
            forward_plan_z = plan_forward_transform(storage, Bounded(),  [3], planner_flag)

            backward_plan_x = plan_backward_transform(storage, Periodic(), [1], planner_flag)
            backward_plan_y = plan_backward_transform(reshape(storage, (Ny, Nx, Nz)), Bounded(),  [1], planner_flag)
            backward_plan_z = plan_backward_transform(storage, Bounded(),  [3], planner_flag)

            forward_transforms = (
                DiscreteTransform(forward_plan_z, Forward(), arch, grid, [3]),
                DiscreteTransform(forward_plan_y, Forward(), arch, grid, [2]),
                DiscreteTransform(forward_plan_x, Forward(), arch, grid, [1])
            )

            backward_transforms = (
                DiscreteTransform(backward_plan_x, Backward(), arch, grid, [1]),
                DiscreteTransform(backward_plan_y, Backward(), arch, grid, [2]),
                DiscreteTransform(backward_plan_z, Backward(), arch, grid, [3])
            )

        elseif topo == (Periodic, Bounded, Periodic)
            forward_plan_x = plan_forward_transform(storage, Periodic(), [1], planner_flag)
            forward_plan_y = plan_forward_transform(reshape(storage, (Ny, Nx, Nz)), Bounded(),  [1], planner_flag)
            forward_plan_z = plan_forward_transform(storage, Periodic(), [3], planner_flag)

            backward_plan_x = plan_backward_transform(storage, Periodic(), [1], planner_flag)
            backward_plan_y = plan_backward_transform(reshape(storage, (Ny, Nx, Nz)), Bounded(),  [1], planner_flag)
            backward_plan_z = plan_backward_transform(storage, Periodic(), [3], planner_flag)

            forward_transforms = (
                DiscreteTransform(forward_plan_y, Forward(), arch, grid, [2]),
                DiscreteTransform(forward_plan_x, Forward(), arch, grid, [1]),
                DiscreteTransform(forward_plan_z, Forward(), arch, grid, [3])
            )

            backward_transforms = (
                DiscreteTransform(backward_plan_z, Backward(), arch, grid, [3]),
                DiscreteTransform(backward_plan_x, Backward(), arch, grid, [1]),
                DiscreteTransform(backward_plan_y, Backward(), arch, grid, [2])
            )

        elseif topo == (Bounded, Periodic, Bounded)
            forward_plan_x = plan_forward_transform(storage, Bounded(), [1], planner_flag)
            forward_plan_y = plan_forward_transform(reshape(storage, (Ny, Nx, Nz)), Periodic(), [1], planner_flag)
            forward_plan_z = plan_forward_transform(storage, Bounded(), [3], planner_flag)

            backward_plan_x = plan_backward_transform(storage, Bounded(), [1], planner_flag)
            backward_plan_y = plan_backward_transform(reshape(storage, (Ny, Nx, Nz)), Periodic(),  [1], planner_flag)
            backward_plan_z = plan_backward_transform(storage, Bounded(), [3], planner_flag)

            forward_transforms = (
                DiscreteTransform(forward_plan_x, Forward(), arch, grid, [1]),
                DiscreteTransform(forward_plan_z, Forward(), arch, grid, [3]),
                DiscreteTransform(forward_plan_y, Forward(), arch, grid, [2])
            )

            backward_transforms = (
                DiscreteTransform(backward_plan_y, Backward(), arch, grid, [2]),
                DiscreteTransform(backward_plan_x, Backward(), arch, grid, [1]),
                DiscreteTransform(backward_plan_z, Backward(), arch, grid, [3])
            )

        elseif topo == (Bounded, Bounded, Periodic)
            forward_plan_x = plan_forward_transform(storage, Bounded(),  [1], planner_flag)
            forward_plan_y = plan_forward_transform(reshape(storage, (Ny, Nx, Nz)), Bounded(),  [1], planner_flag)
            forward_plan_z = plan_forward_transform(storage, Periodic(), [3], planner_flag)

            backward_plan_x = plan_backward_transform(storage, Bounded(),  [1], planner_flag)
            backward_plan_y = plan_backward_transform(reshape(storage, (Ny, Nx, Nz)), Bounded(),  [1], planner_flag)
            backward_plan_z = plan_backward_transform(storage, Periodic(), [3], planner_flag)

            forward_transforms = (
                DiscreteTransform(forward_plan_x, Forward(), arch, grid, [1]),
                DiscreteTransform(forward_plan_y, Forward(), arch, grid, [2]),
                DiscreteTransform(forward_plan_z, Forward(), arch, grid, [3])
            )

            backward_transforms = (
                DiscreteTransform(backward_plan_z, Backward(), arch, grid, [3]),
                DiscreteTransform(backward_plan_x, Backward(), arch, grid, [1]),
                DiscreteTransform(backward_plan_y, Backward(), arch, grid, [2])
            )

        elseif topo == (Bounded, Bounded, Bounded)
            forward_plan_x = plan_forward_transform(storage, Bounded(), [1], planner_flag)
            forward_plan_y = plan_forward_transform(reshape(storage, (Ny, Nx, Nz)), Bounded(), [1], planner_flag)
            forward_plan_z = plan_forward_transform(storage, Bounded(), [3], planner_flag)

            backward_plan_x = plan_backward_transform(storage, Bounded(), [1], planner_flag)
            backward_plan_y = plan_backward_transform(reshape(storage, (Ny, Nx, Nz)), Bounded(),  [1], planner_flag)
            backward_plan_z = plan_backward_transform(storage, Bounded(), [3], planner_flag)

            forward_transforms = (
                DiscreteTransform(forward_plan_x, Forward(), arch, grid, [1]),
                DiscreteTransform(forward_plan_y, Forward(), arch, grid, [2]),
                DiscreteTransform(forward_plan_z, Forward(), arch, grid, [3])
            )

            backward_transforms = (
                DiscreteTransform(backward_plan_x, Backward(), arch, grid, [1]),
                DiscreteTransform(backward_plan_y, Backward(), arch, grid, [2]),
                DiscreteTransform(backward_plan_z, Backward(), arch, grid, [3])
            )
        end

    else
        # This is the case where batching transforms is possible. It's always possible on the CPU
        # since FFTW is awesome so it includes all topologies on the CPU.
        #
        # On the GPU batching is possible when the topology is not one of non_batched_topologies
        # (where an FFT is needed along dimension 2), so it includes (Periodic, Periodic, Periodic),
        # (Periodic, Periodic, Bounded), and (Bounded, Periodic, Periodic).

        forward_periodic_plan = plan_forward_transform(storage, Periodic(), periodic_dims, planner_flag)
        forward_bounded_plan = plan_forward_transform(storage, Bounded(), bounded_dims, planner_flag)

        forward_transforms = (
            DiscreteTransform(forward_bounded_plan, Forward(), arch, grid, bounded_dims),
            DiscreteTransform(forward_periodic_plan, Forward(), arch, grid, periodic_dims)
        )

        backward_periodic_plan = plan_backward_transform(storage, Periodic(), periodic_dims, planner_flag)
        backward_bounded_plan = plan_backward_transform(storage, Bounded(), bounded_dims, planner_flag)

        backward_transforms = (
            DiscreteTransform(backward_periodic_plan, Backward(), arch, grid, periodic_dims),
            DiscreteTransform(backward_bounded_plan, Backward(), arch, grid, bounded_dims)
        )
    end

    transforms = (forward = forward_transforms, backward = backward_transforms)

    return transforms
end

" Used by FourierTridiagonalPoissonSolver "
function plan_transforms(arch, grid::VerticallyStretchedRectilinearGrid, storage, planner_flag)
    Nx, Ny, Nz = size(grid)
    TX, TY, TZ = topo = topology(grid)

    # Limit ourselves to x, y transforms (z uses a tridiagonal solve)
    periodic_dims = findall(t -> t == Periodic, (TX, TY))
    bounded_dims = findall(t -> t == Bounded, (TX, TY))

    if arch isa GPU && topo in non_batched_topologies
        if (TX, TY) == (Periodic, Bounded)
            forward_plan_x = plan_forward_transform(storage, Periodic(), [1], planner_flag)
            forward_plan_y = plan_forward_transform(reshape(storage, (Ny, Nx, Nz)), Bounded(), [1], planner_flag)

            backward_plan_x = plan_backward_transform(storage, Periodic(), [1], planner_flag)
            backward_plan_y = plan_backward_transform(reshape(storage, (Ny, Nx, Nz)), Bounded(),  [1], planner_flag)

            forward_transforms = (
                DiscreteTransform(forward_plan_y, Forward(), arch, grid, [2]),
                DiscreteTransform(forward_plan_x, Forward(), arch, grid, [1])
            )

            backward_transforms = (
                DiscreteTransform(backward_plan_x, Backward(), arch, grid, [1]),
                DiscreteTransform(backward_plan_y, Backward(), arch, grid, [2])
            )

        elseif (TX, TY) == (Bounded, Periodic)
            forward_plan_x = plan_forward_transform(storage, Bounded(), [1], planner_flag)
            forward_plan_y = plan_forward_transform(reshape(storage, (Ny, Nx, Nz)), Periodic(), [1], planner_flag)

            backward_plan_x = plan_backward_transform(storage, Bounded(), [1], planner_flag)
            backward_plan_y = plan_backward_transform(reshape(storage, (Ny, Nx, Nz)), Periodic(),  [1], planner_flag)

            forward_transforms = (
                DiscreteTransform(forward_plan_x, Forward(), arch, grid, [1]),
                DiscreteTransform(forward_plan_y, Forward(), arch, grid, [2])
            )

            backward_transforms = (
                DiscreteTransform(backward_plan_y, Backward(), arch, grid, [2]),
                DiscreteTransform(backward_plan_x, Backward(), arch, grid, [1])
            )

        elseif (TX, TY) == (Bounded, Bounded)
            forward_plan_x = plan_forward_transform(storage, Bounded(), [1], planner_flag)
            forward_plan_y = plan_forward_transform(reshape(storage, (Ny, Nx, Nz)), Bounded(), [1], planner_flag)

            backward_plan_x = plan_backward_transform(storage, Bounded(), [1], planner_flag)
            backward_plan_y = plan_backward_transform(reshape(storage, (Ny, Nx, Nz)), Bounded(),  [1], planner_flag)

            forward_transforms = (
                DiscreteTransform(forward_plan_x, Forward(), arch, grid, [1]),
                DiscreteTransform(forward_plan_y, Forward(), arch, grid, [2])
            )

            backward_transforms = (
                DiscreteTransform(backward_plan_x, Backward(), arch, grid, [1]),
                DiscreteTransform(backward_plan_y, Backward(), arch, grid, [2])
            )
        end
    else
        # This is the case where batching transforms is possible. It's always possible on the CPU
        # since FFTW is awesome so it includes all topologies on the CPU.
        #
        # On the GPU batching is possible when the topology is not one of non_batched_topologies
        # (where an FFT is needed along dimension 2), so it includes (Periodic, Periodic, Periodic),
        # (Periodic, Periodic, Bounded), and (Bounded, Periodic, Periodic).

        forward_periodic_plan = plan_forward_transform(storage, Periodic(), periodic_dims, planner_flag)
        forward_bounded_plan = plan_forward_transform(storage, Bounded(), bounded_dims, planner_flag)

        forward_transforms = (
            DiscreteTransform(forward_bounded_plan, Forward(), arch, grid, bounded_dims),
            DiscreteTransform(forward_periodic_plan, Forward(), arch, grid, periodic_dims)
        )

        backward_periodic_plan = plan_backward_transform(storage, Periodic(), periodic_dims, planner_flag)
        backward_bounded_plan = plan_backward_transform(storage, Bounded(), bounded_dims, planner_flag)

        backward_transforms = (
            DiscreteTransform(backward_periodic_plan, Backward(), arch, grid, periodic_dims),
            DiscreteTransform(backward_bounded_plan, Backward(), arch, grid, bounded_dims)
        )
    end

    transforms = (forward = forward_transforms, backward = backward_transforms)

    return transforms
end
