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

function plan_forward_transform(A::CuArray, ::Union{Bounded, Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return CUDA.CUFFT.plan_fft!(A, dims)
end

function plan_backward_transform(A::CuArray, ::Union{Bounded, Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return CUDA.CUFFT.plan_ifft!(A, dims)
end

plan_backward_transform(A::Union{Array, CuArray}, ::Flat, args...) = nothing
plan_forward_transform(A::Union{Array, CuArray}, ::Flat, args...) = nothing

batchable_GPU_topologies = ((Periodic, Periodic, Periodic),
                            (Periodic, Periodic, Bounded),
                            (Bounded,  Periodic, Periodic))

# In principle the order in which the transforms are applied does not matter of course,
# but in practice we want to perform the `Bounded` forward transforms first because on
# the GPU we take the real part after a forward transform, so if the `Periodic`
# transform is performed first we lose the information in the imaginary components after a
# `Bounded` forward transform.
# 
# For the same reason, `Bounded` backward transforms are applied after `Periodic`
# backward transforms.
#
# Note that `Flat` "transforms" have no effect. To avoid defining forward_orders and `backward_orders`
# for Flat we reuse the orderings that apply to combinations of Periodic and Bounded.

forward_orders(::Type{Periodic}, ::Type{Bounded},  ::Type{Bounded})  = (3, 2, 1)
forward_orders(::Type{Periodic}, ::Type{Bounded},  ::Type{Periodic}) = (2, 1, 3)
forward_orders(::Type{Bounded},  ::Type{Periodic}, ::Type{Bounded})  = (1, 3, 2)
forward_orders(::Type{Bounded},  ::Type{Bounded},  ::Type{Periodic}) = (1, 2, 3)
forward_orders(::Type{Bounded},  ::Type{Bounded},  ::Type{Bounded})  = (1, 2, 3)

backward_orders(::Type{Periodic}, ::Type{Bounded},  ::Type{Bounded})  = (1, 2, 3)
backward_orders(::Type{Periodic}, ::Type{Bounded},  ::Type{Periodic}) = (3, 1, 2)
backward_orders(::Type{Bounded},  ::Type{Periodic}, ::Type{Bounded})  = (2, 1, 3)
backward_orders(::Type{Bounded},  ::Type{Bounded},  ::Type{Periodic}) = (3, 1, 2)
backward_orders(::Type{Bounded},  ::Type{Bounded},  ::Type{Bounded})  = (1, 2, 3)

" Used by FFTBasedPoissonSolver "
function plan_transforms(arch, grid::RegularRectilinearGrid, storage, planner_flag)
    Nx, Ny, Nz = size(grid)
    topo = topology(grid)
    periodic_dims = findall(t -> t == Periodic, topo)
    bounded_dims = findall(t -> t == Bounded, topo)

    # Convert Flat to Bounded for inferring batchability and transform ordering
    # Note that transforms are omitted in Flat directions.
    unflattened_topo = (T() isa Flat ? Bounded : T for T in topo)

    if arch isa GPU && !(unflattened_topo in batchable_GPU_topologies)

        rs_storage = reshape(storage, (Ny, Nx, Nz))
        forward_plan_x = plan_forward_transform(storage,    topo[1](), [1], planner_flag)
        forward_plan_y = plan_forward_transform(rs_storage, topo[2](), [1], planner_flag)
        forward_plan_z = plan_forward_transform(storage,    topo[3](), [3], planner_flag)

        backward_plan_x = plan_backward_transform(storage,    topo[1](), [1], planner_flag)
        backward_plan_y = plan_backward_transform(rs_storage, topo[2](), [1], planner_flag)
        backward_plan_z = plan_backward_transform(storage,    topo[3](), [3], planner_flag)

        forward_plans = (forward_plan_x, forward_plan_y, forward_plan_z)
        backward_plans = (backward_plan_x, backward_plan_y, backward_plan_z)

        f_order = forward_orders(unflattened_topo...)
        b_order = backward_orders(unflattened_topo...)

        forward_transforms = (
            DiscreteTransform(forward_plans[f_order[1]], Forward(), arch, grid, [f_order[1]]),
            DiscreteTransform(forward_plans[f_order[2]], Forward(), arch, grid, [f_order[2]]),
            DiscreteTransform(forward_plans[f_order[3]], Forward(), arch, grid, [f_order[3]])
        )

        backward_transforms = (
            DiscreteTransform(backward_plans[b_order[1]], Backward(), arch, grid, [b_order[1]]),
            DiscreteTransform(backward_plans[b_order[2]], Backward(), arch, grid, [b_order[2]]),
            DiscreteTransform(backward_plans[b_order[3]], Backward(), arch, grid, [b_order[3]])
        )

    else
        # This is the case where batching transforms is possible. It's always possible on the CPU
        # since FFTW is awesome so it includes all topologies on the CPU.
        #
        # `batchable_GPU_topologies` occurs when there are two adjacent `Periodic` dimensions:
        # (Periodic, Periodic, Periodic), (Periodic, Periodic, Bounded), and (Bounded, Periodic, Periodic).

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

    # Convert Flat to Bounded for ordering purposes (transforms are omitted in Flat directions anyways)
    unflattened_topo = (T() isa Flat ? Bounded : T for T in topo)

    if arch isa CPU || topo == (Periodic, Periodic, Bounded)
        # This is the case where batching transforms is possible. It's always possible on the CPU
        # since FFTW is awesome so it includes all topologies on the CPU.
        #
        # On the GPU and for vertically Bounded grids, batching is possible when topo == (Periodic, Periodic, Bounded).

        forward_periodic_plan = plan_forward_transform(storage, Periodic(), periodic_dims, planner_flag)
        forward_bounded_plan = plan_forward_transform(storage, Bounded(), bounded_dims, planner_flag)

        forward_transforms = (DiscreteTransform(forward_bounded_plan, Forward(), arch, grid, bounded_dims),
                              DiscreteTransform(forward_periodic_plan, Forward(), arch, grid, periodic_dims))

        backward_periodic_plan = plan_backward_transform(storage, Periodic(), periodic_dims, planner_flag)
        backward_bounded_plan = plan_backward_transform(storage, Bounded(), bounded_dims, planner_flag)

        backward_transforms = (DiscreteTransform(backward_periodic_plan, Backward(), arch, grid, periodic_dims),
                               DiscreteTransform(backward_bounded_plan, Backward(), arch, grid, bounded_dims))

    else # we are on the GPU and we cannot / should not batch
        rs_storage = reshape(storage, (Ny, Nx, Nz))

        forward_plan_x = plan_forward_transform(storage,    topo[1](), [1], planner_flag)
        forward_plan_y = plan_forward_transform(rs_storage, topo[2](), [1], planner_flag)

        backward_plan_x = plan_backward_transform(storage,    topo[1](), [1], planner_flag)
        backward_plan_y = plan_backward_transform(rs_storage, topo[2](), [1], planner_flag)

        forward_plans = (forward_plan_x, forward_plan_y)
        backward_plans = (backward_plan_x, backward_plan_y)

        f_order = forward_orders(unflattened_topo...)
        b_order = backward_orders(unflattened_topo...)

        # Extract third dimension
        f_order = Tuple(f_order[i] for i in findall(d -> d != 3, f_order))
        b_order = Tuple(b_order[i] for i in findall(d -> d != 3, b_order))

        forward_transforms = (DiscreteTransform(forward_plans[f_order[1]], Forward(), arch, grid, [f_order[1]]),
                              DiscreteTransform(forward_plans[f_order[2]], Forward(), arch, grid, [f_order[2]]))

        backward_transforms = (DiscreteTransform(backward_plans[b_order[1]], Backward(), arch, grid, [b_order[1]]),
                               DiscreteTransform(backward_plans[b_order[2]], Backward(), arch, grid, [b_order[2]]))
    end

    transforms = (forward=forward_transforms, backward=backward_transforms)

    return transforms
end
