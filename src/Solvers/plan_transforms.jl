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

using Oceananigans.Grids: XYRegularRG, XZRegularRG, YZRegularRG, XYZRegularRG, regular_dimensions, stretched_dimensions

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
function plan_transforms(grid::XYZRegularRG, storage, planner_flag)
    Nx, Ny, Nz = size(grid)
    topo = topology(grid)
    periodic_dims = findall(t -> t == Periodic, topo)
    bounded_dims = findall(t -> t == Bounded, topo)

    # Convert Flat to Bounded for inferring batchability and transform ordering
    # Note that transforms are omitted in Flat directions.
    unflattened_topo = Tuple(T() isa Flat ? Bounded : T for T in topo)

    arch = architecture(grid)

    if arch isa GPU && !(unflattened_topo in batchable_GPU_topologies)

        reshaped_storage = reshape(storage, (Ny, Nx, Nz))
        forward_plan_x = plan_forward_transform(storage,          topo[1](), [1], planner_flag)
        forward_plan_y = plan_forward_transform(reshaped_storage, topo[2](), [1], planner_flag)
        forward_plan_z = plan_forward_transform(storage,          topo[3](), [3], planner_flag)

        backward_plan_x = plan_backward_transform(storage,          topo[1](), [1], planner_flag)
        backward_plan_y = plan_backward_transform(reshaped_storage, topo[2](), [1], planner_flag)
        backward_plan_z = plan_backward_transform(storage,          topo[3](), [3], planner_flag)

        forward_plans = (forward_plan_x, forward_plan_y, forward_plan_z)
        backward_plans = (backward_plan_x, backward_plan_y, backward_plan_z)

        f_order = forward_orders(unflattened_topo...)
        b_order = backward_orders(unflattened_topo...)

        forward_transforms = (
            DiscreteTransform(forward_plans[f_order[1]], Forward(), grid, [f_order[1]]),
            DiscreteTransform(forward_plans[f_order[2]], Forward(), grid, [f_order[2]]),
            DiscreteTransform(forward_plans[f_order[3]], Forward(), grid, [f_order[3]])
        )

        backward_transforms = (
            DiscreteTransform(backward_plans[b_order[1]], Backward(), grid, [b_order[1]]),
            DiscreteTransform(backward_plans[b_order[2]], Backward(), grid, [b_order[2]]),
            DiscreteTransform(backward_plans[b_order[3]], Backward(), grid, [b_order[3]])
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
            DiscreteTransform(forward_bounded_plan, Forward(), grid, bounded_dims),
            DiscreteTransform(forward_periodic_plan, Forward(), grid, periodic_dims)
        )

        backward_periodic_plan = plan_backward_transform(storage, Periodic(), periodic_dims, planner_flag)
        backward_bounded_plan = plan_backward_transform(storage, Bounded(), bounded_dims, planner_flag)

        backward_transforms = (
            DiscreteTransform(backward_periodic_plan, Backward(), grid, periodic_dims),
            DiscreteTransform(backward_bounded_plan, Backward(), grid, bounded_dims)
        )
    end

    transforms = (forward=forward_transforms, backward=backward_transforms)

    return transforms
end

""" Used by FourierTridiagonalPoissonSolver. """
function plan_transforms(grid::Union{XYRegularRG, XZRegularRG, YZRegularRG}, storage, planner_flag)
    Nx, Ny, Nz = size(grid)
    topo = topology(grid)

    irreg_dim = stretched_dimensions(grid)[1]
    reg_dims  = regular_dimensions(grid)
    !(topo[irreg_dim] === Bounded) && error("Transforms can be planned only when the stretched direction's topology is `Bounded`.")

    periodic_dims = Tuple( dim for dim in findall(t -> t == Periodic, topo) if dim ≠ irreg_dim )
    bounded_dims  = Tuple( dim for dim in findall(t -> t == Bounded,  topo) if dim ≠ irreg_dim )

    arch = architecture(grid)

    if arch isa CPU
        # This is the case where batching transforms is possible. It's always possible on the CPU
        # since FFTW is awesome so it includes all topologies on the CPU.
        #
        # On the GPU and for vertically Bounded grids, batching is possible either in horizontally-periodic
        # domains, or for domains that are `Bounded, Periodic, Bounded`.

        forward_periodic_plan = plan_forward_transform(storage, Periodic(), periodic_dims, planner_flag)
        forward_bounded_plan  = plan_forward_transform(storage, Bounded(),  bounded_dims,  planner_flag)

        forward_transforms = (DiscreteTransform(forward_bounded_plan,  Forward(), grid, bounded_dims),
                              DiscreteTransform(forward_periodic_plan, Forward(), grid, periodic_dims))

        backward_periodic_plan = plan_backward_transform(storage, Periodic(), periodic_dims, planner_flag)
        backward_bounded_plan  = plan_backward_transform(storage, Bounded(),  bounded_dims,  planner_flag)

        backward_transforms = (DiscreteTransform(backward_periodic_plan, Backward(), grid, periodic_dims),
                               DiscreteTransform(backward_bounded_plan,  Backward(), grid, bounded_dims))

    elseif bounded_dims == ()
        # We're on the GPU and either (Periodic, Periodic), (Flat, Periodic), or
        # (Periodic, Flat) in the regular dimensions. So, we pretend like we need a 2D
        # doubly-periodic transform (even if one dimension is Flat).

        forward_periodic_plan = plan_forward_transform(storage, Periodic(), reg_dims, planner_flag)
        backward_periodic_plan = plan_backward_transform(storage, Periodic(), reg_dims, planner_flag)

        forward_transforms = tuple(DiscreteTransform(forward_periodic_plan, Forward(), grid, reg_dims))
        backward_transforms = tuple(DiscreteTransform(backward_periodic_plan, Backward(), grid, reg_dims))

    else # we are on the GPU and we cannot / should not batch!
        Nx, Ny, Nz = size(grid)
        reshaped_storage = reshape(storage, (Ny, Nx, Nz))

        if irreg_dim == 1
            forward_plan_1 = plan_forward_transform(reshaped_storage, topo[2](), [1], planner_flag)
            forward_plan_2 = plan_forward_transform(storage,          topo[3](), [3], planner_flag)

            backward_plan_1 = plan_backward_transform(reshaped_storage, topo[2](), [1], planner_flag)
            backward_plan_2 = plan_backward_transform(storage,          topo[3](), [3], planner_flag)

        elseif irreg_dim == 2
            forward_plan_1 = plan_forward_transform(storage, topo[1](), [1], planner_flag)
            forward_plan_2 = plan_forward_transform(storage, topo[3](), [3], planner_flag)

            backward_plan_1 = plan_backward_transform(storage, topo[1](), [1], planner_flag)
            backward_plan_2 = plan_backward_transform(storage, topo[3](), [3], planner_flag)

        elseif irreg_dim == 3
            forward_plan_1 = plan_forward_transform(storage,          topo[1](), [1], planner_flag)
            forward_plan_2 = plan_forward_transform(reshaped_storage, topo[2](), [1], planner_flag)

            backward_plan_1 = plan_backward_transform(storage,          topo[1](), [1], planner_flag)
            backward_plan_2 = plan_backward_transform(reshaped_storage, topo[2](), [1], planner_flag)
        end

        forward_plans  = Dict(reg_dims[1] => forward_plan_1,  reg_dims[2] => forward_plan_2)
        backward_plans = Dict(reg_dims[1] => backward_plan_1, reg_dims[2] => backward_plan_2)

        # Transform Flat topologies into Bounded
        unflattened_topo = Tuple(T() isa Flat ? Bounded : T for T in topo)
        f_order = forward_orders(unflattened_topo...)
        b_order = backward_orders(unflattened_topo...)

        # Extract stretched dimension
        f_order = Tuple(f_order[i] for i in findall(d -> d != irreg_dim, f_order))
        b_order = Tuple(b_order[i] for i in findall(d -> d != irreg_dim, b_order))

        forward_transforms = (DiscreteTransform(forward_plans[f_order[1]], Forward(), grid, [f_order[1]]),
                              DiscreteTransform(forward_plans[f_order[2]], Forward(), grid, [f_order[2]]))

        backward_transforms = (DiscreteTransform(backward_plans[b_order[1]], Backward(), grid, [b_order[1]]),
                               DiscreteTransform(backward_plans[b_order[2]], Backward(), grid, [b_order[2]]))
    end

    transforms = (forward=forward_transforms, backward=backward_transforms)

    return transforms
end
