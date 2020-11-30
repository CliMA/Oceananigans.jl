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

# struct Transform
#     transform :: T
#     buffer_needed :: Bool
#     transpose :: T
# end

plan_forward_transform(A::Array, ::Periodic, dims, planner_flag=FFTW.PATIENT) =
    FFTW.plan_fft!(A, dims, flags=planner_flag)

plan_forward_transform(A::Array, ::Bounded, dims, planner_flag=FFTW.PATIENT) =
    FFTW.plan_r2r!(A, FFTW.REDFT10, dims, flags=planner_flag)

plan_backward_transform(A::Array, ::Periodic, dims, planner_flag=FFTW.PATIENT) =
    FFTW.plan_ifft!(A, dims, flags=planner_flag)

plan_backward_transform(A::Array, ::Bounded, dims, planner_flag=FFTW.PATIENT) =
    FFTW.plan_r2r!(A, FFTW.REDFT01, dims, flags=planner_flag)

plan_forward_transform(A::CuArray, topo, dims) = plan_fft!(A, dims)
plan_backward_transform(A::CuArray, topo, dims) = plan_ifft!(A, dims)

function plan_transforms(::CPU, topo, storage, planner_flag)
    periodic_dims = findall(t -> t == Periodic, topo)
    bounded_dims = findall(t -> t == Bounded, topo)

    forward_transforms = (

        periodic = length(periodic_dims) > 0 ?
            plan_forward_transform(storage, Periodic(), periodic_dims, planner_flag) : nothing,

        bounded = length(bounded_dims) > 0 ?
            plan_forward_transform(storage, Bounded(), bounded_dims, planner_flag) : nothing
    )

    backward_transforms = (

        periodic = length(periodic_dims) > 0 ?
            plan_backward_transform(storage, Periodic(), periodic_dims, planner_flag) : nothing,

        bounded = length(bounded_dims) > 0 ?
            plan_backward_transform(storage, Bounded(), bounded_dims, planner_flag) : nothing
    )

    transforms = (forward = forward_transforms, backward = backward_transforms)

    buffer_needed = false

    return transforms, buffer_needed
end
