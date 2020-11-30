struct Transform{P}
               plan :: P
#    twiddle_factors :: T
end

# function Transform(plan, arch, grid, dims)
#     isnothing(plan) && return Transform(nothing, nothing)

#     if arch isa CPU
#         twiddle_factors = nothing
#     elseif arch isa GPU && topo isa Bounded
#         Ns = size(grid)
#         N = Ns[dim]


#         inds⁺ = reshape()

#         ky⁺ = reshape(0:Ny-1,       1, Ny, 1)
#         kz⁺ = reshape(0:Nz-1,       1, 1, Nz)
#         ky⁻ = reshape(0:-1:-(Ny-1), 1, Ny, 1)
#         kz⁻ = reshape(0:-1:-(Nz-1), 1, 1, Nz)

#         ω_4Ny⁺ = ω.(4Ny, ky⁺) |> CuArray
#         ω_4Nz⁺ = ω.(4Nz, kz⁺) |> CuArray
#         ω_4Ny⁻ = ω.(4Ny, ky⁻) |> CuArray
#         ω_4Nz⁻ = ω.(4Nz, kz⁻) |> CuArray
#         reshaped_size(N, dim)
#         twiddle_factors = (
#             forward =
#             backward =
#         )
# end

(transform::Transform{<:Nothing})(A) = nothing

function (transform::Transform)(A)
    transform.plan * A
    return nothing
end

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

function plan_forward_transform(A::CuArray, topo, dims, planner_flag)
    length(dims) == 0 && return nothing
    return plan_fft!(A, dims)
end

function plan_backward_transform(A::CuArray, topo, dims, planner_flag)
    length(dims) == 0 && return nothing
    return plan_ifft!(A, dims)
end

function plan_transforms(arch, topo, storage, planner_flag)
    periodic_dims = findall(t -> t == Periodic, topo)
    bounded_dims = findall(t -> t == Bounded, topo)

    forward_transforms = (
        periodic = plan_forward_transform(storage, Periodic(), periodic_dims, planner_flag) |> Transform,
        bounded = plan_forward_transform(storage, Bounded(), bounded_dims, planner_flag) |> Transform
    )

    backward_transforms = (
        periodic = plan_backward_transform(storage, Periodic(), periodic_dims, planner_flag) |> Transform,
        bounded = plan_backward_transform(storage, Bounded(), bounded_dims, planner_flag) |> Transform
    )

    transforms = (forward = forward_transforms, backward = backward_transforms)

    buffer_needed = false

    return transforms, buffer_needed
end
