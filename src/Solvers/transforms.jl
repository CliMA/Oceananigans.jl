abstract type AbstractTransformDirection end

struct Forward <: AbstractTransformDirection end
struct Backward <: AbstractTransformDirection end

struct Transform{P, D, N, T}
               plan :: P
          direction :: D
      normalization :: N
            twiddle :: T
end

normalization_factor(arch, topo, direction, N) = 1

# FFTW.REDFT01 needs to be normalized by 1/2N.
# See: http://www.fftw.org/fftw3_doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html#g_t1d-Real_002deven-DFTs-_0028DCTs_0029
normalization_factor(::CPU, ::Bounded, ::Backward, N) = 1/(2N)

twiddle_factors(arch, grid, dim) = nothing

# GPU DCTs need twiddle factors.
function twiddle_factors(arch::GPU, grid, dims)
    length(dims) > 1 && return nothing
    dim = dims[1]

    topo = topology(grid)
    topo[dim] != Bounded && return nothing

    Ns = size(grid)
    N = Ns[dim]

    inds⁺ = reshape(0:N-1, reshaped_size(N, dim)...)
    inds⁻ = reshape(0:-1:-(N-1), reshaped_size(N, dim)...)

    ω_4N⁺ = ω.(4N, inds⁺)
    ω_4N⁻ = ω.(4N, inds⁻)

    # The zeroth coefficient of the IDCT (DCT-III or FFTW.REDFT01)
    # is not multiplied by 2.
    ω_4N⁻[1] *= 1/2

    twiddle_factors = (
        forward = arch_array(arch, ω_4N⁺),
        backward = arch_array(arch, ω_4N⁻)
    )

    return twiddle_factors
end

function Transform(plan, direction, arch, grid, dims)
    isnothing(plan) && return Transform(nothing, nothing, nothing, nothing)

    N = size(grid)
    topo = topology(grid)
    normalization = prod(normalization_factor(arch, topo[d](), direction, N[d]) for d in dims)
    twiddle = twiddle_factors(arch, grid, dims)

    @show topo, dims
    @show twiddle

    return Transform(plan, direction, normalization, twiddle)
end

(transform::Transform{<:Nothing})(A) = nothing

function (transform::Transform)(A)
    if transform.direction isa Backward && !isnothing(transform.twiddle)
        @info "backward twiddle"
        @show size(transform.twiddle.backward)
        @. A *= transform.twiddle.backward
    end

    transform.plan * A

    if transform.direction isa Forward && !isnothing(transform.twiddle)
        @info "forward twiddle"
        @show size(transform.twiddle.forward)
        @. A = 2 * real(transform.twiddle.forward * A)
    end

    # Avoid a kernel launch if possible.
    if transform.normalization != 1
        @show transform.normalization
        @. A *= transform.normalization
    end

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

function plan_transforms(arch, grid, storage, planner_flag)
    topo = topology(grid)
    periodic_dims = findall(t -> t == Periodic, topo)
    bounded_dims = findall(t -> t == Bounded, topo)

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

    transforms = (forward = forward_transforms, backward = backward_transforms)

    buffer_needed = false

    return transforms, buffer_needed
end
