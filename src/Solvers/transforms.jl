abstract type AbstractTransformDirection end

struct Forward <: AbstractTransformDirection end
struct Backward <: AbstractTransformDirection end

struct Transform{P, D, N, T, R}
               plan :: P
          direction :: D
      normalization :: N
            twiddle :: T
          transpose :: R
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
    isnothing(plan) && return Transform{Nothing,Nothing,Nothing,Nothing,Nothing}(nothing, nothing, nothing, nothing, nothing)

    N = size(grid)
    topo = topology(grid)
    normalization = prod(normalization_factor(arch, topo[d](), direction, N[d]) for d in dims)
    twiddle = twiddle_factors(arch, grid, dims)
    transpose = arch isa GPU && dims == [2] ? (2, 1, 3) : nothing

    return Transform{typeof(plan),typeof(direction),typeof(normalization),typeof(twiddle),typeof(transpose)}(
        plan, direction, normalization, twiddle, transpose)
end

(transform::Transform{<:Nothing})(A, B) = nothing

function (transform::Transform)(A, B)
    if transform.direction isa Backward && !isnothing(transform.twiddle)
        @. A *= transform.twiddle.backward
    end

    if !isnothing(transform.transpose)
        permutedims!(B, A, transform.transpose)
        transform.plan * B
        permutedims!(A, B, transform.transpose)
    else
        transform.plan * A
    end

    if transform.direction isa Forward && !isnothing(transform.twiddle)
        @. A = 2 * real(transform.twiddle.forward * A)
    end

    # Avoid a kernel launch if possible.
    if transform.normalization != 1
        @. A *= transform.normalization
    end

    return nothing
end
