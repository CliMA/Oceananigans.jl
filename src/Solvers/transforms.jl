abstract type AbstractTransformDirection end

struct Forward <: AbstractTransformDirection end
struct Backward <: AbstractTransformDirection end

struct DiscreteTransform{P, A, G, D, Δ, Ω, N, T, Σ}
              plan :: P
      architecture :: A
              grid :: G
         direction :: D
              dims :: Δ
          topology :: Ω
     normalization :: N
           twiddle :: T
    transpose_dims :: Σ
end

#####
##### Normalization factors
#####

normalization_factor(arch, topo, direction, N) = 1

"""
    normalization_factor(::CPU, ::Bounded, ::Backward, N)

`FFTW.REDFT01` needs to be normalized by 1/2N.
See: http://www.fftw.org/fftw3_doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html#g_t1d-Real_002deven-DFTs-_0028DCTs_0029
"""
normalization_factor(::CPU, ::Bounded, ::Backward, N) = 1/(2N)

#####
##### Twiddle factors
#####

twiddle_factors(arch, grid, dim) = nothing

"""
    twiddle_factors(arch::GPU, grid, dims)

Twiddle factored are needed to perform DCTs on the GPU. See equations (19a) and (22) of [Makhoul80](@cite)
for the forward and backward twiddle factors respectively.
"""
function twiddle_factors(arch::GPU, grid, dims)
    # We only perform 1D DCTs.
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

#####
##### Discrete transforms
#####

NoTransform() = DiscreteTransform([nothing for _ in fieldnames(DiscreteTransform)]...)

function DiscreteTransform(plan, direction, arch, grid, dims)
    isnothing(plan) && return NoTransform()

    N = size(grid)
    topo = topology(grid)
    normalization = prod(normalization_factor(arch, topo[d](), direction, N[d]) for d in dims)
    twiddle = twiddle_factors(arch, grid, dims)
    transpose = arch isa GPU && dims == [2] ? (2, 1, 3) : nothing

    topo = [topology(grid)[d]() for d in dims]
    topo = length(topo) == 1 ? topo[1] : topo

    dims = length(dims) == 1 ? dims[1] : dims

    return DiscreteTransform(plan, arch, grid, direction, dims, topo, normalization, twiddle, transpose)
end

(transform::DiscreteTransform{<:Nothing})(A, B) = nothing

function (transform::DiscreteTransform)(A, B)
    if transform.direction isa Backward && !isnothing(transform.twiddle)
        @. A *= transform.twiddle.backward
    end

    @show typeof(A)
    @show typeof(B)

    if transform.direction isa Forward && transform.architecture isa GPU && transform.topology isa Bounded
        @info "Permuting!"
        permute_indices!(B, A, transform.architecture, transform.grid, transform.dims)
        copyto!(A, B)
    end

    if !isnothing(transform.transpose_dims)
        @info "Transposing!"
        permutedims!(B, A, transform.transpose_dims)
        transform.plan * B
        permutedims!(A, B, transform.transpose_dims)
    else
        transform.plan * A
    end

    if transform.direction isa Forward && !isnothing(transform.twiddle)
        @. A = 2 * real(transform.twiddle.forward * A)
    end

    if transform.direction isa Backward && transform.architecture isa GPU && transform.topology isa Bounded
        @info "Unpermuting!"
        unpermute_indices!(B, A, transform.architecture, transform.grid, transform.dims)
        copyto!(A, B)
        @. A = real(A)
    end

    # Avoid a tiny kernel launch if possible.
    if transform.normalization != 1
        @. A *= transform.normalization
    end

    return nothing
end

# TODO:
# Dispatch on Forward/Backward
# apply_twiddle!, apply_normalization!, etc.
