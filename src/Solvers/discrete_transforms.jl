import Oceananigans.Architectures: architecture

abstract type AbstractTransformDirection end

struct Forward <: AbstractTransformDirection end
struct Backward <: AbstractTransformDirection end

struct DiscreteTransform{P, D, G, Δ, Ω, N, T, Σ}
               plan :: P
               grid :: G
          direction :: D
               dims :: Δ
           topology :: Ω
      normalization :: N
    twiddle_factors :: T # # https://en.wikipedia.org/wiki/Twiddle_factor
     transpose_dims :: Σ
end

architecture(transform::DiscreteTransform) = architecture(transform.grid)

#####
##### Normalization factors
#####

normalization_factor(arch, topo, direction, N) = 1

"""
    normalization_factor(::CPU, ::Bounded, ::Backward, N)

`FFTW.REDFT01` needs to be normalized by 1/2N.
See: http://www.fftw.org/fftw3_doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html#g_t1d-Real_002deven-DFTs-_0028DCTs_0029
"""
normalization_factor(::CPU, ::Bounded, ::Backward, N) = 1 / 2N

#####
##### Twiddle factors
#####

twiddle_factors(arch, grid, dim) = nothing

"""
    twiddle_factors(arch::GPU, grid, dims)

Twiddle factors are needed to perform DCTs on the GPU. See equations (19a) and (22) of [Makhoul80](@cite)
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
##### Constructing discrete transforms
#####

NoTransform() = DiscreteTransform([nothing for _ in fieldnames(DiscreteTransform)]...)

function DiscreteTransform(plan, direction, grid, dims)
    arch = architecture(grid)

    isnothing(plan) && return NoTransform()

    N = size(grid)
    topo = topology(grid)
    normalization = prod(normalization_factor(arch, topo[d](), direction, N[d]) for d in dims)
    twiddle = twiddle_factors(arch, grid, dims)
    transpose = arch isa GPU && dims == [2] ? (2, 1, 3) : nothing

    topo = [topology(grid)[d]() for d in dims]
    topo = length(topo) == 1 ? topo[1] : topo

    dims = length(dims) == 1 ? dims[1] : dims

    return DiscreteTransform(plan, grid, direction, dims, topo, normalization, twiddle, transpose)
end

#####
##### Applying discrete transforms
#####

(transform::DiscreteTransform{<:Nothing})(A, buffer) = nothing

function (transform::DiscreteTransform{P, <:Forward})(A, buffer) where P
    maybe_permute_indices!(A, buffer, architecture(transform), transform.grid, transform.dims, transform.topology)
    apply_transform!(A, buffer, transform.plan, transform.transpose_dims)
    maybe_twiddle_forward!(A, transform.twiddle_factors)
    maybe_normalize!(A, transform.normalization)
    return nothing
end

function (transform::DiscreteTransform{P, <:Backward})(A, buffer) where P
    maybe_twiddle_backward!(A, transform.twiddle_factors)
    apply_transform!(A, buffer, transform.plan, transform.transpose_dims)
    maybe_unpermute_indices!(A, buffer, architecture(transform), transform.grid, transform.dims, transform.topology)
    maybe_normalize!(A, transform.normalization)
    return nothing
end

maybe_permute_indices!(A, B, arch, grid, dim, dim_topo) = nothing

function maybe_permute_indices!(A, B, arch::GPU, grid, dim, ::Bounded)
    permute_indices!(B, A, arch, grid, dim)
    copyto!(A, B)
    return nothing
end

maybe_unpermute_indices!(A, B, arch, grid, dim, dim_topo) = nothing

function maybe_unpermute_indices!(A, B, arch::GPU, grid, dim, ::Bounded)
    unpermute_indices!(B, A, arch, grid, dim)
    copyto!(A, B)
    @. A = real(A)
    return nothing
end

function apply_transform!(A, B, plan, ::Nothing)
    plan * A
    return nothing
end

function apply_transform!(A, B, plan, transpose_dims)
    old_size = size(A)
    transposed_size = [old_size[d] for d in transpose_dims]

    if old_size == transposed_size
        permutedims!(B, A, transpose_dims)
        plan * B
        permutedims!(A, B, transpose_dims)
    else
        B_reshaped = reshape(B, transposed_size...)
        permutedims!(B_reshaped, A, transpose_dims)
        plan * B_reshaped
        permutedims!(A, B_reshaped, transpose_dims)
    end

    return nothing
end

maybe_twiddle_forward!(A, ::Nothing) = nothing

function maybe_twiddle_forward!(A, twiddle)
    @. A = 2 * real(twiddle.forward * A)
    return nothing
end

maybe_twiddle_backward!(A, ::Nothing) = nothing

function maybe_twiddle_backward!(A, twiddle)
    @. A *= twiddle.backward
    return nothing
end

function maybe_normalize!(A, normalization)
    # Avoid a tiny kernel launch if possible.
    if normalization != 1
        @. A *= normalization
    end
    return nothing
end
