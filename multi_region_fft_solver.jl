using Oceananigans.Architectures: device_event

import Oceananigans.Architectures: architecture

struct MultiRegionFFTBasedPoissonSolver{G, GG, Λ, S, B, T}
            grid :: G
     global_grid :: GG
     eigenvalues :: Λ
         storage :: S
          buffer :: B
      transforms :: T
end

architecture(solver::MultiRegionFFTBasedPoissonSolver) = architecture(solver.grid)

function MultiRegionFFTBasedPoissonSolver(grid, planner_flag=FFTW.PATIENT)
    topo = (TX, TY, TZ) =  topology(grid)

    global_grid = reconstruct_global_grid(grid)

    λx = poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX())
    λy = poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY())
    λz = poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ())

    arch = architecture(grid)

    eigenvalues = (λx = unified_array(arch, λx),
                   λy = unified_array(arch, λy),
                   λz = unified_array(arch, λz))

    storage = unified_array(arch, zeros(complex(eltype(global_grid)), size(global_grid)...))

    # Permutation on the grid will go here!
    transforms = plan_transforms(grid, storage, planner_flag)

    # Need buffer for index permutations and transposes.
    buffer_needed = arch isa GPU && Bounded in topo
    buffer = buffer_needed ? similar(storage) : nothing

    return MultiRegionFFTBasedPoissonSolver(grid, global_grid, eigenvalues, storage, buffer, transforms)
end

"""
    solve!(ϕ, solver::FFTBasedPoissonSolver, b, m=0)

Solves the "generalized" Poisson equation,

```math
(∇² + m) ϕ = b,
```

where ``m`` is a number, using a eigenfunction expansion of the discrete Poisson operator
on a staggered grid and for periodic or Neumann boundary conditions.

In-place transforms are applied to ``b``, which means ``b`` must have complex-valued
elements (typically the same type as `solver.storage`).

!!! info "Alternative names for "generalized" Poisson equation
    Equation ``(∇² + m) ϕ = b`` is sometimes called the "screened Poisson" equation
    when ``m < 0``, or the Helmholtz equation when ``m > 0``.
"""
function solve!(ϕ, solver::MultiRegionFFTBasedPoissonSolver, b, m=0)
    arch = architecture(solver)
    λx, λy, λz = solver.eigenvalues

    # Temporarily store the solution in ϕc
    ϕc = solver.storage

    # Transform b *in-place* to eigenfunction space (we have to encode the)
    @apply_regionally regional_transform!(b, solver, solver.transforms.forward[1])
    # If we didn't have unified memory the transpose would go in here!
    @apply_regionally regional_transform!(b, solver, solver.transforms.forward[2:3])

    # Solve the discrete screened Poisson equation (∇² + m) ϕ = b.
    @. ϕc = - b / (λx + λy + λz - m)

    # If m === 0, the "zeroth mode" at `i, j, k = 1, 1, 1` is undetermined;
    # we set this to zero by default. Another slant on this "problem" is that
    # λx[1, 1, 1] + λy[1, 1, 1] + λz[1, 1, 1] = 0, which yields ϕ[1, 1, 1] = Inf or NaN.
    m === 0 && CUDA.@allowscalar ϕc[1, 1, 1] = 0

    # Apply backward transforms in order
    # Transform b *in-place* to eigenfunction space (we have to encode the)
    @apply_regionally regional_transform!(b, solver, solver.transforms.backward[1])
    # If we didn't have unified memory the transpose would go in here!
    @apply_regionally regional_transform!(b, solver, solver.transforms.backward[2:3])

    copy_event = launch!(arch, solver.grid, :xyz, copy_real_component!, ϕ, ϕc, dependencies=device_event(arch))
    wait(device(arch), copy_event)

    return ϕ
end

# Here we have to encode the grid transformation
function regional_transform!(b, solver, direction) 
    # Perform FFT in two direction
    [transform!(b, solver.buffer) for transform! in direction]
end

@kernel function copy_real_component!(ϕ, ϕc)
    i, j, k = @index(Global, NTuple)
    @inbounds ϕ[i, j, k] = real(ϕc[i, j, k])
end

import Oceananigans.Architectures: architecture

abstract type AbstractTransformDirection end

struct MultiRegionForward <: AbstractTransformDirection end
struct MultiRegionBackward <: AbstractTransformDirection end

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
function twiddle_factors(arch::GPU, grid::MultiRegionGrid, dims)
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
        forward = unified_array(arch, ω_4N⁺),
        backward = unified_array(arch, ω_4N⁻)
    )

    return twiddle_factors
end

#####
##### Constructing discrete transforms
#####

NoTransform() = DiscreteTransform([nothing for _ in fieldnames(DiscreteTransform)]...)

function DiscreteTransform(plan, direction, grid::MultiRegionGrid, dims)
    arch = architecture(grid)

    global_grid = reconstruct_global_grid(grid)

    isnothing(plan) && return NoTransform()

    N = size(global_grid)
    topo = topology(global_grid)
    normalization = prod(normalization_factor(arch, topo[d](), direction, N[d]) for d in dims)
    twiddle = twiddle_factors(arch, global_grid, dims)
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

" Used by FFTBasedPoissonSolver "
function plan_transforms(grid::MultiRegionGrid, storage, planner_flag)
    global_grid = reconstruct_global_grid(grid)
    Nx, Ny, Nz = size(grid)
    topo = topology(grid)
    periodic_dims = findall(t -> t == Periodic, topo)
    bounded_dims = findall(t -> t == Bounded, topo)

    # Convert Flat to Bounded for inferring batchability and transform ordering
    # Note that transforms are omitted in Flat directions.
    unflattened_topo = Tuple(T() isa Flat ? Bounded : T for T in topo)

    arch = architecture(grid)

    if arch isa GPU && !(unflattened_topo in batchable_GPU_topologies)

        rs_storage = reshape(storage, (Ny, Nx, Nz))

        # X will be the non local direction?
        forward_plan_x = plan_forward_transform(storage,    topo[1](), [1], planner_flag)
        
        forward_plan_y = plan_forward_transform(rs_storage, topo[2](), [1], planner_flag)
        forward_plan_z = plan_forward_transform(storage,    topo[3](), [3], planner_flag)

        # X will be the non local direction?
        backward_plan_x = plan_backward_transform(storage,    topo[1](), [1], planner_flag)
        
        backward_plan_y = plan_backward_transform(rs_storage, topo[2](), [1], planner_flag)
        backward_plan_z = plan_backward_transform(storage,    topo[3](), [3], planner_flag)

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
    end

    transforms = (forward=forward_transforms, backward=backward_transforms)

    return transforms
end