import PencilFFTs
using PencilArrays: Permutation

import Oceananigans.Solvers: poisson_eigenvalues, solve!
import Oceananigans.Architectures: architecture

struct DistributedFFTBasedPoissonSolver{P, F, L, λ, S, I}
    plan :: P
    global_grid :: F
    local_grid :: L
    eigenvalues :: λ
    storage :: S
    input_permutation :: I
end

architecture(solver::DistributedFFTBasedPoissonSolver) =
    architecture(solver.global_grid)

infer_transform(grid, d) = infer_transform(topology(grid, d)())
infer_transform(::Periodic) = PencilFFTs.Transforms.FFT!()
infer_transform(::Bounded) = PencilFFTs.Transforms.R2R!(FFTW.REDFT10)
infer_transform(::Flat) = PencilFFTs.Transforms.NoTransform!()

"""
    DistributedFFTBasedPoissonSolver(global_grid, local_grid)

Return a FFT-based solver for the Poisson equation,

```math
∇²x = b
```

for `MultiArch`itectures.

Supported configurations
========================

We support two "modes":

    1. Two-dimensional decompositions in (x, y) for problems with either
       `Nz > Rx` or `Nz > Ry` (therefore, three-dimensional).

    2. One-dimensional decompositions in either x or y for problems that are
       either two-dimensional, or have limited dimensionality in z.

Above, `Nz = size(global_grid, 3)` and `Rx, Ry, Rz = architecture(local_grid).ranks`.

Other configurations that are decomposed in (x, y) but have too few `Nz`,
or any configuration decomposed in z, are not supported.

Algorithm for two-dimensional decompositions
============================================

For two-dimensional decompositions (useful for three-dimensional problems),
there are three forward transforms, three backward transforms,
and four transpositions requiring MPI communication. In the schematic below, the first
dimension is always the local dimension. In our implementation of the PencilFFTs algorithm,
we require _either_ `Nz >= Rx`, _or_ `Nz >= Ry`, where `Nz` is the number of vertical cells,
`Rx` is the number of ranks in x, and `Ry` is the number of ranks in `y`.
Below, we outline the algorithm for the case `Nz >= Rx`.
If `Nz < Rx`, but `Nz > Ry`, a similar algorithm applies with x and y swapped:

1. `first(storage)` is initialized with layout (z, x, y).
2. Transform along z.
3  Transpose + communicate to storage[2] in layout (x, z, y),
   which is distributed into `(Rx, Ry)` processes in (z, y).
4. Transform along x.
5. Transpose + communicate to last(storage) in layout (y, x, z),
   which is distributed into `(Rx, Ry)` processes in (x, z).
6. Transform in y.

At this point the three in-place forward transforms are complete, and we
solve the Poisson equation by updating `last(storage)`.
Then the process is reversed to obtain `first(storage)` in physical
space and with the layout (z, x, y).

Restrictions
============

The algorithm for two-dimensional decompositions requires that `Nz = size(global_grid, 3)` is larger
than either `Rx = ranks[1]` or `Ry = ranks[2]`, where `ranks` are configured when building `MultiArch`.
If `Nz` does not satisfy this condition, we can only support a one-dimensional decomposition.

Algorithm for one-dimensional decompositions
============================================

This algorithm requires a one-dimensional decomposition with _either_ `Rx = 1`
_or_ `Ry = 1`, and is important to support two-dimensional transforms.

For one-dimensional decompositions, we place the decomposed direction _last_.
If the number of ranks is `Rh = max(Rx, Ry)`, this algorithm requires that 
_both_ `Nx > Rh` _and_ `Ny > Rh`. The resulting flow of transposes and transforms
is similar to the two-dimensional case. It remains somewhat of a mystery why this
succeeds (ie, why the last transform is correctly decomposed).
"""
function DistributedFFTBasedPoissonSolver(global_grid, local_grid)

    arch = architecture(local_grid)
    Rx, Ry, Rz = arch.ranks
    communicator = arch.communicator

    # We don't support distributing anything in z.
    Rz == 1 || throw(ArgumentError("Non-singleton ranks in the vertical are not supported by DistributedFFTBasedPoissonSolver."))

    gNx, gNy, gNz = size(global_grid)

    # Build _global_ eigenvalues
    topo = (TX, TY, TZ) = topology(global_grid)
    λx = poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX())
    λy = poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY())
    λz = poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ())

    # Drop singleton dimensions for compatibility with PencilFFTs' localgrid
    λx = dropdims(λx, dims=(2, 3))
    λy = dropdims(λy, dims=(1, 3))
    λz = dropdims(λz, dims=(1, 2))

    unpermuted_eigenvalues = (λx, λy, λz)

    # First we check if we can do a two-dimensional decomposition
    if gNz >= Rx 
        input_permutation = Permutation(3, 1, 2)
        permuted_size = (gNz, gNx, gNy)
        processors_per_dimension = (Rx, Ry)
    elseif gNz >= Ry
        input_permutation = Permutation(3, 2, 1)
        permuted_size = (gNz, gNy, gNx)
        processors_per_dimension = (Ry, Rx)

    else # it has to be a one-dimensional decomposition

        Rx > 1 && Ry > 1 &&
            throw(ArgumentError("DistributedFFTBasedPoissonSolver requires either " *
                                "i) Nz > Rx, ii) Nz > Ry, iii) Rx = 1 _or_ iv) Ry = 1."))

        if Rx == 1 # x-local, y-distributed
            permuted_size = (gNz, gNx, gNy)
            input_permutation = Permutation(3, 1, 2)
            processors_per_dimension = (1, Ry)
        else # Ry == 1, y-local, x-distributed
            permuted_size = (gNz, gNy, gNx)
            input_permutation = Permutation(3, 2, 1)
            processors_per_dimension = (1, Rx)
        end
    end

    transforms = Tuple(infer_transform(global_grid, d) for d in Tuple(input_permutation))
    plan = PencilFFTs.PencilFFTPlan(permuted_size, transforms, processors_per_dimension, communicator)

    # Allocate memory for in-place FFT + transpositions
    storage = PencilFFTs.allocate_input(plan)
    
    # Permute the λ appropriately
    permuted_eigenvalues = Tuple(unpermuted_eigenvalues[d] for d in Tuple(input_permutation))
    eigenvalues = PencilFFTs.localgrid(last(storage), permuted_eigenvalues)

    return DistributedFFTBasedPoissonSolver(plan, global_grid, local_grid, eigenvalues, storage, input_permutation)
end

# solve! requires that `b` in `A x = b` (the right hand side)
# was computed and stored in first(solver.storage) prior to calling `solve!(x, solver)`.
# See: Models/NonhydrostaticModels/solve_for_pressure.jl
function solve!(x, solver::DistributedFFTBasedPoissonSolver)
    arch = architecture(solver.global_grid)
    multi_arch = architecture(solver.local_grid)

    # Apply forward transforms to b = first(solver.storage).
    solver.plan * solver.storage

    # Solve the discrete Poisson equation in wavenumber space
    # for x̂. We solve for x̂ in place, reusing b̂.
    x̂ = b̂ = last(solver.storage)
    λ = solver.eigenvalues
    @. x̂ = - b̂ / (λ[1] + λ[2] + λ[3])

    # Set the zeroth wavenumber and volume mean, which are undetermined
    # in the Poisson equation, to zero.
    if MPI.Comm_rank(multi_arch.communicator) == 0
        # This is an assumption: we *hope* PencilArrays allocates in this way
        parent(x̂)[1, 1, 1] = 0
    end

    # Apply backward transforms to x̂ = last(solver.storage).
    solver.plan \ solver.storage
    
    # xc is the backward transform of x̂.
    xc = first(solver.storage)
	
    # Copy the real component of xc to x.
    copy_event = launch!(arch, solver.local_grid, :xyz,
                         copy_permuted_real_component!, x, parent(xc), solver.input_permutation,
                         dependencies = device_event(arch))

    wait(device(arch), copy_event)

    return x
end

const ZXYPermutation = Permutation{(3, 1, 2), 3}
const ZYXPermutation = Permutation{(3, 2, 1), 3}

@kernel function copy_permuted_real_component!(ϕ, ϕc, ::ZXYPermutation)
    i, j, k = @index(Global, NTuple)
    @inbounds ϕ[i, j, k] = real(ϕc[k, i, j])
end

@kernel function copy_permuted_real_component!(ϕ, ϕc, ::ZYXPermutation)
    i, j, k = @index(Global, NTuple)
    @inbounds ϕ[i, j, k] = real(ϕc[k, j, i])
end

