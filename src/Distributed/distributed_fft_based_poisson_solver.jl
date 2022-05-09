using Oceananigans.Grids: Center
using Oceananigans.Operators: Δzᶜᶜᶜ

import PencilArrays
import PencilFFTs
import FFTW

using PencilArrays: Permutation
using PencilFFTs: PencilFFTPlan

import Oceananigans.Solvers: poisson_eigenvalues, solve!, BatchedTridiagonalSolver, compute_batched_tridiagonals
import Oceananigans.Architectures: architecture

struct DistributedFFTBasedPoissonSolver{T, P, F, L, λ, R, I, ST, SD}
    plan :: P
    global_grid :: F
    local_grid :: L
    eigenvalues :: λ
    unpermuted_right_hand_side :: R
    input_permutation :: I
    transposition_storage :: ST
    tridiagonal_vertical_solver :: T
    tridiagonal_storage :: SD
end

const DistributedFourierTridiagonalPoissonSolver = DistributedFFTBasedPoissonSolver{<:BatchedTridiagonalSolver}

architecture(solver::DistributedFFTBasedPoissonSolver) =
    architecture(solver.global_grid)

infer_transform(::Periodic) = PencilFFTs.Transforms.FFT!()
infer_transform(::Bounded) = PencilFFTs.Transforms.R2R!(FFTW.REDFT10)
infer_transform(::Flat) = PencilFFTs.Transforms.NoTransform!()

"""
    DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)

Return a solver for the Poisson equation,

```math
∇²x = b
```

for `MultiArch`itectures and `RectilinearGrid`s. If `global_grid` is regularly spaced
in all directions, the discrete 2nd-order Poisson equation is solved for `x` with an eigenfunction
expansion that utilitizes fast Fourier transforms in (x, y, z).

If `global_grid` is vertically-stretched and horizontally-regular, the Poisson equation
is solved with Fourier transforms in (x, y) and a tridiagonal solve in z.

Other stretching configurations for `RectilinearGrid` are not supported.

Supported domain decompositions
===============================

We support two "modes":

    1. Vertical pencil decompositions: two-dimensional decompositions in (x, y)
       for three dimensional problems that satisfy either `Nz > Rx` or `Nz > Ry`.

    2. One-dimensional decompositions in either x or y.

Above, `Nz = size(global_grid, 3)` and `Rx, Ry, Rz = architecture(local_grid).ranks`.

Other configurations that are decomposed in (x, y) but have too few `Nz`,
or any configuration decomposed in z, are not supported.

Algorithm for two-dimensional decompositions
============================================

For two-dimensional decompositions (useful for three-dimensional problems),
there are either

1. Three forward transforms, three backward transforms,
   and four transpositions requiring MPI communication for a three-dimensionally
   regular `RectilinearGrid`.

2. Two forward transforms, two backward transforms, one tridiagonal solve,
   and _eight_ transpositions requiring MPI communication for a horizontally-regular,
   vertically-stretched `RectilinearGrid`.

We sketch these below in a schematic where the first dimension is _defined_ as the local
dimension. In our implementation of the PencilFFTs algorithm,
we require _either_ `Nz >= Rx`, _or_ `Nz >= Ry`, where `Nz` is the number of vertical cells,
`Rx` is the number of ranks in x, and `Ry` is the number of ranks in `y`.
Below, we outline the algorithm for the case `Nz >= Rx`.
If `Nz < Rx`, but `Nz > Ry`, a similar algorithm applies with x and y swapped:

1. `first(transposition_storage)` is initialized with layout (z, x, y).
2. If on a fully regular `RectlinearGrid`, transform along z. If on a vertically-stretched
   `RectilinearGrid`, we perform no transform.
3  Transpose + communicate to transposition_storage[2] in layout (x, z, y),
   which is distributed into `(Rx, Ry)` processes in (z, y).
4. Transform along x.
5. Transpose + communicate to last(transposition_storage) in layout (y, x, z),
   which is distributed into `(Rx, Ry)` processes in (x, z).
6. Transform in y.

5. Next we solve the Poisson equation. If on fully-regular `RectilinearGrid`,
    a. Solve the Poisson equation by updating `last(transposition_storage)`.
       and dividing by the sum of the eigenvalues
       and zero out the volume mean, zeroth mode, or (1, 1, 1) element of
       the transformed array.

    If using a vertically-stretched `RectilinearGrid`, then
    
      i. Transpose back to (x, z, y)
     ii. Transpose back to (z, x, y)
    iii. Solve the Poisson equation with a tridiagonal solve.
     iv. Transpose back to (x, z, y)
      v. Transpose back to (y, x, z)

At this point we are ready to perform a backwards transform to return to physical
space and obtain the solution in  `first(transposition_storage)` with the layout (z, x, y).

Restrictions
============

The algorithm for two-dimensional decompositions requires that `Nz = size(global_grid, 3)` is larger
than either `Rx = ranks[1]` or `Ry = ranks[2]`, where `ranks` are configured when building `MultiArch`.
If `Nz` does not satisfy this condition, we can only support a one-dimensional decomposition.

Algorithm for one-dimensional decompositions
============================================

For one-dimensional decompositions (ie, rank configurations with only one non-singleton
dimension) we require _either_ `Rx = 1` _or_ `Ry = 1`. One-dimensional domain
decompositions are required for two-dimensional transforms.

For one-dimensional decomopsitions, the z-direction is placed last. If
`topology(global_grid, 3) isa Flat`, or if the vertical direction is stretched
and thus requires a vertical tridiagonal solver, we omit transpositions and
FFTs in the z-direction entirely, yielding an algorithm with 2 transposes --- compared to 4
for two-dimensional decompositions for fully-regular solves, and _8_ for
two-dimensional decompositions and vertically-stretched solves.
"""
function DistributedFFTBasedPoissonSolver(global_grid, local_grid)

    arch = architecture(local_grid)
    Rx, Ry, Rz = arch.ranks
    communicator = arch.communicator
    gNx, gNy, gNz = size(global_grid)

    # Check input:
    #     1. Number of ranks in vertical must be 1 (for now).
    #     2. If vertically-stretched, we'll use a tridiagonal solve in the vertical.
    #     3. If using a two-dimensional process grid, the vertical dimension must be
    #        large enough to support the required transposes (either Nz > Rx or Nz > Ry).

    Rz == 1 || throw(ArgumentError("Non-singleton ranks in the vertical are not supported by DistributedFFTBasedPoissonSolver."))
    two_dimensional_decomposition = Rx > 1 && Ry > 1
    two_dimensional_decomposition && gNz < Rx && gNz < Ry &&
        throw(ArgumentError("DistributedFFTBasedPoissonSolver requires either " *
                            "i) Nz > Rx, ii) Nz > Ry, or iii) a one-dimensional process grid with Rx = 1 or Ry = 1."))

    using_tridiagonal_vertical_solver = !(global_grid isa RegRectilinearGrid) 

    # Build _global_ eigenvalues
    TX, TY, TZ = topology(global_grid)

    # Neutralize transforms and eigenvalues in vertical direction if using tridiagonal vertical solve
    effective_topology = using_tridiagonal_vertical_solver ? (TX(), TY(), Flat()) :
                                                             (TX(), TY(), TZ())
    
    if two_dimensional_decomposition
        if gNz >= Rx 
            input_permutation = Permutation(3, 1, 2)
            permuted_size = (gNz, gNx, gNy)
            processors_per_dimension = (Rx, Ry)
            extra_dims = ()
        else # gNz >= Ry
            input_permutation = Permutation(3, 2, 1)
            permuted_size = (gNz, gNy, gNx)
            processors_per_dimension = (Ry, Rx)
            extra_dims = ()
        end
    else # one-dimensional decomposition

        if Rx == 1 # x-local, y-distributed
            permuted_size = (gNx, gNy, gNz)
            input_permutation = Permutation(1, 2, 3)
            processors_per_dimension = (Ry, 1)
        else # Ry == 1, y-local, x-distributed
            permuted_size = (gNy, gNx, gNz)
            input_permutation = Permutation(2, 1, 3)
            processors_per_dimension = (Rx, 1)
        end
    end

    # Create eigenvalues for permutation
    λx = poisson_eigenvalues(gNx, global_grid.Lx, 1, effective_topology[1])
    λy = poisson_eigenvalues(gNy, global_grid.Ly, 2, effective_topology[2])
    λz = poisson_eigenvalues(gNz, global_grid.Lz, 3, effective_topology[3])

    # Drop singleton dimensions for compatibility with PencilFFTs' localgrid
    λx = dropdims(λx, dims=(2, 3))
    λy = dropdims(λy, dims=(1, 3))
    λz = dropdims(λz, dims=(1, 2))

    permuted_eigenvalues = Tuple((λx, λy, λz)[d] for d in Tuple(input_permutation))
    transforms = Tuple(infer_transform(effective_topology[d]) for d in Tuple(input_permutation))

    # If the _effective_ z-topology is Flat --- which occurs either because topology(global_grid, 3)
    # is actually Flat, or because we are using_tridiagonal_vertical_solver --- we use "extra_dims"
    # to represent the vertical (3rd) dimension to avoid needless transform / permutation.
    #
    # With this option, we have to remove the last element from both the permuted size and process grid.
    # Note that setting extra_dims = tuple(gNz) means that input / storage will have size (N1, N2, gNz).
    if effective_topology[3] isa Flat && !two_dimensional_decomposition # this is enforced above as well
        extra_dims = tuple(gNz)
        permuted_size = permuted_size[1:2]
        processors_per_dimension = tuple(processors_per_dimension[1]) 
        permuted_eigenvalues = permuted_eigenvalues[1:2]
        transforms = transforms[1:2]
    else
        extra_dims = ()
    end

    @info "Building PencilFFTPlan with transforms $transforms, process grid $processors_per_dimension, and extra_dims $extra_dims..."
    plan = PencilFFTPlan(permuted_size, transforms, processors_per_dimension, communicator; extra_dims)

    # Allocate memory for in-place FFT + transpositions
    transposition_storage = PencilFFTs.allocate_input(plan)

    # Store a view of the right hand side that "appears" to have the permutation (x, y, z).
    permuted_right_hand_side = first(transposition_storage)
    unpermuted_right_hand_side = PermutedDimsArray(parent(permuted_right_hand_side), Tuple(input_permutation))

    if using_tridiagonal_vertical_solver
        ri, rj, rk = arch.local_index
        nx, ny, nz = size(local_grid) # probably don't need this
        local_λx = partition(λx, Center(), nx, Rx, ri)
        local_λy = partition(λy, Center(), ny, Ry, rj)
        lower_diagonal, diagonal, upper_diagonal = compute_batched_tridiagonals(local_grid, local_λx, local_λy)
        tridiagonal_vertical_solver = BatchedTridiagonalSolver(local_grid; lower_diagonal, diagonal, upper_diagonal)
        tridiagonal_storage = zeros(eltype(first(transposition_storage)), architecture(local_grid), size(local_grid)...)
        eigenvalues = nothing
    else
        tridiagonal_vertical_solver = nothing
        tridiagonal_storage = nothing
        eigenvalues = PencilFFTs.localgrid(last(transposition_storage), permuted_eigenvalues)
    end

    return DistributedFFTBasedPoissonSolver(plan,
                                            global_grid,
                                            local_grid,
                                            eigenvalues,
                                            unpermuted_right_hand_side,
                                            input_permutation,
                                            transposition_storage,
                                            tridiagonal_vertical_solver,
                                            tridiagonal_storage)
end

"""
    solve_transformed_poission_equation!(solver)

Solve the discrete Poisson equation after transforming into the three-dimensional
eigenfunction space of the second-order discrete Poisson operator.
We solve the Poisson equation "in-place", re-using the transformed
right-hand-side, `b̂ = last(solver.transposition_storage)` to store the solution `x̂`.
"""
function solve_transformed_poisson_equation!(solver)
    x̂ = b̂ = last(solver.transposition_storage)
    λ = solver.eigenvalues

    # It's a little better than writing @. x̂ = - b / +(λ...)
    if length(λ) === 3
        @. x̂ = - b̂ / (λ[1] + λ[2] + λ[3])
    elseif length(λ) === 2
        @. x̂ = - b̂ / (λ[1] + λ[2])
    end

    # Set the zeroth wavenumber and volume mean, which are undetermined
    # in the Poisson equation, to zero.
    multi_arch = architecture(solver.local_grid)

    i, j, k = PencilArrays.range_local(x̂)
    if (i[1], j[1], k[1]) === (1, 1, 1) # we have the zeroth node!
        parent(x̂)[1, 1, 1] = 0
    end

    return nothing
end

"""
    solve_transformed_poission_equation!(solver::DistributedFourierTridiagonalPoissonSolver)

Solve the discrete Poisson equation after transforming into the two-dimensional
eigenfunction space of the horizontal component of the second-order discrete Poisson operator.

We use a vertical tridiagonal solver which updates the solution "in-place",
re-using the transformed right-hand-side, `b̂ = last(solver.transposition_storage)` to store the solution `x̂`.
"""
function solve_transformed_poisson_equation!(solver::DistributedFourierTridiagonalPoissonSolver)

    perm = solver.input_permutation # permutation of the "input" to the solver, ie the rhs.
    if perm === Permutation(3, 1, 2) || perm === Permutation(3, 2, 1)

        # In these cases, the input to the solve is permuted such that the
        # z-dimension is first, and local. This means that the _output_ to the
        # transformed data has a permutation in which the z-dimension is last,
        # and distributed across processes. As a result, we must perform 2
        # transposes to return the z-dimension to process-continguous,
        # perform a tridiagonal solve, and then transpose back to prepare for
        # the backward transforms.,

        # Perform transposes+communication to obtain a local continguous
        # z-dimension. In our notation below "h" corresponds to x or y,
        # "z" is vertical.
        zhh_storage = solver.transposition_storage[1]
        hzh_storage = solver.transposition_storage[2]
        hhz_storage = solver.transposition_storage[3]

        # Perform two transposes so that the z-dimension is local to each process.
        PencilFFTs.transpose!(hzh_storage, hhz_storage)
        PencilFFTs.transpose!(zhh_storage, hzh_storage)

        # Solve tridiagonal system of linear equations in z at every column:
        #     1. Copy transformed rhs into new array
        #     2. Execute tridiagonal solve
        #
        # Note: solver.unpermuted_right_hand_side is an "unpermuted" view into
        # parent(zhh_storage) = parent(first(solver.transposition_storage))
        
        x̂ = solver.unpermuted_right_hand_side
        b̂ = solver.tridiagonal_storage
        b̂ .= x̂

        # Solve and store the result in x̂
        solve!(x̂, solver.tridiagonal_vertical_solver, b̂)
        
        # Perform two transposes to return `solver.transposition_storage` to the
        # configuration needed for backwards transforms.
        PencilFFTs.transpose!(hzh_storage, zhh_storage)
        PencilFFTs.transpose!(hhz_storage, hzh_storage)

    else # z is local, and we are good to go!

        x̂ = solver.unpermuted_right_hand_side
        b̂ = solver.tridiagonal_storage
        b̂ .= x̂
        solve!(x̂, solver.tridiagonal_vertical_solver, b̂)

    end

    return nothing
end

# solve! requires that `b` in `A x = b` (the right hand side)
# was computed and stored in first(solver.transposition_storage) prior to calling `solve!(x, solver)`.
# See: Models/NonhydrostaticModels/solve_for_pressure.jl
function solve!(x, solver::DistributedFFTBasedPoissonSolver)
    # Because (why though?) the tridiagonal source term is multiplied by Δz...
    preprocess_source_term!(solver)

    # Apply forward transforms to b = first(solver.transposition_storage).
    solver.plan * solver.transposition_storage

    solve_transformed_poisson_equation!(solver)
    
    # Apply backward transforms to x̂ = last(solver.transposition_storage).
    solver.plan \ solver.transposition_storage
    
    # xc is the backward transform of x̂.
    xc = first(solver.transposition_storage)

    # Zero volume mean (if it wasn't already)
    arch = architecture(solver.global_grid)

    if isnothing(solver.tridiagonal_vertical_solver) # well then we didn't use a tridiagonal solve
        mean_xc = 0 # guaranteed zero by solve_transformed_poisson_equation!
    else
        mean_xc = MPI.Allreduce(sum(real(xc)), +, arch.communicator)
    end
	
    # Copy the real component of xc to x.
    copy_event = launch!(arch, solver.local_grid, :xyz,
                         copy_permuted_real_component!, x, solver.input_permutation, parent(xc), mean_xc,
                         dependencies = device_event(arch))

    wait(device(arch), copy_event)

    return x
end

const ZXYPermutation = Permutation{(3, 1, 2), 3}
const ZYXPermutation = Permutation{(3, 2, 1), 3}
const XYZPermutation = Permutation{(1, 2, 3), 3}
const YXZPermutation = Permutation{(2, 1, 3), 3}

real_ϕc(i, j, k, ::XYZPermutation, ϕc) = @inbounds real(ϕc[i, j, k])
real_ϕc(i, j, k, ::YXZPermutation, ϕc) = @inbounds real(ϕc[j, i, k])
real_ϕc(i, j, k, ::ZYXPermutation, ϕc) = @inbounds real(ϕc[k, j, i])
real_ϕc(i, j, k, ::ZXYPermutation, ϕc) = @inbounds real(ϕc[k, i, j])

@kernel function copy_permuted_real_component!(ϕ, perm, ϕc, mean_xc)
    i, j, k = @index(Global, NTuple)
    @inbounds ϕ[i, j, k] = real_ϕc(i, j, k, perm, ϕc) - mean_xc
end

preprocess_source_term!(solver) = nothing

function preprocess_source_term!(solver::DistributedFourierTridiagonalPoissonSolver)
    arch = architecture(solver.global_grid)
    input = solver.unpermuted_right_hand_side

    # Copy the real component of xc to x.
    event = launch!(arch, solver.local_grid, :xyz,
                    multiply_by_Δzᶜᶜᶜ!, input, solver.local_grid,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@kernel function multiply_by_Δzᶜᶜᶜ!(a, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds a[i, j, k] *= Δzᶜᶜᶜ(i, j, k, grid)
end
