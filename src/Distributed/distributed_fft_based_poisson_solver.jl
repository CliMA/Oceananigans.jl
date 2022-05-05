import PencilFFTs

import Oceananigans.Solvers: poisson_eigenvalues, solve!
import Oceananigans.Architectures: architecture

struct DistributedFFTBasedPoissonSolver{P, F, L, λ, S}
    # Do we need backwards_plan :: B too?
    plan :: P
    global_grid :: F
    local_grid :: L
    eigenvalues :: λ
    storage :: S
end

architecture(solver::DistributedFFTBasedPoissonSolver) =
    architecture(solver.global_grid)

function DistributedFFTBasedPoissonSolver(global_grid, local_grid)

    arch = architecture(local_grid)
    Rx, Ry, Rz = arch.ranks
    Rz == 1 || throw(ArgumentError("Non-singleton ranks in the vertical are not supported by DistributedFFTBasedPoissonSolver."))

    # Create a PencilFFTPlan.
    # 
    # Note:
    #
    #   * This only works for triply periodic models now...
    #
    #   * Because PencilFFT cannot partition the _first_ dimension,
    #     and Oceananigans cannot partition the _last_ (z) dimension,
    #     we support pencil partitioning by permuting the PencilFFTs storage
    #     to have the layout (z, y, x).
    #
    #   * The transformed data must be placed in first(solver.storage).
    #
    #   * After performing a transform, last(solver.storage) contains the "output", and has
    #     the layout (x, y, z).
    #
    #   * After performing an inverse transform, first(solver.storage) has the layout (z, y, x).
    #
    
    gNx, gNy, gNz = size(global_grid)
    permuted_size = (gNz, gNy, gNx)
    processors_per_dimension = (Ry, Rx)

    communicator = MPI.COMM_WORLD
	
    # To support Bounded, we need something like
    periodic_transform = PencilFFTs.Transforms.FFT!()
    # bounded_transform = PencilFFTs.Transforms.R2R!(FFTW.REDFT10)
    # transforms = Tuple(T() isa Periodic ? periodic_transform : bounded_transform for T in topology(global_grid))
    transforms = periodic_transform
    plan = PencilFFTs.PencilFFTPlan(permuted_size, transforms, processors_per_dimension, communicator)

    # Maybe also
    # bwd_bounded_transform = PencilFFTs.Transforms.R2R!(FFTW.REDFT01)
    # bwd_transforms = Tuple(T() isa Periodic ? periodic_transform : bwd_bounded_transform for T in topology(global_grid))
    # backwards_plan = PencilFFTs.PencilFFTPlan(permuted_size, bwd_transforms, processors_per_dimension, communicator)
	
    # Allocate memory for in-place FFT + transpositions
    storage = PencilFFTs.allocate_input(plan)

    # Build _global_ eigenvalues
    topo = (TX, TY, TZ) = topology(global_grid)
    λx = poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX())
    λy = poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY())
    λz = poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ())

    # We add singleton dimensions because its "convenient", but
    # PencilFFTs doesn't want that.
    λx = dropdims(λx, dims=(2, 3))
    λy = dropdims(λy, dims=(1, 3))
    λz = dropdims(λz, dims=(1, 2))

    # Note the permutation: (z, y, x).
    eigenvalues = PencilFFTs.localgrid(last(storage), (λz, λy, λx))

    return DistributedFFTBasedPoissonSolver(plan, global_grid, local_grid, eigenvalues, storage)
end

# solve! requires that `b` in `A x = b` (the right hand side)
# was computed and stored in first(solver.storage) prior to calling `solve!(x, solver)`.
# See: Models/NonhydrostaticModels/solve_for_pressure.jl
function solve!(x, solver::DistributedFFTBasedPoissonSolver)
    arch = architecture(solver)

    λx = solver.eigenvalues[1]
    λy = solver.eigenvalues[2]
    λz = solver.eigenvalues[3]

    # Apply forward transforms to b = first(solver.storage).
    solver.plan * solver.storage

    # Solve the discrete Poisson equation in wavenumber space
    # for x̂. We solve for x̂ in place, reusing b̂.
    x̂ = b̂ = last(solver.storage)

    @. x̂ = - b̂ / (λx + λy + λz)

    # Set the zeroth wavenumber and volume mean, which are undetermined
    # in the Poisson equation, to zero.
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        x̂[1, 1, 1] = 0
    end

    # Apply backward transforms to x̂ = last(solver.storage).
    solver.plan \ solver.storage
    
    # xc is the backward transform of x̂.
    xc = first(solver.storage)
	
    # Copy the real component of xc to x. Note that the axes of xc are permuted compared to x,
    # if x's axes are (1, 2, 3), then the layout of xc is (3, 2, 1).
    copy_event = launch!(arch, solver.local_grid, :xyz, copy_permuted_real_component!, x, xc, dependencies=device_event(arch))

    wait(device(arch), copy_event)

    return x
end

@kernel function copy_permuted_real_component!(ϕ, ϕc)
    i, j, k = @index(Global, NTuple)
    # Note the index permutation
    @inbounds ϕ[i, j, k] = real(ϕc[k, j, i])
end
