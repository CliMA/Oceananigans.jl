import PencilFFTs

import Oceananigans.Solvers: poisson_eigenvalues, solve!
import Oceananigans.Architectures: architecture

struct DistributedFFTBasedPoissonSolver{P, F, L, λ, S}
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
    #   * This only works for triply periodic models...
    #   * Because PencilFFT does not support partitioning along the x-dimension,
    #     but Oceananigans does not support partitioning along the _z_-dimension,
    #     we permute the PencilFFTs storage object to have the layout (z, x, y).
    
    gNx, gNy, gNz = size(global_grid)
    permuted_size = (gNz, gNx, gNy)
    processors_per_dimension = (Rx, Ry)

    communicator = MPI.COMM_WORLD
    transforms = PencilFFTs.Transforms.FFT!() # only
    plan = PencilFFTs.PencilFFTPlan(permuted_size, transforms, processors_per_dimension, communicator)

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

    # Note the permutation: (z, x, y).
    eigenvalues = PencilFFTs.localgrid(last(storage), (λz, λx, λy))

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

    # Apply forward transforms.
    solver.plan * solver.storage

    # Solve the discrete Poisson equation, storing the solution
    # temporarily in xc and later extracting the real part into 
    # the solution, x.
    xhat = b = last(solver.storage)
    @. xhat = - b / (λx + λy + λz)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        xc[1, 1, 1] = 0
    end

    # Apply backward transforms.
    solver.plan \ solver.storage
    
    # xc is Complex and the physical space outcome of the inverse transform of xhat.
    xc = first(solver.storage)
	
    # Copy just the real component of xc to x.
    copy_event = launch!(arch, solver.local_grid, :xyz, copy_permuted_real_component!, x, xc, dependencies=device_event(arch))
    wait(device(arch), copy_event)

    return x
end

@kernel function copy_permuted_real_component!(ϕ, ϕc)
    i, j, k = @index(Global, NTuple)
    # Note the index permutation
    @inbounds ϕ[i, j, k] = real(ϕc[k, i, j])
end
