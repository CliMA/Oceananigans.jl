module DistributedComputations

export
    Distributed, Partition, Equal, Fractional,
    child_architecture, reconstruct_global_grid, partition,
    inject_halo_communication_boundary_conditions,
    DistributedFFTBasedPoissonSolver, mpi_initialized, mpi_rank,
    mpi_size, global_barrier, global_communicator,
    @root, @onrank, @distribute, @handshake

using MPI

using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids
using OffsetArrays
using CUDA: CuArray
using Oceananigans.Grids: XYZRegularRG
using Oceananigans.Solvers: GridWithFFTSolver, GridWithFourierTridiagonalSolver

import Oceananigans.Solvers: fft_poisson_solver

include("distributed_macros.jl")
include("distributed_architectures.jl")
include("partition_assemble.jl")
include("distributed_grids.jl")
include("distributed_immersed_boundaries.jl")
include("distributed_on_architecture.jl")
include("distributed_kernel_launching.jl")
include("halo_communication_bcs.jl")
include("communication_buffers.jl")
include("distributed_fields.jl")
include("halo_communication.jl")
include("transposable_field.jl")
include("distributed_transpose.jl")
include("plan_distributed_transforms.jl")
include("distributed_fft_based_poisson_solver.jl")
include("distributed_fft_tridiagonal_solver.jl")

fft_poisson_solver(grid::DistributedGrid) = fft_poisson_solver(grid, reconstruct_global_grid(grid))

fft_poisson_solver(local_grid::DistributedGrid, global_grid::XYZRegularRG) =
    DistributedFFTBasedPoissonSolver(global_grid, local_grid)

fft_poisson_solver(local_grid::DistributedGrid, global_grid::GridWithFourierTridiagonalSolver) =
    DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)

import Oceananigans.Solvers: compute_preconditioner_rhs!, precondition!
using Oceananigans.Solvers: fft_preconditioner_rhs!

function compute_preconditioner_rhs!(solver::DistributedFFTBasedPoissonSolver, rhs)
    grid = solver.local_grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, fft_preconditioner_rhs!, solver.storage.zfield, rhs)
    return nothing
end

function precondition!(p, preconditioner::DistributedFFTBasedPoissonSolver, r, args...)
    compute_preconditioner_rhs!(preconditioner, r)
    shift = - sqrt(eps(eltype(r))) # to make the operator strictly negative definite
    solve!(p, preconditioner, shift)
    p .*= -1
    return p
end

end # module
