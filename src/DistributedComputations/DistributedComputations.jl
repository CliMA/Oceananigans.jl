module DistributedComputations

export
    Distributed, NCCLDistributed, Partition, Equal, Fractional,
    child_architecture, reconstruct_global_grid, partition,
    inject_halo_communication_boundary_conditions,
    DistributedFFTBasedPoissonSolver, TransposableField, mpi_initialized, mpi_rank,
    mpi_size, global_barrier, global_communicator, sanitize_environ!,
    @root, @onrank, @distribute, @handshake

using MPI

using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids
using OffsetArrays
using Oceananigans.Grids: XYZRegularRG
using Oceananigans.Solvers: GridWithFourierTridiagonalSolver

import Oceananigans.Solvers: fft_poisson_solver
using DocStringExtensions: TYPEDSIGNATURES

include("distributed_macros.jl")
include("sanitize_environ.jl")
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

"""
$(TYPEDSIGNATURES)

Return a [`Distributed`](@ref) architecture whose halo exchange in `x` and `y` (including
corners) and whose transposes for the distributed FFT-based pressure solvers use NCCL
(NVIDIA Collective Communications Library). MPI is still used to launch the job, to bootstrap
the NCCL communicator, and for reductions, broadcasts and barriers; device buffers passed to
these MPI calls are staged through host memory, so MPI does not need to be CUDA-aware.

Provided by the `OceananigansNCCLExt` extension: requires `using CUDA` and `using NCCL`,
NVIDIA GPUs, and one MPI rank per GPU.

Positional arguments
====================

- `child_architecture`: must be a CUDA `GPU()`. Default: `GPU()`.

Keyword arguments
=================

Same as [`Distributed`](@ref): `partition` (use `z = 1`), `devices`, `communicator`,
`synchronized_communication`.

See the manual section [Multi-GPU simulations with NCCL](@ref nccl_multi_gpu).
"""
function NCCLDistributed(child_architecture = GPU(); kwargs...)
    error("""
    NCCLDistributed is provided via an extension and requires CUDA and NCCL,
    and a CUDA `GPU()` child architecture.

    Fix:
      julia> using CUDA, NCCL

      julia> NCCLDistributed(GPU(); ...)

    If NCCL isn't installed:
      julia> using Pkg; Pkg.add("NCCL")
    """)
end

fft_poisson_solver(grid::DistributedRectilinearGrid) = fft_poisson_solver(grid, reconstruct_global_grid(grid))

fft_poisson_solver(local_grid::DistributedRectilinearGrid, global_grid::XYZRegularRG) =
    DistributedFFTBasedPoissonSolver(global_grid, local_grid)

fft_poisson_solver(local_grid::DistributedRectilinearGrid, global_grid::GridWithFourierTridiagonalSolver) =
    DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)

import Oceananigans.Solvers: compute_preconditioner_rhs!, precondition!

# But we need to define the precondition! methods here
function precondition!(p, preconditioner::DistributedFFTBasedPoissonSolver, r, args...)
    compute_preconditioner_rhs!(preconditioner, r)
    solve!(p, preconditioner)
    return p
end

function precondition!(p, preconditioner::DistributedFourierTridiagonalPoissonSolver, r, args...)
    compute_preconditioner_rhs!(preconditioner, r)
    solve!(p, preconditioner)
    return p
end

# Correctly pass architecture to determine the default weno_weight_computation
Oceananigans.Advection.default_weno_weight_computation(arch::Distributed) =
    Oceananigans.Advection.default_weno_weight_computation(child_architecture(arch))


end # module
