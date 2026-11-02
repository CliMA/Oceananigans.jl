module DistributedComputations

export
    Distributed, Partition, Equal, Fractional,
    child_architecture, reconstruct_global_grid, partition,
    inject_halo_communication_boundary_conditions,
    DistributedFFTBasedPoissonSolver, TransposableField, mpi_initialized, mpi_rank,
    mpi_size, global_barrier, global_communicator, sanitize_environ!,
    @root, @onrank, @distribute, @handshake,
    SimplifiedGeneralizedBlockDistribution, create_cost_map, create_balanced_partition

using DocStringExtensions: TYPEDSIGNATURES
using GPUArraysCore: GPUArraysCore
using LinearAlgebra: LinearAlgebra
using MPI: MPI
using OffsetArrays: OffsetArrays, OffsetArray
using Statistics: Statistics

using Oceananigans: Oceananigans
using Oceananigans.Architectures: Architectures, AbstractArchitecture, AbstractSerialArchitecture, CPU,
                                  GPU, ndevices
using Oceananigans.BoundaryConditions: BoundaryConditions, Bottom, BottomAndTop, East, North, South,
                                       SouthAndNorth, Top, West, WestAndEast
using Oceananigans.Fields: Fields, AbstractField, CenterField, data, field
using Oceananigans.Grids: Grids, BoundaryTopology, Bounded, Center, CenterFoldConnected, CenterFolded, Closed,
                          Connected, Face, FaceFoldConnected, FaceFolded, Flat, Open, OrthogonalSphericalShellGrid,
                          Periodic, TwoSided, XYZRegularRG, grid
using Oceananigans.ImmersedBoundaries: ImmersedBoundaries, ImmersedBoundaryGrid,
                                       materialize_immersed_boundary
using Oceananigans.Solvers: Solvers, GridWithFourierTridiagonalSolver, compute_preconditioner_rhs!,
                            fft_poisson_solver
using Oceananigans.Utils: Utils


include("distributed_macros.jl")
include("sanitize_environ.jl")
include("distributed_architectures.jl")
include("partition_assemble.jl")
include("distributed_grids.jl")
include("distributed_immersed_boundaries.jl")
include("distributed_on_architecture.jl")
include("distributed_kernel_launching.jl")
include("events.jl")
include("halo_communication_bcs.jl")
include("communication_state.jl")
include("communication_buffers.jl")
include("progress_worker.jl")
include("distributed_fields.jl")
include("halo_communication.jl")
include("transposable_field.jl")
include("distributed_transpose.jl")
include("plan_distributed_transforms.jl")
include("distributed_fft_based_poisson_solver.jl")
include("distributed_fft_tridiagonal_solver.jl")
include("load_balancing.jl")

function NCCLDistributed end

Solvers.fft_poisson_solver(grid::DistributedRectilinearGrid) = fft_poisson_solver(grid, reconstruct_global_grid(grid))

Solvers.fft_poisson_solver(local_grid::DistributedRectilinearGrid, global_grid::XYZRegularRG) =
    DistributedFFTBasedPoissonSolver(global_grid, local_grid)

Solvers.fft_poisson_solver(local_grid::DistributedRectilinearGrid, global_grid::GridWithFourierTridiagonalSolver) =
    DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)

# But we need to define the precondition! methods here
function Solvers.precondition!(p, preconditioner::DistributedFFTBasedPoissonSolver, r, args...)
    compute_preconditioner_rhs!(preconditioner, r)
    solve!(p, preconditioner)
    return p
end

function Solvers.precondition!(p, preconditioner::DistributedFourierTridiagonalPoissonSolver, r, args...)
    compute_preconditioner_rhs!(preconditioner, r)
    solve!(p, preconditioner)
    return p
end

# Correctly pass architecture to determine the default weno_weight_computation
Oceananigans.Advection.default_weno_weight_computation(arch::Distributed) =
    Oceananigans.Advection.default_weno_weight_computation(child_architecture(arch))


end # module
