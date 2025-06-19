using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, DiagonallyDominantPreconditioner, compute_laplacian!
using Oceananigans.DistributedComputations: reconstruct_global_field, @handshake, Equal
# using Oceananigans.Solvers: KrylovSolver
using Statistics
using Random
using LinearAlgebra: norm
Random.seed!(123)

# arch = Distributed(CPU(), partition = Partition(x=2, y=Equal()))
# arch = Distributed(CPU(), partition = Partition(x=1, y=Equal()))
arch = Distributed(CPU())
@show arch
Nx = Ny = Nz = 64
topology = (Periodic, Periodic, Periodic)

x = y = z = (0, 2π)

local_grid  = RectilinearGrid(arch; x, y, z, topology, size=(Nx, Ny, Nz))
global_grid = RectilinearGrid(;     x, y, z, topology, size=(Nx, Ny, Nz))

@inline bottom_height(x, y) = 0.5

local_grid = ImmersedBoundaryGrid(local_grid, GridFittedBottom(bottom_height))
global_grid = ImmersedBoundaryGrid(global_grid, GridFittedBottom(bottom_height))

@handshake @info rank, "build local grid"

δ = 0.1 # Gaussian width
gaussian(ξ, η, ζ) = exp(-(ξ^2 + η^2 + ζ^2) / 2δ^2)
Ngaussians = 17
Ξs = [2π * rand(3) for _ = 1:Ngaussians]

function many_gaussians(ξ, η, ζ)
    val = zero(ξ)
    for Ξ₀ in Ξs
        ξ₀, η₀, ζ₀ = Ξ₀
        val += gaussian(ξ - ξ₀, η - η₀, ζ - ζ₀)
    end
    return val
end

# Note: we use in-place transforms, so the RHS has to be AbstractArray{Complex{T}}.
# So, we first fill up "b" and then copy it into "bc = fft_solver.storage",
# which has the correct type.
b_local  = CenterField(local_grid)
b_global = CenterField(global_grid)
set!(b_local,  many_gaussians)
set!(b_global, many_gaussians)

@info "local mean $(mean(b_local))"
@info "global mean $(mean(b_global))"

b_global .-= mean(b_global)
b_local .-= mean(b_local)

@info "b_global mean after cleanup $(mean(b_global))"
@info "b_local mean after cleanup $(mean(b_local))"

xpcg_local  = CenterField(local_grid)
xpcg_global = CenterField(global_grid)

dot_parallel = Oceananigans.Solvers.dot(b_local, b_local)
dot_serial   = Oceananigans.Solvers.dot(b_global, b_global)

norm_parallel = Oceananigans.Solvers.norm(b_local)
norm_serial   = Oceananigans.Solvers.norm(b_global)

@info "length local", length(b_local)
@info "length global", length(b_global)

@info arch.local_rank, dot_parallel, dot_serial, dot_parallel ≈ dot_serial
@info arch.local_rank, norm_parallel, norm_serial, norm_parallel ≈ norm_serial

reltol = abstol = 1e-7
# preconditioner_global = FFTBasedPoissonSolver(global_grid)
# preconditioner_local  = Oceananigans.DistributedComputations.DistributedFFTBasedPoissonSolver(global_grid, local_grid)
preconditioner_global = FFTBasedPoissonSolver(global_grid.underlying_grid)
preconditioner_local  = Oceananigans.DistributedComputations.DistributedFFTBasedPoissonSolver(global_grid.underlying_grid, local_grid.underlying_grid)
# preconditioner_local  = nothing
# preconditioner_global = nothing

pcg_local  = ConjugateGradientPoissonSolver(local_grid,  maxiter=1000; reltol, abstol, preconditioner=preconditioner_local)
pcg_global = ConjugateGradientPoissonSolver(global_grid, maxiter=1000; reltol, abstol, preconditioner=preconditioner_global)

# Serial solution
solve!(xpcg_global, pcg_global.conjugate_gradient_solver, b_global)
fill!(xpcg_global, 0)
t_pcg = @timed solve!(xpcg_global, pcg_global.conjugate_gradient_solver, b_global)
@info "PCG iteration $(pcg_global.conjugate_gradient_solver.iteration), time $(t_pcg.time), residual $(norm(pcg_global.conjugate_gradient_solver.residual))"

# Distributed solution
solve!(xpcg_local, pcg_local.conjugate_gradient_solver, b_local)
fill!(xpcg_local, 0)
t_pcg = @timed solve!(xpcg_local, pcg_local.conjugate_gradient_solver, b_local)
@info "PCG iteration $(pcg_local.conjugate_gradient_solver.iteration), time $(t_pcg.time), residual $(norm(pcg_local.conjugate_gradient_solver.residual))"

# xpcg_reconstruct = reconstruct_global_field(xpcg_local)

# @info arch.local_rank maximum(abs, interior(xpcg_reconstruct))
# @info arch.local_rank maximum(abs, interior(xpcg_global))
# @info arch.local_rank maximum(abs, interior(xpcg_reconstruct) .- interior(xpcg_global))


