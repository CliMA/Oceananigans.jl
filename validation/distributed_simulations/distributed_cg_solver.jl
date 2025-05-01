using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, DiagonallyDominantPreconditioner, compute_laplacian!
using Oceananigans.DistributedComputations: reconstruct_global_field, @handshake
# using Oceananigans.Solvers: KrylovSolver
using Statistics
using Random
Random.seed!(123)

arch = Distributed(CPU())
Nx = Ny = Nz = 32
topology = (Periodic, Periodic, Periodic)

x = y = z = (0, 2π)

local_grid  = RectilinearGrid(arch; x, y, z, topology, size=(Nx, Ny, Nz))
global_grid = RectilinearGrid(;     x, y, z, topology, size=(Nx, Ny, Nz))

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

xpcg_local  = CenterField(local_grid)
xpcg_global = CenterField(global_grid)

dot_parallel = Oceananigans.Solvers._dot(b_local, b_local)
dot_serial   = Oceananigans.Solvers._dot(b_global, b_global)

norm_parallel = Oceananigans.Solvers._norm(b_local)
norm_serial   = Oceananigans.Solvers._norm(b_global)

@handshake @info arch.local_rank, dot_parallel, dot_serial, dot_parallel ≈ dot_serial
@handshake @info arch.local_rank, norm_parallel, norm_serial, norm_parallel ≈ norm_serial

reltol = abstol = 1e-7
pcg_local = ConjugateGradientPoissonSolver(local_grid, maxiter=20; reltol, abstol, preconditioner=nothing)
pcg_global = ConjugateGradientPoissonSolver(global_grid, maxiter=20; reltol, abstol, preconditioner=nothing)

# Serial solution
solve!(xpcg_global, pcg_global.conjugate_gradient_solver, b_global)
fill!(xpcg_global, 0)
t_pcg = @timed solve!(xpcg_global, pcg_global.conjugate_gradient_solver, b_global)
@handshake @info "PCG iteration $(pcg_global.conjugate_gradient_solver.iteration), time $(t_pcg.time)"

# Distributed solution
solve!(xpcg_local, pcg_local.conjugate_gradient_solver, b_local)
fill!(xpcg_local, 0)
t_pcg = @timed solve!(xpcg_local, pcg_local.conjugate_gradient_solver, b_local)
@handshake @info "PCG iteration $(pcg_local.conjugate_gradient_solver.iteration), time $(t_pcg.time)"

xpcg_reconstruct = reconstruct_global_field(xpcg_local)

@handshake @info arch.local_rank maximum(abs, interior(xpcg_reconstruct))
@handshake @info arch.local_rank maximum(abs, interior(xpcg_global))
@handshake @info arch.local_rank maximum(abs, interior(xpcg_reconstruct) .- interior(xpcg_global))


