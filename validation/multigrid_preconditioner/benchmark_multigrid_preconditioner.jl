using Oceananigans
using Oceananigans.Solvers: MultigridPreconditioner, ConjugateGradientPoissonSolver,
                            DiagonallyDominantPreconditioner, ColumnwiseTridiagonalPreconditioner,
                            compute_symmetric_laplacian!, fft_poisson_solver, iteration, solve!
using Oceananigans.Grids: inactive_cell
using Oceananigans.Architectures: on_architecture
using Random
using Printf

arch = CPU()

# Solve V∇²ϕ = b with b = V∇²w for random w, so the right-hand side is consistent by
# construction, and report (iterations, wall time, true relative residual).
function benchmark_preconditioner(grid, preconditioner; reltol=1e-8, maxiter=2000, repetitions=3)
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(grid)
    active = [!inactive_cell(i, j, k, cpu_grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz]

    Random.seed!(3)
    w = CenterField(grid)
    set!(w, randn(eltype(grid), size(grid)...) .* active)

    b = CenterField(grid)
    compute_symmetric_laplacian!(b, w)

    solver = ConjugateGradientPoissonSolver(grid; preconditioner, reltol, abstol=0, maxiter)
    ϕ = CenterField(grid)
    solve!(ϕ, solver.conjugate_gradient_solver, b)  # warm up

    wall_time = minimum((fill!(parent(ϕ), 0); @elapsed solve!(ϕ, solver.conjugate_gradient_solver, b))
                        for _ in 1:repetitions)

    Aϕ = CenterField(grid)
    compute_symmetric_laplacian!(Aϕ, ϕ)
    residual = maximum(abs, Array(interior(Aϕ)) .- Array(interior(b))) / maximum(abs, Array(interior(b)))

    return iteration(solver), wall_time, residual
end

function compare_preconditioners(name, grid, preconditioners)
    println("=== ", name, " ===")
    for (preconditioner_name, preconditioner) in preconditioners
        preconditioner === nothing && continue
        iterations, wall_time, residual = benchmark_preconditioner(grid, preconditioner)
        @printf "%-22s iterations = %5d   wall time = %8.1f ms   true relative residual = %.2e\n" preconditioner_name iterations 1000wall_time residual
    end
    println()
end

zface(k) = -(1 - tanh(2 * (k - 1) / 32) / tanh(2))
chebfaces(N, a, b) = [a + (b - a) * (1 - cos(π * (i - 1) / N)) / 2 for i in 1:N+1]
sinusoidal_bottom(x, y) = -0.7 + 0.5 * sin(6x) * cos(4y)

# 1. z-stretched with sinusoidal immersed bottom: FFT preconditioning available
underlying = RectilinearGrid(arch, size=(64, 64, 32), x=(0, 1), y=(0, 1), z=zface,
                             topology=(Periodic, Periodic, Bounded))
grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(sinusoidal_bottom))
compare_preconditioners("z-stretched immersed bottom 64×64×32", grid,
                        [("DiagonallyDominant", DiagonallyDominantPreconditioner()),
                         ("ColumnwiseTridiagonal", ColumnwiseTridiagonalPreconditioner(grid)),
                         ("Multigrid", MultigridPreconditioner(grid)),
                         ("FFT", fft_poisson_solver(underlying))])

# 2. stretched in all three directions: no FFT-based preconditioner exists
underlying = RectilinearGrid(arch, size=(64, 64, 32), x=chebfaces(64, 0, 1), y=chebfaces(64, 0, 1),
                             z=zface, topology=(Bounded, Bounded, Bounded))
grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(sinusoidal_bottom))
compare_preconditioners("xyz-stretched immersed bottom 64×64×32 (no FFT possible)", grid,
                        [("DiagonallyDominant", DiagonallyDominantPreconditioner()),
                         ("ColumnwiseTridiagonal", ColumnwiseTridiagonalPreconditioner(grid)),
                         ("Multigrid", MultigridPreconditioner(grid))])

# 3. thin ridge nearly separating two basins: hard case for global spectral preconditioning
underlying = RectilinearGrid(arch, size=(64, 64, 32), x=(0, 1), y=(0, 1), z=(-1, 0),
                             topology=(Periodic, Periodic, Bounded))
ridge(x, y) = abs(x - 0.5) < 0.016 ? -0.05 : -1.0
grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(ridge))
compare_preconditioners("thin ridge to 95% depth 64×64×32", grid,
                        [("DiagonallyDominant", DiagonallyDominantPreconditioner()),
                         ("Multigrid", MultigridPreconditioner(grid)),
                         ("FFT", fft_poisson_solver(underlying))])

# 4. isotropic aspect ratio with an immersed sphere: hard case for column preconditioning
underlying = RectilinearGrid(arch, size=(64, 64, 64), x=(0, 1), y=(0, 1), z=(-1, 0),
                             topology=(Bounded, Bounded, Bounded))
sphere(x, y, z) = (x - 0.5)^2 + (y - 0.5)^2 + (z + 0.5)^2 < 0.09
grid = ImmersedBoundaryGrid(underlying, GridFittedBoundary(sphere))
compare_preconditioners("isotropic immersed sphere 64³", grid,
                        [("ColumnwiseTridiagonal", ColumnwiseTridiagonalPreconditioner(grid)),
                         ("Multigrid", MultigridPreconditioner(grid)),
                         ("FFT", fft_poisson_solver(underlying))])

# 5. realistic latitude-longitude basin: no FFT-based preconditioner exists
zdeep(k) = -3000 * (1 - tanh(2 * (k - 1) / 32) / tanh(2))
underlying = LatitudeLongitudeGrid(arch, size=(64, 64, 32), longitude=(0, 60), latitude=(15, 75), z=zdeep)
bathymetry(λ, φ) = -3000 + 2000 * exp(-((λ - 30) / 8)^2) + 800 * sind(6λ) * cosd(4φ)
grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(bathymetry))
compare_preconditioners("latitude-longitude 3km basin 64×64×32 (no FFT possible)", grid,
                        [("DiagonallyDominant", DiagonallyDominantPreconditioner()),
                         ("ColumnwiseTridiagonal", ColumnwiseTridiagonalPreconditioner(grid)),
                         ("Multigrid", MultigridPreconditioner(grid))])
