include("dependencies_for_runtests.jl")
include("dependencies_for_poisson_solvers.jl")

using Oceananigans.Solvers: MultigridPreconditioner, ConjugateGradientPoissonSolver,
                            DiagonallyDominantPreconditioner, compute_symmetric_laplacian!,
                            iteration
using Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!
using Oceananigans.Grids: inactive_cell
using Random: seed!

# Apply the level-1 conductance stencil on the CPU; the multigrid fine-level operator must
# reproduce the symmetric volume-weighted Laplacian exactly on every supported grid.
function multigrid_stencil_laplacian(preconditioner, ϕa)
    level = first(preconditioner.levels)
    Cx, Cy, Cz, D = Array(level.Cx), Array(level.Cy), Array(level.Cz), Array(level.D)
    Nx, Ny, Nz = size(D)
    Aϕ = zeros(eltype(D), Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        i⁻ = i == 1 ? Nx : i - 1
        i⁺ = i == Nx ? 1 : i + 1
        j⁻ = j == 1 ? Ny : j - 1
        j⁺ = j == Ny ? 1 : j + 1
        k⁻ = max(k - 1, 1)
        k⁺ = min(k + 1, Nz)
        Aϕ[i, j, k] = D[i, j, k]  * ϕa[i, j, k] +
                      Cx[i, j, k] * ϕa[i⁻, j, k] + Cx[i+1, j, k] * ϕa[i⁺, j, k] +
                      Cy[i, j, k] * ϕa[i, j⁻, k] + Cy[i, j+1, k] * ϕa[i, j⁺, k] +
                      Cz[i, j, k] * ϕa[i, j, k⁻] + Cz[i, j, k+1] * ϕa[i, j, k⁺]
    end
    return Aϕ
end

function active_cell_mask(grid)
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, Nz = size(grid)
    return [!inactive_cell(i, j, k, cpu_grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
end

function test_multigrid_operator_equivalence(grid)
    preconditioner = MultigridPreconditioner(grid)
    ϕ = CenterField(grid)
    ∇²ϕ = CenterField(grid)

    seed!(42)
    active = active_cell_mask(grid)
    ϕa = randn(eltype(grid), size(grid)...) .* active
    set!(ϕ, ϕa)

    compute_symmetric_laplacian!(∇²ϕ, ϕ)
    Aϕ = multigrid_stencil_laplacian(preconditioner, ϕa)
    ∇²ϕa = Array(interior(∇²ϕ))

    scale = maximum(abs, ∇²ϕa)
    mismatch = maximum(abs, (Aϕ .- ∇²ϕa) .* active)

    @test mismatch <= 1000 * eps(eltype(grid)) * scale
end

function test_multigrid_pressure_solution(grid; maxiter=1000)
    reltol = abstol = sqrt(eps(eltype(grid)))
    preconditioner = MultigridPreconditioner(grid)
    solver = ConjugateGradientPoissonSolver(grid; preconditioner, reltol, abstol, maxiter)
    R, U = random_divergent_source_term(grid)

    p_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Center()))
    ϕ   = CenterField(grid, boundary_conditions=p_bcs)
    ∇²ϕ = CenterField(grid, boundary_conditions=p_bcs)

    solve_for_pressure!(ϕ, solver, nothing, U, 1)
    @test iteration(solver) < maxiter

    compute_∇²!(∇²ϕ, ϕ, architecture(grid), grid)
    @test Array(interior(∇²ϕ)) ≈ Array(interior(R))

    return iteration(solver)
end

function test_multigrid_with_nonhydrostatic_model(grid)
    preconditioner = MultigridPreconditioner(grid)
    pressure_solver = ConjugateGradientPoissonSolver(grid; preconditioner)
    model = NonhydrostaticModel(grid; pressure_solver, advection=WENO(),
                                tracers=:b, buoyancy=BuoyancyTracer())

    seed!(198)
    set!(model, u=(ξ...) -> 0.1 + 0.01 * randn(), b=(x, y, z) -> 1e-3 * z)

    δ = Field(∂x(model.velocities.u) + ∂y(model.velocities.v) + ∂z(model.velocities.w))
    for _ in 1:5
        time_step!(model, 0.01)
    end
    compute!(δ)

    @test maximum(abs, interior(δ)) < 1e-8
    @test maximum(abs, interior(model.velocities.u)) < 1e2
end

@testset "Multigrid preconditioner" begin
    @info "Testing MultigridPreconditioner..."

    for arch in archs
        @testset "Multigrid level hierarchy [$(typeof(arch))]" begin
            @info "  Testing multigrid level hierarchy [$(typeof(arch))]..."
            grid = RectilinearGrid(arch, size=(16, 16, 8), extent=(1, 1, 1))
            preconditioner = MultigridPreconditioner(grid)
            @test length(preconditioner.levels) == 4
            @test size(preconditioner.levels[2]) == (8, 8, 8)
            @test size(preconditioner.levels[4]) == (2, 2, 8)

            # sizes need not be powers of two: odd extents agglomerate a remainder cell
            odd_grid = RectilinearGrid(arch, size=(33, 17, 5), extent=(1, 1, 1))
            odd_preconditioner = MultigridPreconditioner(odd_grid)
            @test size(odd_preconditioner.levels[2]) == (17, 9, 5)

            # only non-Flat horizontal directions are coarsened, never the vertical
            flat_grid = RectilinearGrid(arch, size=(32, 8), x=(0, 1), z=(-1, 0),
                                        topology=(Bounded, Flat, Bounded))
            flat_preconditioner = MultigridPreconditioner(flat_grid)
            @test all(size(level)[3] == 8 for level in flat_preconditioner.levels)
            @test all(size(level)[2] == 1 for level in flat_preconditioner.levels)

            @test_throws ArgumentError MultigridPreconditioner(
                RectilinearGrid(arch, size=(8, 8, 8), extent=(1, 1, 1),
                                topology=(Periodic, Periodic, Periodic)))
        end

        @testset "Multigrid fine-level operator equivalence [$(typeof(arch))]" begin
            @info "  Testing multigrid fine-level operator equivalence [$(typeof(arch))]..."
            zstretched(k) = -(1 - tanh(2 * (k - 1) / 8) / tanh(2))
            chebfaces(N) = [(1 - cos(π * (i - 1) / N)) / 2 for i in 1:N+1]

            regular = RectilinearGrid(arch, size=(16, 16, 8), extent=(1, 1, 1))
            bounded = RectilinearGrid(arch, size=(16, 16, 8), x=(0, 1), y=(0, 1), z=(-1, 0),
                                      topology=(Bounded, Bounded, Bounded))
            xyz_stretched = RectilinearGrid(arch, size=(16, 16, 8), x=chebfaces(16), y=chebfaces(16),
                                            z=zstretched, topology=(Bounded, Bounded, Bounded))
            bottom(x, y) = -0.7 + 0.5 * sin(3x) * cos(2y)
            immersed = ImmersedBoundaryGrid(regular, GridFittedBottom(bottom))
            partial_cell = ImmersedBoundaryGrid(bounded, PartialCellBottom(bottom))
            sphere(x, y, z) = (x - 0.5)^2 + (y - 0.5)^2 + (z + 0.5)^2 < 0.09
            enclosed = ImmersedBoundaryGrid(bounded, GridFittedBoundary(sphere))
            latitude_longitude = ImmersedBoundaryGrid(
                LatitudeLongitudeGrid(arch, size=(16, 16, 8), longitude=(0, 60), latitude=(15, 75),
                                      z=(-3000, 0)),
                GridFittedBottom((λ, φ) -> -3000 + 2000 * exp(-((λ - 30) / 8)^2)))

            for grid in (regular, bounded, xyz_stretched, immersed, partial_cell, enclosed, latitude_longitude)
                test_multigrid_operator_equivalence(grid)
            end
        end

        @testset "Multigrid-preconditioned pressure solution [$(typeof(arch))]" begin
            @info "  Testing multigrid-preconditioned divergence-free pressure solution [$(typeof(arch))]..."
            zstretched(k) = -(1 - tanh(2 * (k - 1) / 8) / tanh(2))
            chebfaces(N) = [(1 - cos(π * (i - 1) / N)) / 2 for i in 1:N+1]
            bottom(x, y) = -0.7 + 0.5 * sin(3x) * cos(2y)

            z_stretched_immersed = ImmersedBoundaryGrid(
                RectilinearGrid(arch, size=(16, 16, 8), x=(0, 1), y=(0, 1), z=zstretched,
                                topology=(Periodic, Periodic, Bounded)),
                GridFittedBottom(bottom))

            # stretched in all three directions: no FFT-based preconditioner exists here
            xyz_stretched_immersed = ImmersedBoundaryGrid(
                RectilinearGrid(arch, size=(16, 16, 8), x=chebfaces(16), y=chebfaces(16),
                                z=zstretched, topology=(Bounded, Bounded, Bounded)),
                GridFittedBottom(bottom))

            partial_cell = ImmersedBoundaryGrid(
                RectilinearGrid(arch, size=(16, 16, 8), x=(0, 1), y=(0, 1), z=(-1, 0),
                                topology=(Periodic, Bounded, Bounded)),
                PartialCellBottom(bottom))

            odd_immersed = ImmersedBoundaryGrid(
                RectilinearGrid(arch, size=(11, 9, 7), x=(0, 1), y=(0, 1), z=(-1, 0),
                                topology=(Periodic, Bounded, Bounded)),
                GridFittedBottom(bottom))

            for grid in (z_stretched_immersed, xyz_stretched_immersed, partial_cell, odd_immersed)
                test_multigrid_pressure_solution(grid)
            end
        end

        @testset "Multigrid preconditioner with NonhydrostaticModel [$(typeof(arch))]" begin
            @info "  Testing multigrid preconditioner with NonhydrostaticModel [$(typeof(arch))]..."
            zstretched(k) = -(1 - tanh(2 * (k - 1) / 8) / tanh(2))
            chebfaces(N) = [(1 - cos(π * (i - 1) / N)) / 2 for i in 1:N+1]
            grid = ImmersedBoundaryGrid(
                RectilinearGrid(arch, size=(16, 16, 8), x=chebfaces(16), y=chebfaces(16),
                                z=zstretched, topology=(Bounded, Bounded, Bounded)),
                GridFittedBottom((x, y) -> -0.7 + 0.5 * sin(3x) * cos(2y)))
            test_multigrid_with_nonhydrostatic_model(grid)
        end
    end
end
