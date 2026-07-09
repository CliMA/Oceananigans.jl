include("dependencies_for_runtests.jl")
include("dependencies_for_poisson_solvers.jl")

using Oceananigans.Solvers: MultigridPreconditioner, ConjugateGradientPoissonSolver,
                            DiagonallyDominantPreconditioner, compute_symmetric_laplacian!,
                            iteration, precondition!, FreeSurfaceLaplacian,
                            no_gauge_enforcement!, update_free_surface_correction!
using Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using Oceananigans.Grids: inactive_cell
using Random: seed!

# Apply the level-1 conductance stencil on the CPU; the multigrid fine-level operator must
# reproduce the symmetric volume-weighted Laplacian exactly on every supported grid.
function multigrid_stencil_laplacian(preconditioner, ϕa)
    level = first(preconditioner.levels)
    T = Array(level.T)  # zero for a rigid lid; the Robin top-row correction with a free surface
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
        Dᵏ = D[i, j, k] - (k == Nz ? T[i, j] : zero(eltype(T)))
        Aϕ[i, j, k] = Dᵏ          * ϕa[i, j, k] +
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

function test_multigrid_pressure_solution(grid; maxiter=1000, float_type=eltype(grid))
    reltol = abstol = sqrt(eps(eltype(grid)))
    preconditioner = MultigridPreconditioner(grid; float_type)
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

        @testset "Multigrid preconditioner Float32 robustness [$(typeof(arch))]" begin
            @info "  Testing multigrid preconditioner Float32 robustness [$(typeof(arch))]..."
            # A deep, strongly anisotropic basin: the horizontal couplings are ~10⁻⁶ of the
            # diagonal, comparable to Float32 roundoff, so an unregularized column solve
            # produces Inf/NaN from a pivot that rounds to zero (GPU fused-multiply-add
            # rounding differs from the CPU's).
            zdeep(k) = -3000 * (1 - tanh(2 * (k - 1) / 16) / tanh(2))
            grid = ImmersedBoundaryGrid(
                LatitudeLongitudeGrid(arch, Float32, size=(32, 32, 16), longitude=(0, 60),
                                      latitude=(15, 75), z=zdeep),
                GridFittedBottom((λ, φ) -> -3000 + 2000 * exp(-((λ - 30) / 8)^2)))

            preconditioner = MultigridPreconditioner(grid)
            seed!(11)
            r = CenterField(grid)
            active = active_cell_mask(grid)
            set!(r, randn(Float32, size(grid)...) .* active)
            z = CenterField(grid)
            precondition!(z, preconditioner, r)
            @test all(isfinite, Array(interior(z)))

            # Float32 cannot reach the usual sqrt(eps) discretization accuracy on a grid
            # this ill-conditioned, so only convergence and a usable solution are asserted.
            maxiter = 500
            solver = ConjugateGradientPoissonSolver(grid; preconditioner, maxiter,
                                                    reltol=sqrt(eps(Float32)), abstol=sqrt(eps(Float32)))
            R, U = random_divergent_source_term(grid)
            ϕ   = CenterField(grid)
            ∇²ϕ = CenterField(grid)
            solve_for_pressure!(ϕ, solver, nothing, U, 1)
            @test iteration(solver) < maxiter
            @test all(isfinite, Array(interior(ϕ)))

            compute_∇²!(∇²ϕ, ϕ, architecture(grid), grid)
            a = Array(interior(∇²ϕ))
            b = Array(interior(R))
            @test sqrt(sum(abs2, a .- b) / sum(abs2, b)) < 0.1
        end

        @testset "Multigrid reduced-precision cycle [$(typeof(arch))]" begin
            @info "  Testing multigrid reduced-precision cycle [$(typeof(arch))]..."
            zstretched(k) = -(1 - tanh(2 * (k - 1) / 8) / tanh(2))
            chebfaces(N) = [(1 - cos(π * (i - 1) / N)) / 2 for i in 1:N+1]
            bottom(x, y) = -0.7 + 0.5 * sin(3x) * cos(2y)

            # a Float64 grid with the V-cycle stored and smoothed in Float32: the outer
            # conjugate gradient iteration must still reach its Float64 tolerance
            grid = ImmersedBoundaryGrid(
                RectilinearGrid(arch, size=(16, 16, 8), x=chebfaces(16), y=chebfaces(16),
                                z=zstretched, topology=(Bounded, Bounded, Bounded)),
                GridFittedBottom(bottom))

            preconditioner = MultigridPreconditioner(grid, float_type=Float32)
            @test eltype(first(preconditioner.levels).D) === Float32
            @test occursin("Float32 cycle", summary(preconditioner))

            test_multigrid_pressure_solution(grid, float_type=Float32)
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

            # a conjugate gradient solver whose linear operation lacks the Robin pressure
            # boundary condition cannot support a free surface: fail at construction time
            pressure_solver = ConjugateGradientPoissonSolver(grid; preconditioner=MultigridPreconditioner(grid))
            free_surface = ImplicitFreeSurface(gravitational_acceleration=10)
            @test_throws ArgumentError NonhydrostaticModel(grid; pressure_solver, free_surface)
        end

        @testset "Multigrid preconditioner with a free surface [$(typeof(arch))]" begin
            @info "  Testing multigrid preconditioner with an implicit free surface [$(typeof(arch))]..."
            bottom(x, y) = -0.7 + 0.5 * sin(3x) * cos(2y)
            grid = ImmersedBoundaryGrid(
                RectilinearGrid(arch, size=(16, 16, 8), x=(0, 1), y=(0, 1), z=(-1, 0),
                                topology=(Periodic, Periodic, Bounded)),
                GridFittedBottom(bottom))

            # fine-level operator with the Robin top-row correction ≡ FreeSurfaceLaplacian
            reference_model = NonhydrostaticModel(grid; free_surface=ImplicitFreeSurface(gravitational_acceleration=10))
            free_surface = reference_model.free_surface
            Δt = 0.037
            preconditioner = MultigridPreconditioner(grid)
            update_free_surface_correction!(preconditioner, free_surface, Δt)

            seed!(42)
            active = active_cell_mask(grid)
            ϕa = randn(eltype(grid), size(grid)...) .* active
            ϕ = CenterField(grid)
            set!(ϕ, ϕa)
            ∇²ϕ = CenterField(grid)
            FreeSurfaceLaplacian()(∇²ϕ, ϕ, free_surface, Δt)
            Aϕ = multigrid_stencil_laplacian(preconditioner, ϕa)
            ∇²ϕa = Array(interior(∇²ϕ))
            scale = maximum(abs, ∇²ϕa)
            @test maximum(abs, (Aϕ .- ∇²ϕa) .* active) <= 1000 * eps(eltype(grid)) * scale

            # the correction is refreshed when the time step changes
            fine_correction = copy(Array(first(preconditioner.levels).T))
            update_free_surface_correction!(preconditioner, free_surface, 2Δt)
            @test maximum(abs, fine_correction .- Array(first(preconditioner.levels).T)) > 0

            # stepped free-surface solution agrees with the FT-free-surface-preconditioned default
            function stepped_free_surface_fields(model; Δt=0.01, N=10)
                set!(model.free_surface.displacement, (x, y, z) -> 0.05 * cospi(x) * cospi(y))
                for _ in 1:N
                    time_step!(model, Δt)
                end
                return (Array(interior(model.free_surface.displacement)),
                        Array(interior(model.velocities.u)),
                        Array(interior(model.velocities.w)))
            end

            implicit_free_surface() = ImplicitFreeSurface(gravitational_acceleration=10)
            η_ft, u_ft, w_ft = stepped_free_surface_fields(
                NonhydrostaticModel(grid; free_surface=implicit_free_surface()))

            multigrid_solver(g) = ConjugateGradientPoissonSolver(g;
                                                                 linear_operation = FreeSurfaceLaplacian(),
                                                                 preconditioner = MultigridPreconditioner(g),
                                                                 enforce_gauge_condition! = no_gauge_enforcement!)
            mg_solver = multigrid_solver(grid)
            η_mg, u_mg, w_mg = stepped_free_surface_fields(
                NonhydrostaticModel(grid; free_surface=implicit_free_surface(), pressure_solver=mg_solver))
            @test iteration(mg_solver) < 30
            @test isapprox(η_mg, η_ft, atol=1e-8)
            @test isapprox(u_mg, u_ft, atol=1e-8)
            @test isapprox(w_mg, w_ft, atol=1e-8)

            # free surface + immersed bottom on an xyz-stretched grid, where no FFT-based
            # solver exists: agreement with a DiagonallyDominant-preconditioned reference
            zstretched(k) = -(1 - tanh(2 * (k - 1) / 8) / tanh(2))
            chebfaces(N) = [(1 - cos(π * (i - 1) / N)) / 2 for i in 1:N+1]
            stretched = ImmersedBoundaryGrid(
                RectilinearGrid(arch, size=(16, 16, 8), x=chebfaces(16), y=chebfaces(16),
                                z=zstretched, topology=(Bounded, Bounded, Bounded)),
                GridFittedBottom(bottom))

            stretched_mg_solver = multigrid_solver(stretched)
            η_smg, u_smg, w_smg = stepped_free_surface_fields(
                NonhydrostaticModel(stretched; free_surface=implicit_free_surface(),
                                    pressure_solver=stretched_mg_solver))

            reference_solver = ConjugateGradientPoissonSolver(stretched;
                                                              linear_operation = FreeSurfaceLaplacian(),
                                                              preconditioner = DiagonallyDominantPreconditioner(),
                                                              enforce_gauge_condition! = no_gauge_enforcement!)
            η_dd, u_dd, w_dd = stepped_free_surface_fields(
                NonhydrostaticModel(stretched; free_surface=implicit_free_surface(),
                                    pressure_solver=reference_solver))

            @test iteration(stretched_mg_solver) < iteration(reference_solver)
            @test isapprox(η_smg, η_dd, atol=1e-8)
            @test isapprox(u_smg, u_dd, atol=1e-8)
            @test isapprox(w_smg, w_dd, atol=1e-8)
        end
    end
end
