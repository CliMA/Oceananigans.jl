include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Statistics
using Oceananigans.Operators
using Oceananigans.Grids: inactive_cell
using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    ImplicitFreeSurface,
    FFTImplicitFreeSurfaceSolver,
    PCGImplicitFreeSurfaceSolver,
    DiagonallyDominantInversePreconditioner,
    step_free_surface!,
    implicit_free_surface_linear_operation!

using Oceananigans.Solvers: precondition!


function set_simple_divergent_velocity!(model)
    # Create a divergent velocity
    grid = model.grid

    u, v, w = model.velocities
    η = model.free_surface.displacement

    u .= 0
    v .= 0
    η .= 0

    # pick a surface cell at the middle of the domain
    i, j, k = Int(floor(grid.Nx / 2)) + 1, Int(floor(grid.Ny / 2)) + 1, grid.Nz
    inactive_cell(i, j, k, grid) && error("The nudged cell at ($i, $j, $k) is inactive.")

    Δy = @allowscalar Δyᶜᶠᶜ(i, j, k, grid)
    Δz = @allowscalar Δzᶜᶠᶜ(i, j, k, grid)

    # We prescribe the value of the zonal transport in a cell, i.e., `u * Δy * Δz`. This
    # way `norm(rhs)` of the free-surface solver does not depend on the grid extent/resolution.
    transport = 1e5 # m³ s⁻¹
    @allowscalar u[i, j, k] = transport / (Δy * Δz)

    update_state!(model)

    return nothing
end

function run_implicit_free_surface_solver_tests(arch, grid, free_surface)
    Δt = 900

    # Create a model
    model = HydrostaticFreeSurfaceModel(grid; momentum_advection = nothing, free_surface)

    set_simple_divergent_velocity!(model)
    step_free_surface!(model.free_surface, model, model.timestepper, Δt)

    η = model.free_surface.displacement
    @info "PCG implicit free surface solver test, norm(η_pcg): $(norm(η)), maximum(abs, η_pcg): $(maximum(abs, η))"

    # Extract right hand side "truth"
    right_hand_side = model.free_surface.implicit_step_solver.right_hand_side
    if !(right_hand_side isa Field)
        rhs = Field{Center, Center, Nothing}(grid)
        set!(rhs, reshape(right_hand_side, model.free_surface.implicit_step_solver.matrix_iterative_solver.problem_size...))
        right_hand_side = rhs
    end

    # Compute left hand side "solution"
    g = Oceananigans.defaults.gravitational_acceleration
    η = model.free_surface.displacement

    ∫ᶻ_Axᶠᶜᶜ = KernelFunctionOperation{Face, Center, Nothing}(Oceananigans.Models.HydrostaticFreeSurfaceModels.integrated_x_area, grid)
    ∫ᶻ_Ayᶜᶠᶜ = KernelFunctionOperation{Center, Face, Nothing}(Oceananigans.Models.HydrostaticFreeSurfaceModels.integrated_y_area, grid)

    left_hand_side = ZFaceField(grid, indices = (:, :, grid.Nz + 1))
    implicit_free_surface_linear_operation!(left_hand_side, η, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

    # Compare
    extrema_tolerance = 1e-9
    std_tolerance = 1e-9

    @show norm(left_hand_side)
    @show norm(right_hand_side)

    @allowscalar begin
        @test maximum(abs, interior(left_hand_side) .- interior(right_hand_side)) < extrema_tolerance
        @test std(interior(left_hand_side) .- interior(right_hand_side)) < std_tolerance
    end

    return model.free_surface.implicit_step_solver
end

@testset "Implicit free surface solver tests" begin
    for arch in archs
        A = typeof(arch)

        rectilinear_grid = RectilinearGrid(arch, size = (128, 2, 5),
                                           x = (-5000kilometers, 5000kilometers),
                                           y = (0, 100kilometers),
                                           z = (-500, 0),
                                           halo = (3, 2, 3),
                                           topology = (Bounded, Periodic, Bounded))

        Lz = rectilinear_grid.Lz
        width = rectilinear_grid.Lx / 20

        bump(x, y) = - Lz * (1 - 0.2 * exp(-x^2 / 2width^2))

        underlying_grid = RectilinearGrid(arch, size = (128, 2, 5),
                                          x = (-5000kilometers, 5000kilometers),
                                          y = (0, 100kilometers),
                                          z = [-500, -300, -220, -170, -60, 0],
                                          halo = (3, 2, 3),
                                          topology = (Bounded, Periodic, Bounded))

        bumpy_vertically_stretched_rectilinear_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bump))

        lat_lon_grid = LatitudeLongitudeGrid(arch, size = (50, 50, 5),
                                             longitude = (-20, 30),
                                             latitude = (-10, 40),
                                             z = (-4000, 0))

        for grid in (rectilinear_grid, bumpy_vertically_stretched_rectilinear_grid, lat_lon_grid)
            G = string(nameof(typeof(grid)))

            @info "Testing PreconditionedConjugateGradient implicit free surface solver [$A, $G]..."
            free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient,
                                               abstol=1e-15, reltol=0)
            run_implicit_free_surface_solver_tests(arch, grid, free_surface)

            @info "Testing PreconditionedConjugateGradient implicit free surface solver without preconditioner [$A, $G]..."
            free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient,
                                               preconditioner=nothing, abstol=1e-15, reltol=0)
            run_implicit_free_surface_solver_tests(arch, grid, free_surface)

            @info "Testing PreconditionedConjugateGradient implicit free surface solver with DiagonallyDominantInversePreconditioner [$A, $G]..."
            free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient,
                                               preconditioner=DiagonallyDominantInversePreconditioner(),
                                               abstol=1e-15, reltol=0)
            run_implicit_free_surface_solver_tests(arch, grid, free_surface)
        end

        @info "Testing implicit free surface solvers compared to FFT [$A]..."

        pcg_free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient,
                                               abstol=1e-15, reltol=0, maxiter=128^2)

        fft_free_surface = ImplicitFreeSurface(solver_method=:FastFourierTransform)

        pcg_model = HydrostaticFreeSurfaceModel(rectilinear_grid;
                                                momentum_advection = nothing,
                                                free_surface = pcg_free_surface)

        fft_model = HydrostaticFreeSurfaceModel(rectilinear_grid;
                                                momentum_advection = nothing,
                                                free_surface = fft_free_surface)

        @test fft_model.free_surface.implicit_step_solver isa FFTImplicitFreeSurfaceSolver
        @test pcg_model.free_surface.implicit_step_solver isa PCGImplicitFreeSurfaceSolver

        # default preconditioner of the PCG solver on a non-rectilinear grid
        lat_lon_pcg_model = HydrostaticFreeSurfaceModel(lat_lon_grid; momentum_advection = nothing,
                                                        free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient))
        @test lat_lon_pcg_model.free_surface.implicit_step_solver.preconditioned_conjugate_gradient_solver.preconditioner isa
              DiagonallyDominantInversePreconditioner

        Δt₁ = 900
        Δt₂ = 920.0

        for m in (pcg_model, fft_model)
            set_simple_divergent_velocity!(m)
            step_free_surface!(m.free_surface, m, m.timestepper, Δt₁)
            step_free_surface!(m.free_surface, m, m.timestepper, Δt₁)
            step_free_surface!(m.free_surface, m, m.timestepper, Δt₂)
        end

        pcg_η = pcg_model.free_surface.displacement
        fft_η = fft_model.free_surface.displacement

        pcg_η_cpu = Array(interior(pcg_η))
        fft_η_cpu = Array(interior(fft_η))

        Δη_pcg = pcg_η_cpu .- fft_η_cpu

        @info "FFT/PCG/MAT implicit free surface solver comparison:"
        @info "    maximum(abs, η_pcg - η_fft): $(maximum(abs, Δη_pcg))"
        @info "    maximum(abs, η_pcg): $(maximum(abs, pcg_η_cpu))"
        @info "    maximum(abs, η_fft): $(maximum(abs, fft_η_cpu))"

        @test all(isapprox.(Δη_pcg, 0, atol=1e-15))
    end

    # A lat-lon grid with a small inlet of land at the northern boundary and a long time step:
    # the coefficients of the implicit free-surface operator vary strongly from cell to cell.
    # The conjugate gradient method needs a symmetric preconditioner to converge there.
    for arch in archs
        A = typeof(arch)
        @info "Testing the symmetry of DiagonallyDominantInversePreconditioner [$A]..."

        Nx, Ny = 45, 20
        underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, 4), longitude = (0, 360), latitude = (-80, 80), z = (-4000, 0))
        land = [1 1 0 0 1 1
                1 0 0 0 1 1
                1 1 1 1 1 1]
        depth = fill(-4000.0, Nx, Ny)
        depth[28:33, 18:20] .= ifelse.(land' .== 1, 0.0, -4000.0)
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(depth))
        Δt = 2400

        iterations = map((nothing, DiagonallyDominantInversePreconditioner())) do preconditioner
            free_surface = ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient; preconditioner)
            model = HydrostaticFreeSurfaceModel(grid; free_surface)
            set!(model, u = (λ, φ, z) -> 0.1 * cosd(φ))
            time_step!(model, Δt)
            model.free_surface.implicit_step_solver.preconditioned_conjugate_gradient_solver
        end

        unpreconditioned, preconditioned = iterations
        @test preconditioned.iteration < preconditioned.maxiter
        @test preconditioned.iteration < unpreconditioned.iteration

        # ⟨x, P y⟩ = ⟨P x, y⟩ for random x and y
        free_surface = ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient,
                                           preconditioner = DiagonallyDominantInversePreconditioner())
        model = HydrostaticFreeSurfaceModel(grid; free_surface)
        solver = model.free_surface.implicit_step_solver
        ∫ᶻA = solver.vertically_integrated_lateral_areas
        P = solver.preconditioned_conjugate_gradient_solver.preconditioner
        g = model.free_surface.gravitational_acceleration

        x, y, Px, Py = Tuple(similar(solver.right_hand_side) for _ in 1:4)
        for f in (x, y, Px, Py)
            fill!(parent(f), 0)
        end
        interior(x) .= on_architecture(arch, rand(Nx, Ny, 1))
        interior(y) .= on_architecture(arch, rand(Nx, Ny, 1))

        precondition!(Px, P, x, ∫ᶻA.xᶠᶜᶜ, ∫ᶻA.yᶜᶠᶜ, g, Δt)
        precondition!(Py, P, y, ∫ᶻA.xᶠᶜᶜ, ∫ᶻA.yᶜᶠᶜ, g, Δt)

        x_Py = sum(Array(interior(x)) .* Array(interior(Py)))
        Px_y = sum(Array(interior(Px)) .* Array(interior(y)))
        @test x_Py ≈ Px_y rtol = 1e-12
    end
end
