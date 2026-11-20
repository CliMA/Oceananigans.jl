include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Statistics
using Oceananigans.Operators
using Oceananigans.Grids: inactive_cell
using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    ImplicitFreeSurface,
    FFTImplicitFreeSurfaceSolver,
    PCGImplicitFreeSurfaceSolver,
    step_free_surface!,
    implicit_free_surface_linear_operation!,
    integrated_x_area,
    integrated_y_area

using Oceananigans.Solvers: solve!


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
end

@testset "Stored vertically integrated lateral areas" begin
    for arch in archs
        A = typeof(arch)
        @info "Testing stored vertically integrated lateral areas [$A]..."

        lazy_areas(grid) = (Field(KernelFunctionOperation{Face, Center, Nothing}(integrated_x_area, grid)),
                            Field(KernelFunctionOperation{Center, Face, Nothing}(integrated_y_area, grid)))

        function areas_match(∫ᶻA, grid)
            ax, ay = lazy_areas(grid)
            return Array(interior(∫ᶻA.xᶠᶜᶜ)) == Array(interior(ax)) &&
                   Array(interior(∫ᶻA.yᶜᶠᶜ)) == Array(interior(ay))
        end

        bottom(x, y) = -1000 + 600 * exp(-(x^2 + y^2) / 2e5^2)

        static_grid = ImmersedBoundaryGrid(RectilinearGrid(arch, size = (16, 12, 4), x = (-5e5, 5e5), y = (-5e5, 5e5),
                                                           z = (-1000, 0), topology = (Bounded, Periodic, Bounded)),
                                           GridFittedBottom(bottom))

        free_surface = ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient)
        model = HydrostaticFreeSurfaceModel(static_grid; free_surface)
        @test areas_match(model.free_surface.implicit_step_solver.vertically_integrated_lateral_areas, static_grid)

        # With z-star the column depth follows the free surface, so the areas are refreshed before each solve
        zstar_grid = ImmersedBoundaryGrid(RectilinearGrid(arch, size = (16, 12, 4), x = (-5e5, 5e5), y = (-5e5, 5e5),
                                                          z = MutableVerticalDiscretization(collect(range(-1000, 0, length = 5))),
                                                          topology = (Bounded, Bounded, Bounded)),
                                          GridFittedBottom(bottom))

        free_surface = ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient)
        model = HydrostaticFreeSurfaceModel(zstar_grid; free_surface, vertical_coordinate = ZStarCoordinate())
        set!(model, η = (x, y, z) -> exp(-(x^2 + y^2) / 2e5^2))
        time_step!(model, 600)

        solver = model.free_surface.implicit_step_solver
        ∫ᶻA = solver.vertically_integrated_lateral_areas
        @test !areas_match(∫ᶻA, zstar_grid) # the grid has moved since the last solve

        η = model.free_surface.displacement
        rhs = solver.right_hand_side
        solve!(η, solver, rhs, model.free_surface.gravitational_acceleration, 600)
        @test areas_match(∫ᶻA, zstar_grid)
    end
end
