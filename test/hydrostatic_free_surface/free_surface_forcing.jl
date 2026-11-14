include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

# A horizontally uniform Fη drives no flow, so from rest η(t) = Fη t exactly
# for every free surface and time stepper.
function displacement_after_uniform_forcing(grid, free_surface, timestepper, Fη, Δt, Nsteps)
    forcing = (; η = Forcing((x, y, z, t) -> Fη))
    model = HydrostaticFreeSurfaceModel(grid; free_surface, timestepper, forcing)

    for _ in 1:Nsteps
        time_step!(model, Δt)
    end

    return Array(interior(model.free_surface.displacement))
end

# A localized, time-dependent Fη excites every horizontal wavenumber and exercises the clock.
# The FFT and PCG solvers solve the same implicit free surface problem, so they must agree.
function displacement_after_localized_forcing(grid, free_surface, Δt, Nsteps)
    Lx, Ly = grid.Lx, grid.Ly
    T = 20Δt
    forcing = (; η = Forcing((x, y, z, t) -> 1e-3 * exp(-((x - Lx/2)^2 + (y - Ly/2)^2) / (Lx/10)^2) * (1 + sin(2π * t / T))))
    model = HydrostaticFreeSurfaceModel(grid; free_surface, forcing)

    for _ in 1:Nsteps
        time_step!(model, Δt)
    end

    return Array(interior(model.free_surface.displacement))
end

@testset "Free surface forcing" begin
    Δt = 1 # within the barotropic CFL of the explicit free surface on the grids below
    Nsteps = 10

    for arch in archs, FT in float_types
        A = typeof(arch)
        Fη = convert(FT, 1e-3)

        grid = RectilinearGrid(arch, FT;
                               size = (8, 8, 4),
                               x = (0, 1kilometer),
                               y = (0, 1kilometer),
                               z = (-100, 0),
                               topology = (Periodic, Periodic, Bounded))

        free_surfaces = (
            "ImplicitFreeSurface(:FastFourierTransform)"            => () -> ImplicitFreeSurface(solver_method = :FastFourierTransform),
            "ImplicitFreeSurface(:PreconditionedConjugateGradient)" => () -> ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient),
            "SplitExplicitFreeSurface"                              => () -> SplitExplicitFreeSurface(grid; substeps = 10),
            "ExplicitFreeSurface"                                   => () -> ExplicitFreeSurface(),
        )

        for (name, build_free_surface) in free_surfaces, timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
            @info "Testing uniform η forcing with $name [$A, $FT, $timestepper]..."

            displacement = displacement_after_uniform_forcing(grid, build_free_surface(), timestepper, Fη, Δt, Nsteps)
            exact_displacement = Fη * Nsteps * Δt
            relative_error = maximum(abs, displacement .- exact_displacement) / exact_displacement
            @info "    maximum |η - Fη t| / (Fη t): $relative_error"

            @test relative_error < 100 * eps(FT)
        end
    end

    for arch in archs
        A = typeof(arch)
        @info "Testing localized η forcing with FFT and PCG implicit free surface solvers [$A]..."

        grid = RectilinearGrid(arch;
                               size = (32, 32, 4),
                               x = (0, 1kilometer),
                               y = (0, 1kilometer),
                               z = (-100, 0),
                               topology = (Periodic, Periodic, Bounded))

        fft_free_surface = ImplicitFreeSurface(solver_method = :FastFourierTransform)
        pcg_free_surface = ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient,
                                               abstol = 1e-15, reltol = 0, maxiter = 32^2)

        fft_displacement = displacement_after_localized_forcing(grid, fft_free_surface, Δt, Nsteps)
        pcg_displacement = displacement_after_localized_forcing(grid, pcg_free_surface, Δt, Nsteps)

        @info "    maximum(abs, η_fft - η_pcg): $(maximum(abs, fft_displacement .- pcg_displacement))"
        @info "    maximum(abs, η_pcg): $(maximum(abs, pcg_displacement))"

        @test maximum(abs, pcg_displacement) > 0
        @test maximum(abs, fft_displacement .- pcg_displacement) < 1e-12 * maximum(abs, pcg_displacement)
    end
end
