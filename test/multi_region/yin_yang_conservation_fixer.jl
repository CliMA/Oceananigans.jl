include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "yin_yang_test_utils.jl"))

using Oceananigans.MultiRegion: HydrostaticYinYangConservationFixer, tracer_names

@testset "Yin-Yang conservation fixer of HydrostaticFreeSurfaceModel [$(summary(arch))]" for arch in archs
    # The volume and the content of `s` are conserved apart from these sources, and `one` stays uniform
    source = 1e-6 # per second, per unit volume
    flux = 1e-4   # upward, per unit area
    sphere = YinYangGrid(arch; size = (48, 16, 2), radius = 1e6, z = MutableVerticalDiscretization((-1000, 0)), halo = (4, 4, 4))
    grid = ImmersedBoundaryGrid(sphere, GridFittedBottom((λ, φ) -> ifelse((abs(λ) < 20) & (φ > 0), 0, -1000)))
    boundary_conditions = (; s = FieldBoundaryConditions(top = FluxBoundaryCondition(flux)))
    Δt, Nt = 10, 4

    # The implicit solve is converged to rounding: its residual would change the volume of each column
    for (free_surface, timestepper) in ((SplitExplicitFreeSurface(substeps = 4, extend_halos = false), :SplitRungeKutta3),
                                        (ExplicitFreeSurface(), :QuasiAdamsBashforth2),
                                        (ImplicitFreeSurface(reltol = 1e-14, abstol = 0), :SplitRungeKutta3))
        model = HydrostaticFreeSurfaceModel(grid; free_surface, timestepper, boundary_conditions, momentum_advection = nothing,
                                            buoyancy = nothing, tracers = (:one, :s), forcing = (; s = (λ, φ, z, t) -> source),
                                            closure = HorizontalScalarDiffusivity(κ = 100))
        set!(model; u = (λ, φ, z) -> 20cosd(φ), η = (λ, φ, z) -> cosd(λ) * cosd(φ), one = 1, s = (λ, φ, z) -> 1 + sind(φ) / 2)

        # A second call replaces the first fixer, so the seam budget is removed once
        simulation = Simulation(model; Δt, stop_iteration = Nt, verbose = false)
        add_conservation_fixer!(simulation)
        add_conservation_fixer!(simulation)
        @test count(callback -> callback.func isa HydrostaticYinYangConservationFixer, values(simulation.callbacks)) == 1
        @test summary(simulation.callbacks[:conservation_fixer].func) == "YinYangConservationFixer of the volume and the tracers (:one, :s)"

        V₀, S₀ = active_volume(model), active_content(model.tracers.s)
        run!(simulation)
        @test active_volume(model) ≈ V₀ rtol = 1e-14
        @test maximum(abs, model.tracers.one - 1) < 1e-13
        A = owned_sum(top_area, grid)
        @test active_content(model.tracers.s) ≈ S₀ + Nt * Δt * (source * V₀ - flux * A) rtol = 1e-14
    end

    @test_throws "add_conservation_fixer!" YinYangConservationFixer()(nothing)
    fixer = materialize_conservation_fixer(YinYangConservationFixer(), sphere, (:T, :S, :e), CATKEVerticalDiffusivity())
    @test tracer_names(fixer) == (:T, :S)
    rectilinear_grid = RectilinearGrid(arch; size = (1, 1, 1), extent = (1, 1, 1))
    @test_throws "requires a YinYangGrid" materialize_conservation_fixer(YinYangConservationFixer(), rectilinear_grid, (:c,), nothing)

    # `add_conservation_fixer!` reads only the model and its grid, which these stand-ins provide without compiling a model
    @test isnothing(add_conservation_fixer!((; model = (; grid = rectilinear_grid))))
    @test_throws "requires a HydrostaticFreeSurfaceModel" add_conservation_fixer!((; model = (; grid = sphere)))
end
