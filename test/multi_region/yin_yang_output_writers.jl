include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "yin_yang_test_utils.jl"))

using Oceananigans: fields
using Oceananigans.MultiRegion: YinYangField, remapped_field, yin_yang_masks

@testset "Output of Yin-Yang fields [$(summary(arch))]" for arch in archs
    dir = mktempdir()
    grid = YinYangGrid(arch; size=(48, 16, 2), z=(-1, 0), radius=1)
    u, v = XFaceField(grid), YFaceField(grid)
    set_panel_components!(u, v, solid_body_velocity)
    fill_halo_regions!((u, v))
    model = HydrostaticFreeSurfaceModel(grid; tracers = :c, buoyancy = nothing, closure = nothing,
                                        velocities = PrescribedVelocityFields(; u, v))
    set!(model, c = (λ, φ, z) -> spherical_harmonics(λ, φ) * (1 + z))
    c = model.tracers.c

    # The tracer rotates by 1/4 radian over the averaging window, which ends exactly on the last time step
    Δt, Nt = 1/16, 4
    simulation = Simulation(model; Δt, stop_iteration = Nt, verbose = false)
    simulation.output_writers[:panels] = JLD2Writer(model, (; c, u, v); dir, filename = "panels", schedule = IterationInterval(Nt), with_halos = true)
    c̄ = Field(Average(remapped_field(c), dims=1))
    simulation.output_writers[:averages] = JLD2Writer(model, (; c, c̄); dir, filename = "averages", schedule = AveragedTimeInterval(Nt * Δt))

    # The right Riemann sum of the interior of each panel
    expected = [zero(Array(interior(getregion(c, r)))) for r in 1:2]
    function accumulate_expected(sim)
        sim.model.clock.iteration == 0 && return
        for r in 1:2
            expected[r] .+= Array(interior(getregion(c, r))) ./ Nt
        end
    end
    simulation.callbacks[:expected] = Callback(accumulate_expected, IterationInterval(1))
    run!(simulation)

    cpu_grid = on_architecture(CPU(), grid)
    jldopen(joinpath(dir, "panels.jld2")) do file
        loaded = file["serialized/grid"]
        @test sprint(show, loaded) == sprint(show, cpu_grid)
        @test all(loaded[r].Azᶜᶜᵃ == cpu_grid[r].Azᶜᶜᵃ for r in 1:2)
        @test yin_yang_masks(loaded).cell_weights == yin_yang_masks(cpu_grid).cell_weights
        @test loaded.connectivity.interpolation.face_center.weights.x == cpu_grid.connectivity.interpolation.face_center.weights.x
        @test file["timeseries/u/serialized/location"] == (Face, Center, Center)
        for (name, field) in pairs((; c, u, v)), r in 1:2 # each panel's data, halos included, is saved
            @test file["timeseries/$name/$Nt"].regional_objects[r] ≈ Array{Float32}(on_architecture(CPU(), parent(getregion(field, r))))
        end
    end

    # Every panel is read back as a panel of a field on the Yin-Yang grid
    time_series = FieldTimeSeries(joinpath(dir, "panels.jld2"), "c")
    @test time_series.grid isa YinYangGrid
    @test length(time_series) == 2
    @test time_series[end] isa YinYangField
    @test all(isapprox(interior(getregion(time_series[end], r)), Array(interior(getregion(c, r))); rtol=1e-6) for r in 1:2)

    jldopen(joinpath(dir, "averages.jld2")) do file
        @test keys(file["timeseries/t"]) == ["0", string(Nt)]
        Nx, Ny, Nz = size(grid)
        Hx, Hy, Hz = halo_size(grid)
        for r in 1:2
            average = file["timeseries/c/$Nt"].regional_objects[r][Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz]
            @test average ≈ expected[r] rtol=1e-6
            @test maximum(abs, average .- Array(interior(getregion(c, r)))) > 1e-2
        end
        @test all(isfinite, file["timeseries/c̄/$Nt"])
    end
    rm(dir; recursive = true)
end

@testset "Checkpoints of z-star coordinates on Yin-Yang grids [$(summary(arch))]" for arch in archs
    dir = mktempdir()
    grid = YinYangGrid(arch; size=(48, 16, 2), z=MutableVerticalDiscretization((-1000, 0)), radius=1e6)
    free_surface = SplitExplicitFreeSurface(substeps = 4, extend_halos = false)
    zstar_model() = HydrostaticFreeSurfaceModel(grid; free_surface, tracers = :c, buoyancy = nothing)
    model = zstar_model()
    set!(model, u = (λ, φ, z) -> 20cosd(φ), η = (λ, φ, z) -> cosd(λ) * cosd(φ), c = (λ, φ, z) -> 1 + sind(φ))

    # The conservation fixer keeps no state, so the restart is bitwise with it too
    simulation = Simulation(model; Δt = 10, stop_iteration = 2, verbose = false)
    add_conservation_fixer!(simulation)
    simulation.output_writers[:checkpointer] = Checkpointer(model; dir, prefix = "zstar", schedule = IterationInterval(2))
    run!(simulation)
    simulation.stop_iteration = 4
    run!(simulation)

    new_model = zstar_model()
    new_simulation = Simulation(new_model; Δt = 10, stop_iteration = 4, verbose = false)
    add_conservation_fixer!(new_simulation)
    @test_logs (:info, r"^Picking up simulation from checkpoint file") run!(new_simulation, pickup = joinpath(dir, "zstar_iteration2.jld2"))
    for name in (:c, :η), r in 1:2
        @test interior(getregion(fields(new_model)[name], r)) == interior(getregion(fields(model)[name], r))
    end
    @test_throws r"vertical coordinate of every region" Oceananigans.restore_prognostic_state!(model.vertical_coordinate, grid, NamedTuple())
    rm(dir; recursive = true)
end
