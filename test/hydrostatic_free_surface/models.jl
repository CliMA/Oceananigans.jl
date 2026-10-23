include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant, PrescribedVelocityFields
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface, ImplicitFreeSurface, PrescribedFreeSurface
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SingleColumnGrid
using Oceananigans.Models: reset!
using Oceananigans.Advection: EnergyConserving, EnstrophyConserving, FluxFormAdvection, CrossAndSelfUpwinding, EnergyConservingUpwinding
using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.OutputReaders: Cyclical, InMemory, OnDisk

function time_step_hydrostatic_model_works(grid;
                                           coriolis = nothing,
                                           free_surface = ExplicitFreeSurface(),
                                           momentum_advection = nothing,
                                           tracers = [:b],
                                           tracer_advection = nothing,
                                           closure = nothing,
                                           velocities = nothing)

    buoyancy = BuoyancyTracer()

    model = HydrostaticFreeSurfaceModel(grid; coriolis, tracers, velocities, buoyancy,
                                        momentum_advection, tracer_advection, free_surface, closure)

    simulation = Simulation(model; Δt=1.0, stop_iteration=1, verbose=false)

    run!(simulation)

    return model.clock.iteration == 1
end

function hydrostatic_free_surface_model_tracers_and_forcings_work(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(2π, 2π, 2π))
    model = HydrostaticFreeSurfaceModel(grid; tracers=(:T, :S, :c, :d))

    @test model.tracers.T isa Field
    @test model.tracers.S isa Field
    @test model.tracers.c isa Field
    @test model.tracers.d isa Field

    @test haskey(model.forcing, :u)
    @test haskey(model.forcing, :v)
    @test haskey(model.forcing, :η)
    @test haskey(model.forcing, :T)
    @test haskey(model.forcing, :S)
    @test haskey(model.forcing, :c)
    @test haskey(model.forcing, :d)

    simulation = Simulation(model; Δt=1.0, stop_iteration=1, verbose=false)
    run!(simulation)

    @test model.clock.iteration == 1

    return nothing
end

function time_step_hydrostatic_model_with_catke_works(arch, FT)
    grid = LatitudeLongitudeGrid(arch, FT,
                                 topology = (Bounded, Bounded, Bounded),
                                 size = (8, 8, 8),
                                 longitude = (0, 1),
                                 latitude = (0, 1),
                                 z = (-100, 0))

    model = HydrostaticFreeSurfaceModel(grid;
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b,),
                                        closure = CATKEVerticalDiffusivity(eltype(grid)))

    simulation = Simulation(model; Δt=1.0, stop_iteration=1, verbose=false)

    run!(simulation)

    return model.clock.iteration == 1
end

topo_1d = (Flat, Flat, Bounded)

topos_2d = ((Periodic, Flat, Bounded),
            (Flat, Bounded,  Bounded),
            (Bounded, Flat,  Bounded))

topos_3d = ((Periodic, Periodic, Bounded),
            (Periodic, Bounded,  Bounded),
            (Bounded,  Bounded,  Bounded))

# Logged when an advection scheme is reduced to first order in the z-direction of grids with a single cell
reduced_z_advection_log = (:info, "Using the advection scheme UpwindBiased(order=1) in the z-direction because size(grid, 3) = 1")

@testset "Hydrostatic free surface Models" begin
    @testset "$topo_1d model construction" begin
        for arch in archs, FT in [Float64] #float_types
            grid = RectilinearGrid(arch, FT, topology=topo_1d, size=1, extent=1)
            model = HydrostaticFreeSurfaceModel(grid)
            @test model isa HydrostaticFreeSurfaceModel

            # SingleColumnGrid tests
            @test grid isa SingleColumnGrid
            @test isnothing(model.free_surface)
        end
    end

    @testset "reset! for $topo_1d models" begin
        for arch in archs, closure in (nothing, CATKEVerticalDiffusivity(), TKEDissipationVerticalDiffusivity())
            grid = RectilinearGrid(arch, topology=topo_1d, size=4, extent=1)
            model = HydrostaticFreeSurfaceModel(grid; closure, tracers=:b, buoyancy=BuoyancyTracer())
            set!(model, u=1, b=1)
            time_step!(model, 1)
            reset!(model)
            @test maximum(abs, model.velocities.u) == 0
            @test maximum(abs, model.tracers.b) == 0
            @test maximum(abs, model.timestepper.G⁻.b) == 0

            if !isnothing(closure)
                u⁻, v⁻ = model.closure_fields.previous_velocities
                @test maximum(abs, u⁻) == 0
                @test maximum(abs, model.closure_fields.κu) == 0
            end
        end
    end

    for topo in topos_2d
        @testset "$topo model construction" begin
            for arch in archs, FT in float_types
                grid = RectilinearGrid(arch, FT, topology=topo, size=(1, 1), extent=(1, 2))
                model = HydrostaticFreeSurfaceModel(grid)
                @test model isa HydrostaticFreeSurfaceModel
            end
        end
    end

    for topo in topos_3d
        @testset "$topo model construction" begin
            for arch in archs, FT in float_types
                grid = RectilinearGrid(arch, FT, topology=topo, size=(1, 1, 1), extent=(1, 2, 3))
                model = HydrostaticFreeSurfaceModel(grid)
                @test model isa HydrostaticFreeSurfaceModel
            end
        end
    end

    for FreeSurface in (ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface, Nothing)
        @testset "$FreeSurface model construction" begin
            for arch in archs, FT in float_types
                grid = RectilinearGrid(arch, FT, size=(1, 1, 1), extent=(1, 2, 3))
                model = HydrostaticFreeSurfaceModel(grid; free_surface=FreeSurface())
                @test model isa HydrostaticFreeSurfaceModel
            end
        end
    end

    @testset "Halo size check in model constructor" begin
        for topo in topos_3d
            grid = RectilinearGrid(topology=topo, size=(1, 1, 1), extent=(1, 2, 3), halo=(1, 1, 1))
            hcabd_closure = ScalarBiharmonicDiffusivity()

            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; tracer_advection=Centered(order=4))
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; tracer_advection=UpwindBiased(order=3))
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; tracer_advection=UpwindBiased(order=5))
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; momentum_advection=UpwindBiased(order=5))
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid; closure=hcabd_closure)

            # Big enough
            bigger_grid = RectilinearGrid(topology=topo, size=(3, 3, 1), extent=(1, 2, 3), halo=(3, 3, 3))

            model = HydrostaticFreeSurfaceModel(bigger_grid; closure=hcabd_closure)
            @test model isa HydrostaticFreeSurfaceModel

            model = @test_logs reduced_z_advection_log HydrostaticFreeSurfaceModel(bigger_grid; momentum_advection=UpwindBiased(order=5))
            @test model isa HydrostaticFreeSurfaceModel

            model = HydrostaticFreeSurfaceModel(bigger_grid; closure=hcabd_closure)
            @test model isa HydrostaticFreeSurfaceModel

            model = HydrostaticFreeSurfaceModel(bigger_grid; tracer_advection=UpwindBiased(order=5))
            @test model isa HydrostaticFreeSurfaceModel
        end
    end

    @testset "Setting HydrostaticFreeSurfaceModel fields" begin
        for arch in archs, FT in float_types
            N = (4, 4, 1)
            L = (2π, 3π, 5π)

            grid = RectilinearGrid(arch, FT, size=N, extent=L)
            model = HydrostaticFreeSurfaceModel(grid)

            x, y, z = nodes(model.grid, (Face(), Center(), Center()), reshape=true)

            u₀(x, y, z) = x * y^2
            u_answer = @. x * y^2

            η₀ = rand(size(grid)...)
            η_answer = deepcopy(η₀)

            set!(model, u=u₀, η=η₀)

            u, v, w = model.velocities
            η = model.free_surface.displacement

            @test all(Array(interior(u)) .≈ u_answer)
            @test all(Array(interior(η)) .≈ η_answer)
        end
    end

    for arch in archs

        for topo in topos_3d
            grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1), topology=topo)

            @testset "Time-stepping Rectilinear HydrostaticFreeSurfaceModels [$arch, $topo]" begin
                @test time_step_hydrostatic_model_works(grid)
            end
        end

        z_face_generator(; Nz=1, p=1, H=1) = k -> -H + (k / (Nz+1))^p # returns a generating function

        H = 7
        halo = (7, 7, 7)
        rectilinear_grid = RectilinearGrid(arch; size=(H, H, 1), extent=(1, 1, 1), halo)
        vertically_stretched_grid = RectilinearGrid(arch; size=(H, H, 1), x=(0, 1), y=(0, 1), z=z_face_generator(), halo=(H, H, H))

        precompute_metrics = true
        lat_lon_sector_grid = LatitudeLongitudeGrid(arch; size=(H, H, H), longitude=(0, 60), latitude=(15, 75), z=(-1, 0), precompute_metrics, halo)
        lat_lon_strip_grid  = LatitudeLongitudeGrid(arch; size=(H, H, H), longitude=(-180, 180), latitude=(15, 75), z=(-1, 0), precompute_metrics, halo)

        z = z_face_generator()
        lat_lon_sector_grid_stretched = LatitudeLongitudeGrid(arch; size=(H, H, H), longitude=(0, 60), latitude=(15, 75), z, precompute_metrics, halo)
        lat_lon_strip_grid_stretched  = LatitudeLongitudeGrid(arch; size=(H, H, H), longitude=(-180, 180), latitude=(15, 75), z, precompute_metrics, halo)

        grids = (rectilinear_grid, vertically_stretched_grid,
                 lat_lon_sector_grid, lat_lon_strip_grid,
                 lat_lon_sector_grid_stretched, lat_lon_strip_grid_stretched)

        free_surfaces = (ExplicitFreeSurface(), ImplicitFreeSurface())

        for grid in grids
            for free_surface in free_surfaces
                topo = topology(grid)
                grid_type = nameof(typeof(grid))
                free_surface_type = nameof(typeof(free_surface))
                test_label = "[$arch, $grid_type, $topo, $free_surface_type]"
                @testset "Time-stepping HydrostaticFreeSurfaceModels with various grids $test_label" begin
                    @test time_step_hydrostatic_model_works(grid; free_surface)
                end
            end
        end

        @testset "Time-stepping HydrostaticFreeSurfaceModels with y-Flat grid [$arch]" begin
            lat_lon_flat_grid = LatitudeLongitudeGrid(arch; size=(H, H), longitude=(-180, 180), z=(-1, 0), precompute_metrics,
                                                      halo=(7, 7), topology=(Periodic, Flat, Bounded))
            @test_broken time_step_hydrostatic_model_works(lat_lon_flat_grid)
            c = CenterField(lat_lon_flat_grid) # just test we can build a field
            @test c.boundary_conditions.north isa Nothing
            @test c.boundary_conditions.south isa Nothing
        end

        for topo in [topos_3d..., topos_2d...]
            size = Flat in topo ? (10, 10) : (10, 10, 10)
            halo = Flat in topo ? (7,  7)  : (7, 7, 7)
            x    = topo[1] == Flat ? nothing : (0, 1)
            y    = topo[2] == Flat ? nothing : (0, 1)

            grid = RectilinearGrid(arch; size, halo, x, y, z=(-1, 0), topology=topo)

            for advection in [WENOVectorInvariant(), VectorInvariant(), WENO()]
                @testset "Time-stepping HydrostaticFreeSurfaceModels with $advection [$arch, $topo]" begin
                    @test time_step_hydrostatic_model_works(grid; momentum_advection=advection)
                end
            end
        end

        for coriolis in (nothing, FPlane(f=1), BetaPlane(f₀=1, β=0.1))
            @testset "Time-stepping HydrostaticFreeSurfaceModels [$arch, $(typeof(coriolis))]" begin
                @test time_step_hydrostatic_model_works(rectilinear_grid, coriolis=coriolis)
            end
        end

        for coriolis in (nothing,
                         HydrostaticSphericalCoriolis(scheme=EnergyConserving()),
                         HydrostaticSphericalCoriolis(scheme=EnstrophyConserving()))

            @testset "Time-stepping HydrostaticFreeSurfaceModels [$arch, $(typeof(coriolis))]" begin
                @test time_step_hydrostatic_model_works(lat_lon_sector_grid; coriolis)
                @test time_step_hydrostatic_model_works(lat_lon_strip_grid; coriolis)
            end
        end

        momentum_advections = (
            Centered(),
            WENO(),
            VectorInvariant(),
            VectorInvariant(vorticity_scheme = EnergyConservingUpwinding(WENO())),
            WENOVectorInvariant(),
            WENOVectorInvariant(; upwinding = CrossAndSelfUpwinding(cross_scheme = WENO())),
            WENOVectorInvariant(; multi_dimensional_stencil = true),
        )

        for momentum_advection in momentum_advections
            @testset "Time-stepping HydrostaticFreeSurfaceModels [$arch, $(nameof(typeof(grid))), $(summary(momentum_advection))]" for grid in (rectilinear_grid, lat_lon_sector_grid)
                logs = momentum_advection isa WENO && size(grid, 3) == 1 ? (reduced_z_advection_log,) : ()
                @test @test_logs logs... time_step_hydrostatic_model_works(grid; momentum_advection)
            end
        end

        # One log for each tracer advected with WENO
        for (tracer_advection, logs) in ((WENO(), (reduced_z_advection_log, reduced_z_advection_log)),
                                         (FluxFormAdvection(WENO(), WENO(), Centered()), ()),
                                         ((b=WENO(), c=nothing), (reduced_z_advection_log,)))

            T = typeof(tracer_advection)
            @testset "Time-stepping HydrostaticFreeSurfaceModels with tracer advection [$arch, $T]" begin
                @test @test_logs logs... time_step_hydrostatic_model_works(rectilinear_grid; tracer_advection, tracers=[:b, :c])
            end
        end

        for closure in (ScalarDiffusivity(),
                        HorizontalScalarDiffusivity(),
                        VerticalScalarDiffusivity(),
                        VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization()),
                        CATKEVerticalDiffusivity(),
                        CATKEVerticalDiffusivity(ExplicitTimeDiscretization()))

            @testset "Time-stepping Curvilinear HydrostaticFreeSurfaceModels [$arch, $(nameof(typeof(closure)))]" begin
                @test_skip time_step_hydrostatic_model_works(arch, vertically_stretched_grid, closure=closure)
                @test time_step_hydrostatic_model_works(lat_lon_sector_grid; closure)
                @test time_step_hydrostatic_model_works(lat_lon_strip_grid; closure)
            end
        end

        closure = ScalarDiffusivity()
        @testset "Time-stepping Rectilinear HydrostaticFreeSurfaceModels [$arch, $(nameof(typeof(closure)))]" begin
            @test time_step_hydrostatic_model_works(rectilinear_grid, closure=closure)
        end

        @testset "Time-stepping HydrostaticFreeSurfaceModels with PrescribedVelocityFields [$arch]" begin
            # Non-parameterized functions
            u(x, y, z, t) = 1
            v(x, y, z, t) = exp(z)
            w(x, y, z, t) = sin(z)
            velocities = PrescribedVelocityFields(u=u, v=v, w=w)

            @test time_step_hydrostatic_model_works(rectilinear_grid, momentum_advection  = nothing, velocities = velocities)
            @test time_step_hydrostatic_model_works(lat_lon_sector_grid, momentum_advection = nothing, velocities = velocities)

            parameters = (U=1, m=0.1, W=0.001)
            u(x, y, z, t, p) = p.U
            v(x, y, z, t, p) = exp(p.m * z)
            w(x, y, z, t, p) = p.W * sin(z)

            velocities = PrescribedVelocityFields(u=u, v=v, w=w, parameters=parameters)

            @test time_step_hydrostatic_model_works(rectilinear_grid, momentum_advection  = nothing, velocities = velocities)
            @test time_step_hydrostatic_model_works(lat_lon_sector_grid, momentum_advection = nothing, velocities = velocities)
        end

        @testset "PrescribedVelocityFields with FieldTimeSeries [$arch]" begin
            grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
            times = 0:0.1:1.0

            # Create velocity FieldTimeSeries and populate with set!
            u_fts = FieldTimeSeries{Face, Center, Center}(grid, times)
            for (n, t) in enumerate(times)
                set!(u_fts, t, n)  # u = t at each time index
            end

            # Use with PrescribedVelocityFields
            velocities = PrescribedVelocityFields(; u=u_fts)
            model = HydrostaticFreeSurfaceModel(grid; velocities, tracers=:c)

            # At t=0, velocity field should interpolate to u=0
            u = model.velocities.u
            @test u[1, 1, 1] ≈ 0.0

            # Time step to t=0.05
            time_step!(model, 0.05)

            # Now u should interpolate to 0.05 (between t=0 and t=0.1)
            @test u[1, 1, 1] ≈ 0.05

            # Time step to t=0.15 (total t=0.2)
            time_step!(model, 0.15)

            # Now u should interpolate to 0.2
            @test u[1, 1, 1] ≈ 0.2
        end

        @testset "PrescribedVelocityFields with FieldTimeSeries output [$arch]" begin
            grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
            times = 0:0.1:1.0

            # Create velocity FieldTimeSeries with u = t
            u_fts = FieldTimeSeries{Face, Center, Center}(grid, times)
            set!(u_fts, (x, y, z, t) -> t)

            velocities = PrescribedVelocityFields(; u=u_fts)
            model = HydrostaticFreeSurfaceModel(grid; velocities, tracers=:c)

            simulation = Simulation(model; Δt=0.05, stop_time=0.5, verbose=false)

            # Output the prescribed velocity (which is a TimeSeriesInterpolation)
            dir = mktempdir()
            simulation.output_writers[:fields] = JLD2Writer(model, (; u=model.velocities.u);
                                                           schedule=TimeInterval(0.1),
                                                           dir,
                                                           filename="test_prescribed_velocity_output.jld2",
                                                           overwrite_files=true)

            run!(simulation)

            # Read output and verify values
            u_output = FieldTimeSeries(simulation.output_writers[:fields].filepath, "u")

            for n in eachindex(u_output.times)
                t = u_output.times[n]
                u_val = u_output[n][1, 1, 1]
                @test u_val ≈ t atol=1e-5
            end

            rm(dir; recursive=true)
        end

        @testset "PrescribedFreeSurface with PrescribedVelocityFields and ZStar [$arch]" begin
            @info "  Testing PrescribedFreeSurface with PrescribedVelocityFields and ZStar [$arch]..."

            # Small grid with MutableVerticalDiscretization (triggers ZStarCoordinate)
            H = 10
            zstar = MutableVerticalDiscretization((-H, 0))

            grid = RectilinearGrid(arch;
                                   size = (4, 4, 4),
                                   x = (0, 100),
                                   y = (0, 100),
                                   z = zstar,
                                   halo = (3, 3, 3),
                                   topology = (Periodic, Periodic, Bounded))

            # Prescribed velocity fields (zero divergence)
            u(x, y, z, t) = 0.0
            v(x, y, z, t) = 0.0

            velocities = PrescribedVelocityFields(; u=u, v=v)

            # Prescribed free surface: η oscillates over one period
            A = 0.1  # amplitude (small relative to H=10)
            displacement(x, y, z, t) = A * sin(2π * t)

            free_surface = PrescribedFreeSurface(displacement=displacement)

            # Build model
            model = HydrostaticFreeSurfaceModel(grid;
                                                 velocities,
                                                 free_surface,
                                                 tracers = nothing,
                                                 buoyancy = nothing)

            @test model isa HydrostaticFreeSurfaceModel
            @test model.free_surface isa PrescribedFreeSurface

            # Check that η appears in fields(model)
            model_fields = fields(model)
            @test haskey(model_fields, :η)

            # Time step for one full period with Δt = 0.01
            Δt = 0.01
            stop_time = 1.0  # one full period of sin(2πt)
            simulation = Simulation(model; Δt=Δt, stop_time=stop_time)

            # σᶜᶜⁿ is an OffsetArray with halo; extract interior values using OffsetArray indexing
            Nx, Ny, _ = size(grid, (Center, Center, Center))

            # Check initial σ == 1.0 (η(0) = 0)
            atol = 1e-10
            σ = view(grid.z.σᶜᶜⁿ, 1:Nx, 1:Ny, 1)
            target = ones(size(σ))
            @test σ ≈ target atol=atol

            # Run for one full period
            run!(simulation)

            # σ should reflect η at the current clock time (step_free_surface! advances
            # the PFS clock to tⁿ⁺¹ before the grid update, matching prognostic behavior).
            target .= (H + A * sin(2π * model.clock.time)) / H
            @test σ ≈ target atol=atol

            # Also verify that σ oscillates correctly at a non-trivial time
            simulation.stop_time = 1.25
            run!(simulation)

            target .= (H + A * sin(2π * model.clock.time)) / H
            @test σ ≈ target atol=atol

            @info "    PrescribedFreeSurface with PrescribedVelocityFields and ZStar test passed"
        end

        @testset "PrescribedFreeSurface with constant Field displacement and ZStar [$arch]" begin
            @info "  Testing PrescribedFreeSurface with constant Field displacement and ZStar [$arch]..."

            H = 10
            zstar = MutableVerticalDiscretization((-H, 0))
            grid = RectilinearGrid(arch;
                                   size = (4, 4, 4),
                                   x = (0, 100), y = (0, 100),
                                   z = zstar,
                                   halo = (3, 3, 3),
                                   topology = (Periodic, Periodic, Bounded))

            # Constant (time-independent) plain Fields for u, v, and η
            u = XFaceField(grid)
            v = YFaceField(grid)
            η = Field{Center, Center, Nothing}(grid; indices = (:, :, 1))
            set!(η, 0.1)  # constant 0.1 m displacement

            velocities   = PrescribedVelocityFields(; u=u, v=v, formulation=DiagnosticVerticalVelocity())
            free_surface = PrescribedFreeSurface(displacement=η)

            # Model construction must not throw (regression test for Field-has-no-clock bug)
            model = HydrostaticFreeSurfaceModel(grid;
                                                 velocities,
                                                 free_surface,
                                                 tracers = :T,
                                                 buoyancy = nothing,
                                                 closure = nothing)

            @test model isa HydrostaticFreeSurfaceModel
            @test model.free_surface isa PrescribedFreeSurface

            # Time-stepping must not throw, and σ must stay constant (∂t_η = 0 ⟹ ∂t_σ = 0)
            simulation = Simulation(model; Δt=0.1, stop_time=1.0)
            run!(simulation)

            Nx, Ny, _ = size(grid, (Center, Center, Center))
            σ = view(grid.z.σᶜᶜⁿ, 1:Nx, 1:Ny, 1)
            expected_σ = (H + 0.1) / H
            @test all(≈(expected_σ, atol=1e-10), σ)

            @info "    PrescribedFreeSurface with constant Field displacement and ZStar test passed"
        end

        @testset "PrescribedFreeSurface with PartlyInMemory FieldTimeSeries and ZStar [$arch]" begin
            @info "  Testing PrescribedFreeSurface with PartlyInMemory FieldTimeSeries and ZStar [$arch]..."

            Nz = 4
            H = 100
            z_faces = collect(range(-H, 0, length = Nz + 1))
            grid = RectilinearGrid(arch;
                                   size = (4, 4, Nz),
                                   x = (0, 100), y = (0, 100),
                                   z = MutableVerticalDiscretization(z_faces),
                                   halo = (4, 4, 4),
                                   topology = (Periodic, Periodic, Bounded))

            # Write η FieldTimeSeries to a temporary JLD2 file with 12 monthly snapshots
            year = years = 365days
            month = year / 12
            fts_times = collect(((1:12) .- 0.5) * month)

            eta_file = "eta.jld2"

            η_ts_ondisk = FieldTimeSeries{Center, Center, Face}(grid, fts_times;
                indices = (:, :, Nz + 1),
                backend = OnDisk(), path = eta_file, name = "eta",
                time_indexing = Cyclical(year))

            for n in 1:12
                η = Field{Center, Center, Face}(grid; indices = (:, :, Nz + 1))
                set!(η, sin(2π * n / 12))
                set!(η_ts_ondisk, η, n)
            end

            # Reload with InMemory(4) — only 4 of 12 snapshots fit in memory
            η_ts = FieldTimeSeries(eta_file, "eta";
                architecture = arch, grid,
                backend = InMemory(4),
                time_indexing = Cyclical(year))

            free_surface = PrescribedFreeSurface(displacement = η_ts)
            model = HydrostaticFreeSurfaceModel(grid;
                tracers = :c,
                free_surface,
                velocities = PrescribedVelocityFields(; formulation = DiagnosticVerticalVelocity()))

            set!(model, c = 1.0)

            # Run long enough to cross a buffer boundary and loop around cyclical year
            stop_time = 2years
            simulation = Simulation(model; Δt = 5400seconds, stop_time)
            run!(simulation)

            @test simulation.model.clock.time ≈ stop_time

            # Clean up
            rm(eta_file)

            @info "    PrescribedFreeSurface with PartlyInMemory FieldTimeSeries and ZStar test passed"
        end

        @testset "DiagnosticVerticalVelocity [$arch]" begin
            @info "  Testing DiagnosticVerticalVelocity [$arch]..."

            Nx, Ny, Nz = 4, 4, 4
            Lx, Ly, H = 100.0, 100.0, 5000.0

            u(x, y, z, t) = sin(2π * x / Lx)
            v(x, y, z, t) = 0.0

            # --- Static grid ---
            static_grid = RectilinearGrid(arch;
                                          size = (Nx, Ny, Nz),
                                          x = (0, Lx), y = (0, Ly), z = (-H, 0),
                                          topology = (Periodic, Periodic, Bounded))

            static_velocities = PrescribedVelocityFields(; u = u,
                                                           v = v,
                                                           formulation = DiagnosticVerticalVelocity())

            static_model = HydrostaticFreeSurfaceModel(static_grid;
                                                       velocities = static_velocities,
                                                       tracers = :c,
                                                       buoyancy = nothing)

            @test static_model isa HydrostaticFreeSurfaceModel
            @test static_model.velocities.w isa Field
            @test static_model.velocities.formulation isa DiagnosticVerticalVelocity

            time_step!(static_model, 1.0)

            w = static_model.velocities.w
            Δx = Lx / Nx
            Δz = H / Nz

            # w should be non-zero for divergent u
            @test !all(iszero, interior(w))

            # Bottom boundary condition: w = 0 at k = 1
            for i in 1:Nx
                @test w[i, 1, 1] == 0
            end

            # Check discrete continuity: w[k+1] = w[k] - Δz * discrete_div_u
            for i in 1:Nx, k in 1:Nz
                discrete_div = (u(i * Δx, 0, 0, 0) - u((i - 1) * Δx, 0, 0, 0)) / Δx
                Δw = w[i, 1, k + 1] - w[i, 1, k]
                @test Δw ≈ -Δz * discrete_div atol = 1e-10
            end

            # --- ZStar grid with PrescribedFreeSurface ---
            zstar = MutableVerticalDiscretization((-H, 0))

            zstar_grid = RectilinearGrid(arch;
                                         size = (Nx, Ny, Nz),
                                         x = (0, Lx), y = (0, Ly), z = zstar,
                                         halo = (3, 3, 3),
                                         topology = (Periodic, Periodic, Bounded))

            A = 1
            uniform_displacement(t) = A * sin(2π / 10 * t)
            displacement(x, y, z, t) = uniform_displacement(t)

            zstar_velocities = PrescribedVelocityFields(; u = u,
                                                          v = v,
                                                          formulation = DiagnosticVerticalVelocity())

            free_surface = PrescribedFreeSurface(displacement = displacement)

            zstar_model = HydrostaticFreeSurfaceModel(zstar_grid;
                                                      velocities = zstar_velocities,
                                                      free_surface,
                                                      tracers = :c,
                                                      buoyancy = nothing)

            @test zstar_model isa HydrostaticFreeSurfaceModel
            @test zstar_model.velocities.w isa Field
            @test zstar_model.velocities.formulation isa DiagnosticVerticalVelocity
            @test zstar_model.free_surface isa PrescribedFreeSurface

            time_step!(zstar_model, 1.0)

            wstar = zstar_model.velocities.w
            @test !all(iszero, interior(wstar))  # ∂t_σ from displacement ≠ -∇·u in general

            # η should match prescribed
            η = zstar_grid.z.ηⁿ[1, 1, 1]
            @test η ≈ uniform_displacement(zstar_model.clock.time) atol = 1e-10

            # σ should reflect η at the current time
            σ = zstar_grid.z.σᶜᶜⁿ[1, 1, 1]
            @test σ ≈ (H + η) / H atol = 1e-10

            # The moving-grid ∂t_σ contribution should make wstar differ from w
            @test interior(wstar) ≉ interior(w)

            # ∂t_σ must be nonzero after a step with time-varying displacement
            # (regression test: timing bug caused ∂t_σ = 0 always)
            @test !all(iszero, parent(zstar_grid.z.∂t_σ))

            # ∂t_σ = (σᶜᶜⁿ − σᶜᶜ⁻) / Δt = (η(Δt) − η(0)) / (H · Δt)
            # Initial η(0) = 0, so expected = uniform_displacement(Δt) / (H * Δt)
            expected_∂t_σ = uniform_displacement(1.0) / (H * 1.0)
            @test zstar_grid.z.∂t_σ[1, 1, 1] ≈ expected_∂t_σ atol=1e-10

            # --- ZStar grid with constant u and spatially-varying displacement ---
            # When displacement is constant in time, ∂t_σ = 0.
            # But w ≠ 0 because the stretched Δz varies with x, creating
            # a non-zero volume flux divergence even for constant u.
            U₀ = 1.0

            σx_zstar = MutableVerticalDiscretization((-H, 0))
            σx_grid = RectilinearGrid(arch;
                                      size = (Nx, Ny, Nz),
                                      x = (0, Lx), y = (0, Ly), z = σx_zstar,
                                      halo = (3, 3, 3),
                                      topology = (Periodic, Periodic, Bounded))

            σx_velocities = PrescribedVelocityFields(; u = (x, y, z, t) -> U₀,
                                                       formulation = DiagnosticVerticalVelocity())

            σx_free_surface = PrescribedFreeSurface(displacement = (x, y, z, t) -> A * sin(2π * x / Lx))

            σx_model = HydrostaticFreeSurfaceModel(σx_grid;
                                                   velocities = σx_velocities,
                                                   free_surface = σx_free_surface,
                                                   tracers = :c,
                                                   buoyancy = nothing)

            @test σx_model isa HydrostaticFreeSurfaceModel
            @test σx_model.velocities.formulation isa DiagnosticVerticalVelocity

            time_step!(σx_model, 1.0)

            wσ = σx_model.velocities.w

            # ∂t_σ should be zero (displacement is time-independent)
            @test all(iszero, parent(σx_grid.z.∂t_σ))

            # w should be non-zero (spatially varying σ creates flux divergence)
            @test !all(iszero, interior(wσ))

            # Bottom BC: w = 0 at k = 1
            for i in 1:Nx
                @test wσ[i, 1, 1] == 0
            end

            # Analytical comparison
            for i in 1:Nx, k in 1:Nz+1
                # x coordinates of the cell centers
                xᶜ = (i - 0.5) * Δx
                xᶜ⁻ = (i - 1.5) * Δx
                xᶜ⁺ = (i + 0.5) * Δx
                # interpolate the free surface and σ on the cell faces
                ηᶠ = A * (sin(2π * xᶜ⁻ / Lx) + sin(2π * xᶜ / Lx)) / 2
                ηᶠ⁺ = A * (sin(2π * xᶜ / Lx) + sin(2π * xᶜ⁺ / Lx)) / 2
                σᶠ = 1 + ηᶠ / H
                σᶠ⁺ = 1 + ηᶠ⁺ / H
                # All levels contribute the same Δw with w[1] = 0.
                Δr = H / Nz
                Δw = -U₀ * Δr * (σᶠ⁺ - σᶠ) / Δx
                wₐ = (k - 1) * Δw
                @test wσ[i, 1, k] ≈ wₐ atol = 1e-10
            end

            # --- ZStar grid with FieldTimeSeries-prescribed u, v, and displacement ---
            fts_times = 0:0.5:10.0

            fts_zstar = MutableVerticalDiscretization((-H, 0))
            fts_grid = RectilinearGrid(arch;
                                       size = (Nx, Ny, Nz),
                                       x = (0, Lx), y = (0, Ly), z = fts_zstar,
                                       halo = (3, 3, 3),
                                       topology = (Periodic, Periodic, Bounded))

            # Build FieldTimeSeries for u, v, and η with the same physics as
            # the function-based ZStar sub-case above
            u_fts = FieldTimeSeries{Face, Center, Center}(fts_grid, fts_times)
            set!(u_fts, (x, y, z, t) -> sin(2π * x / Lx))

            v_fts = FieldTimeSeries{Center, Face, Center}(fts_grid, fts_times)
            set!(v_fts, (x, y, z, t) -> 0)

            η_fts = FieldTimeSeries{Center, Center, Face}(fts_grid, fts_times)
            set!(η_fts, (x, y, z, t) -> uniform_displacement(t))

            fts_velocities = PrescribedVelocityFields(; u = u_fts,
                                                        v = v_fts,
                                                        formulation = DiagnosticVerticalVelocity())

            fts_free_surface = PrescribedFreeSurface(displacement = η_fts)

            fts_model = HydrostaticFreeSurfaceModel(fts_grid;
                                                    velocities = fts_velocities,
                                                    free_surface = fts_free_surface,
                                                    tracers = :c,
                                                    buoyancy = nothing)

            @test fts_model isa HydrostaticFreeSurfaceModel
            @test fts_model.velocities.w isa Field
            @test fts_model.velocities.formulation isa DiagnosticVerticalVelocity
            @test fts_model.free_surface isa PrescribedFreeSurface

            time_step!(fts_model, 1.0)

            wstar_fts = fts_model.velocities.w
            @test !all(iszero, interior(wstar_fts))  # ∂t_σ from displacement ≠ -∇·u in general

            # η should match prescribed
            η = fts_grid.z.ηⁿ[1, 1, 1]
            @test η ≈ uniform_displacement(fts_model.clock.time) atol = 1e-10
            # σ should match the prescribed displacement at the current time
            σ = fts_grid.z.σᶜᶜⁿ[1, 1, 1]
            @test σ ≈ (H + η) / H atol = 1e-10

            # Diagnosed w from FieldTimeSeries inputs should match the
            # function-based ZStar result (identical physics, values O(100)).
            @test interior(wstar_fts) ≈ interior(wstar) atol = 1e-10

            @info "    DiagnosticVerticalVelocity test passed"
        end

        @testset "HydrostaticFreeSurfaceModel with tracers and forcings [$arch]" begin
            hydrostatic_free_surface_model_tracers_and_forcings_work(arch)
        end

        # See: https://github.com/CliMA/Oceananigans.jl/issues/3870
        @testset "HydrostaticFreeSurfaceModel with Float32 CATKE [$arch]" begin
            @test time_step_hydrostatic_model_with_catke_works(arch, Float32)
        end
    end
end
