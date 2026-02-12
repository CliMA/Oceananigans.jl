include("dependencies_for_runtests.jl")

using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant, PrescribedVelocityFields
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface, ImplicitFreeSurface, PrescribedFreeSurface
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SingleColumnGrid
using Oceananigans.Advection: EnergyConserving, EnstrophyConserving, FluxFormAdvection, CrossAndSelfUpwinding
using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.Grids: MutableVerticalDiscretization

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

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)

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

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)
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

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)

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

@testset "Hydrostatic free surface Models" begin
    @info "Testing hydrostatic free surface models..."

    @testset "$topo_1d model construction" begin
        @info "  Testing $topo_1d model construction..."
        for arch in archs, FT in [Float64] #float_types
            grid = RectilinearGrid(arch, FT, topology=topo_1d, size=1, extent=1)
            model = HydrostaticFreeSurfaceModel(grid)
            @test model isa HydrostaticFreeSurfaceModel

            # SingleColumnGrid tests
            @test grid isa SingleColumnGrid
            @test isnothing(model.free_surface)
        end
    end

    for topo in topos_2d
        @testset "$topo model construction" begin
            @info "  Testing $topo model construction..."
            for arch in archs, FT in float_types
                grid = RectilinearGrid(arch, FT, topology=topo, size=(1, 1), extent=(1, 2))
                model = HydrostaticFreeSurfaceModel(grid)
                @test model isa HydrostaticFreeSurfaceModel
            end
        end
    end

    for topo in topos_3d
        @testset "$topo model construction" begin
            @info "  Testing $topo model construction..."
            for arch in archs, FT in float_types
                grid = RectilinearGrid(arch, FT, topology=topo, size=(1, 1, 1), extent=(1, 2, 3))
                model = HydrostaticFreeSurfaceModel(grid)
                @test model isa HydrostaticFreeSurfaceModel
            end
        end
    end

    for FreeSurface in (ExplicitFreeSurface, ImplicitFreeSurface, SplitExplicitFreeSurface, Nothing)
        @testset "$FreeSurface model construction" begin
            @info "  Testing $FreeSurface model construction..."
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

            model = HydrostaticFreeSurfaceModel(bigger_grid; momentum_advection=UpwindBiased(order=5))
            @test model isa HydrostaticFreeSurfaceModel

            model = HydrostaticFreeSurfaceModel(bigger_grid; closure=hcabd_closure)
            @test model isa HydrostaticFreeSurfaceModel

            model = HydrostaticFreeSurfaceModel(bigger_grid; tracer_advection=UpwindBiased(order=5))
            @test model isa HydrostaticFreeSurfaceModel
        end
    end

    @testset "Setting HydrostaticFreeSurfaceModel fields" begin
        @info "  Testing setting hydrostatic free surface model fields..."
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
                @info "  Testing time-stepping Rectilinear HydrostaticFreeSurfaceModels [$arch, $topo]..."
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
                grid_type = typeof(grid).name.wrapper
                free_surface_type = typeof(free_surface).name.wrapper
                test_label = "[$arch, $grid_type, $topo, $free_surface_type]"
                @testset "Time-stepping HydrostaticFreeSurfaceModels with various grids $test_label" begin
                    @info "  Testing time-stepping HydrostaticFreeSurfaceModels with various grids $test_label..."
                    @test time_step_hydrostatic_model_works(grid; free_surface)
                end
            end
        end

        @info " Time-stepping HydrostaticFreeSurfaceModels with y-Flat grid"
        lat_lon_flat_grid = LatitudeLongitudeGrid(arch; size=(H, H), longitude=(-180, 180), z=(-1, 0), precompute_metrics,
                                                  halo=(7, 7), topology=(Periodic, Flat, Bounded))
        @test_broken time_step_hydrostatic_model_works(lat_lon_flat_grid)
        c = CenterField(lat_lon_flat_grid) # just test we can build a field
        @test c.boundary_conditions.north isa Nothing
        @test c.boundary_conditions.south isa Nothing

        for topo in [topos_3d..., topos_2d...]
            size = Flat in topo ? (10, 10) : (10, 10, 10)
            halo = Flat in topo ? (7,  7)  : (7, 7, 7)
            x    = topo[1] == Flat ? nothing : (0, 1)
            y    = topo[2] == Flat ? nothing : (0, 1)

            grid = RectilinearGrid(arch; size, halo, x, y, z=(-1, 0), topology=topo)

            for advection in [WENOVectorInvariant(), VectorInvariant(), WENO()]
                @testset "Time-stepping HydrostaticFreeSurfaceModels with $advection [$arch, $topo]" begin
                    @info "  Testing time-stepping HydrostaticFreeSurfaceModels with $advection [$arch, $topo]..."
                    @test time_step_hydrostatic_model_works(grid; momentum_advection=advection)
                end
            end
        end

        for coriolis in (nothing, FPlane(f=1), BetaPlane(f₀=1, β=0.1))
            @testset "Time-stepping HydrostaticFreeSurfaceModels [$arch, $(typeof(coriolis))]" begin
                @info "  Testing time-stepping HydrostaticFreeSurfaceModels [$arch, $(typeof(coriolis))]..."
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
            WENOVectorInvariant(),
            WENOVectorInvariant(; upwinding = CrossAndSelfUpwinding(cross_scheme = WENO())),
            WENOVectorInvariant(; multi_dimensional_stencil = true),
        )

        for momentum_advection in momentum_advections
            @testset "Time-stepping HydrostaticFreeSurfaceModels [$arch, $(summary(momentum_advection))]" begin
                for grid in (rectilinear_grid, lat_lon_sector_grid)
                    @info "  Testing time-stepping HydrostaticFreeSurfaceModels [$arch, $(nameof(typeof(grid))), $(summary(momentum_advection))]..."
                    @test time_step_hydrostatic_model_works(grid; momentum_advection)
                end
            end
        end

        for tracer_advection in [WENO(),
                                 FluxFormAdvection(WENO(), WENO(), Centered()),
                                 (b=WENO(), c=nothing)]

            T = typeof(tracer_advection)
            @testset "Time-stepping HydrostaticFreeSurfaceModels with tracer advection [$arch, $T]" begin
                @info "  Testing time-stepping HydrostaticFreeSurfaceModels with tracer advection [$arch, $T]..."
                @test time_step_hydrostatic_model_works(rectilinear_grid; tracer_advection, tracers=[:b, :c])
            end
        end

        for closure in (ScalarDiffusivity(),
                        HorizontalScalarDiffusivity(),
                        VerticalScalarDiffusivity(),
                        VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization()),
                        CATKEVerticalDiffusivity(),
                        CATKEVerticalDiffusivity(ExplicitTimeDiscretization()))

            @testset "Time-stepping Curvilinear HydrostaticFreeSurfaceModels [$arch, $(typeof(closure).name.wrapper)]" begin
                @info "  Testing time-stepping Curvilinear HydrostaticFreeSurfaceModels [$arch, $(typeof(closure).name.wrapper)]..."
                @test_skip time_step_hydrostatic_model_works(arch, vertically_stretched_grid, closure=closure)
                @test time_step_hydrostatic_model_works(lat_lon_sector_grid; closure)
                @test time_step_hydrostatic_model_works(lat_lon_strip_grid; closure)
            end
        end

        closure = ScalarDiffusivity()
        @testset "Time-stepping Rectilinear HydrostaticFreeSurfaceModels [$arch, $(typeof(closure).name.wrapper)]" begin
            @info "  Testing time-stepping Rectilinear HydrostaticFreeSurfaceModels [$arch, $(typeof(closure).name.wrapper)]..."
            @test time_step_hydrostatic_model_works(rectilinear_grid, closure=closure)
        end

        @testset "Time-stepping HydrostaticFreeSurfaceModels with PrescribedVelocityFields [$arch]" begin
            @info "  Testing time-stepping HydrostaticFreeSurfaceModels with PrescribedVelocityFields [$arch]..."

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
            @info "  Testing PrescribedVelocityFields with FieldTimeSeries [$arch]..."

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

            @info "    PrescribedVelocityFields with FieldTimeSeries test passed"
        end

        @testset "PrescribedVelocityFields with FieldTimeSeries output [$arch]" begin
            @info "  Testing PrescribedVelocityFields with FieldTimeSeries output [$arch]..."

            grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
            times = 0:0.1:1.0

            # Create velocity FieldTimeSeries with u = t
            u_fts = FieldTimeSeries{Face, Center, Center}(grid, times)
            set!(u_fts, (x, y, z, t) -> t)

            velocities = PrescribedVelocityFields(; u=u_fts)
            model = HydrostaticFreeSurfaceModel(grid; velocities, tracers=:c)

            simulation = Simulation(model; Δt=0.05, stop_time=0.5)

            # Output the prescribed velocity (which is a TimeSeriesInterpolation)
            test_filename = "test_prescribed_velocity_output.jld2"
            simulation.output_writers[:fields] = JLD2Writer(model, (; u=model.velocities.u);
                                                           schedule=TimeInterval(0.1),
                                                           filename=test_filename,
                                                           overwrite_existing=true)

            run!(simulation)

            # Read output and verify values
            u_output = FieldTimeSeries(test_filename, "u")

            for n in eachindex(u_output.times)
                t = u_output.times[n]
                u_val = u_output[n][1, 1, 1]
                @test u_val ≈ t atol=1e-5
            end

            # Clean up
            rm(test_filename)

            @info "    PrescribedVelocityFields with FieldTimeSeries output test passed"
        end

        @testset "PrescribedFreeSurface with PrescribedVelocityFields and ZStar [$arch]" begin
            @info "  Testing PrescribedFreeSurface with PrescribedVelocityFields and ZStar [$arch]..."

            # Small grid with MutableVerticalDiscretization (triggers ZStarCoordinate)
            H = 10
            z_faces = MutableVerticalDiscretization((-H, 0))

            grid = RectilinearGrid(arch;
                                   size = (4, 4, 4),
                                   x = (0, 100),
                                   y = (0, 100),
                                   z = z_faces,
                                   halo = (3, 3, 3),
                                   topology = (Periodic, Periodic, Bounded))

            # Prescribed velocity fields (zero divergence)
            u_prescribed(x, y, z, t) = 0.0
            v_prescribed(x, y, z, t) = 0.0

            velocities = PrescribedVelocityFields(; u=u_prescribed, v=v_prescribed)

            # Prescribed free surface: η oscillates over one period
            A = 0.1  # amplitude (small relative to H=10)
            η_prescribed(x, y, z, t) = A * sin(2π * t)

            free_surface = PrescribedFreeSurface(displacement=η_prescribed)

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

            # Time step for one full period with dt = 0.01
            dt = 0.01
            T_period = 1.0  # one full period of sin(2πt)
            simulation = Simulation(model; Δt=dt, stop_time=T_period)

            # σᶜᶜⁿ is an OffsetArray with halo; extract interior values using OffsetArray indexing
            Nx, Ny, _ = size(grid)
            σ_interior(σ) = [σ[i, j, 1] for i in 1:Nx, j in 1:Ny]

            # Check initial σ ≈ 1.0 (η(0) = 0)
            σ_initial = σ_interior(grid.z.σᶜᶜⁿ)
            @test all(σ_initial .≈ 1.0)

            # Run for one full period
            run!(simulation)

            # σ should reflect η at the current clock time (step_free_surface! advances
            # the PFS clock to tⁿ⁺¹ before the grid update, matching prognostic behavior).
            σ_final = σ_interior(grid.z.σᶜᶜⁿ)
            expected_σ = (H + A * sin(2π * model.clock.time)) / H

            tol = 1e-10
            @test all(abs.(σ_final .- expected_σ) .< tol)

            # Also verify that σ oscillates correctly at a non-trivial time
            simulation.stop_time = 1.25
            run!(simulation)

            σ_quarter = σ_interior(grid.z.σᶜᶜⁿ)
            expected_σ_quarter = (H + A * sin(2π * model.clock.time)) / H

            @test all(abs.(σ_quarter .- expected_σ_quarter) .< tol)

            @info "    PrescribedFreeSurface with PrescribedVelocityFields and ZStar test passed"
        end

        @testset "DiagnosticVerticalVelocity [$arch]" begin
            @info "  Testing DiagnosticVerticalVelocity [$arch]..."

            Nx, Ny, Nz = 4, 4, 4
            Lx, Ly, H = 100.0, 100.0, 10.0

            u_prescribed(x, y, z, t) = sin(2π * x / Lx)
            v_prescribed(x, y, z, t) = 0.0

            # --- Static grid ---
            static_grid = RectilinearGrid(arch;
                                          size = (Nx, Ny, Nz),
                                          x = (0, Lx), y = (0, Ly), z = (-H, 0),
                                          topology = (Periodic, Periodic, Bounded))

            static_velocities = PrescribedVelocityFields(; u = u_prescribed,
                                                           v = v_prescribed,
                                                           w = DiagnosticVerticalVelocity())

            static_model = HydrostaticFreeSurfaceModel(static_grid;
                                                       velocities = static_velocities,
                                                       tracers = :c,
                                                       buoyancy = nothing)

            @test static_model isa HydrostaticFreeSurfaceModel
            @test static_model.velocities.w isa DiagnosticVerticalVelocity

            time_step!(static_model, 1.0)

            w_static = static_model.velocities.w.field
            Δx = Lx / Nx
            Δz = H / Nz

            # w should be non-zero for divergent u
            @test !all(iszero, interior(w_static))

            # Bottom boundary condition: w = 0 at k = 1
            for i in 1:Nx
                @test w_static[i, 1, 1] == 0
            end

            # Check discrete continuity: w[k+1] = w[k] - Δz * discrete_div_u
            for i in 1:Nx, k in 1:Nz
                u_right = u_prescribed(i * Δx, 0, 0, 0)
                u_left  = u_prescribed((i - 1) * Δx, 0, 0, 0)
                discrete_div = (u_right - u_left) / Δx

                Δw = w_static[i, 1, k + 1] - w_static[i, 1, k]
                @test Δw ≈ -Δz * discrete_div atol = 1e-10
            end

            # --- ZStar grid with PrescribedFreeSurface ---
            z_faces = MutableVerticalDiscretization((-H, 0))

            zstar_grid = RectilinearGrid(arch;
                                         size = (Nx, Ny, Nz),
                                         x = (0, Lx), y = (0, Ly), z = z_faces,
                                         halo = (3, 3, 3),
                                         topology = (Periodic, Periodic, Bounded))

            A = 0.1
            η_prescribed(x, y, z, t) = A * sin(2π * t)

            zstar_velocities = PrescribedVelocityFields(; u = u_prescribed,
                                                          v = v_prescribed,
                                                          w = DiagnosticVerticalVelocity())

            free_surface = PrescribedFreeSurface(displacement = η_prescribed)

            zstar_model = HydrostaticFreeSurfaceModel(zstar_grid;
                                                      velocities = zstar_velocities,
                                                      free_surface,
                                                      tracers = :c,
                                                      buoyancy = nothing)

            @test zstar_model isa HydrostaticFreeSurfaceModel
            @test zstar_model.velocities.w isa DiagnosticVerticalVelocity
            @test zstar_model.free_surface isa PrescribedFreeSurface

            time_step!(zstar_model, 1.0)

            w_zstar = zstar_model.velocities.w.field
            @test !all(iszero, interior(w_zstar))

            # σ should reflect η at the current time
            σ_val = zstar_grid.z.σᶜᶜⁿ[1, 1, 1]
            expected_σ = (H + A * sin(2π * zstar_model.clock.time)) / H
            @test σ_val ≈ expected_σ atol = 1e-10

            # The moving-grid ∂t_σ contribution should make w_zstar differ from w_static
            @test interior(w_zstar) != interior(w_static)

            @info "    DiagnosticVerticalVelocity test passed"
        end

        @testset "HydrostaticFreeSurfaceModel with tracers and forcings [$arch]" begin
            @info "  Testing HydrostaticFreeSurfaceModel with tracers and forcings [$arch]..."
            hydrostatic_free_surface_model_tracers_and_forcings_work(arch)
        end

        # See: https://github.com/CliMA/Oceananigans.jl/issues/3870
        @testset "HydrostaticFreeSurfaceModel with Float32 CATKE [$arch]" begin
            @info "  Testing HydrostaticFreeSurfaceModel with Float32 CATKE [$arch]..."
            @test time_step_hydrostatic_model_with_catke_works(arch, Float32)
        end
    end
end
