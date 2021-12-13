using Oceananigans: CPU, GPU
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant, PrescribedVelocityFields
using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedField, ExplicitFreeSurface
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface, ImplicitFreeSurface
using Oceananigans.Coriolis: VectorInvariantEnergyConserving, VectorInvariantEnstrophyConserving
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity
using Oceananigans.Grids: Periodic, Bounded

function time_step_hydrostatic_model_works(grid;
                                           coriolis = nothing,
                                           free_surface = ExplicitFreeSurface(),
                                           momentum_advection = nothing,
                                           closure = nothing,
                                           velocities = nothing)

    tracers = [:T, :S]
    closure isa CATKEVerticalDiffusivity && push!(tracers, :e)

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        momentum_advection = momentum_advection,
                                        free_surface = free_surface,
                                        coriolis = coriolis,
                                        tracers = tracers,
                                        velocities = velocities,
                                        closure = closure)

    simulation = Simulation(model, Δt=1.0, stop_iteration=1)

    run!(simulation)

    return model.clock.iteration == 1
end

function hydrostatic_free_surface_model_tracers_and_forcings_work(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(2π, 2π, 2π))
    model = HydrostaticFreeSurfaceModel(grid=grid, tracers=(:T, :S, :c, :d))

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
            grid = RectilinearGrid(arch, FT, topology=topo_1d, size=(1), extent=(1))
            model = HydrostaticFreeSurfaceModel(grid=grid)        
            @test model isa HydrostaticFreeSurfaceModel
        end
    end
    
    for topo in topos_2d
        @testset "$topo model construction" begin
            @info "  Testing $topo model construction..."
            for arch in archs, FT in float_types
                grid = RectilinearGrid(arch, FT, topology=topo, size=(1, 1), extent=(1, 2))
                model = HydrostaticFreeSurfaceModel(grid=grid)
                @test model isa HydrostaticFreeSurfaceModel
            end
        end
    end
    
    for topo in topos_3d
        @testset "$topo model construction" begin
            @info "  Testing $topo model construction..."
            for arch in archs, FT in float_types
                grid = RectilinearGrid(arch, FT, topology=topo, size=(1, 1, 1), extent=(1, 2, 3))
                model = HydrostaticFreeSurfaceModel(grid=grid)
                @test model isa HydrostaticFreeSurfaceModel
            end
        end
    end

    @testset "Halo size check in model constructor" begin
        for topo in topos_3d
            grid = RectilinearGrid(arch, FT, topology=topo, size=(1, 1, 1), extent=(1, 2, 3))
            hcabd_closure = HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity()

            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid=grid, tracer_advection=CenteredFourthOrder())
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid=grid, tracer_advection=UpwindBiasedThirdOrder())
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid=grid, tracer_advection=UpwindBiasedFifthOrder())
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid=grid, momentum_advection=UpwindBiasedFifthOrder())
            @test_throws ArgumentError HydrostaticFreeSurfaceModel(grid=grid, closure=hcabd_closure)

            # Big enough
            bigger_grid = RectilinearGrid(arch, FT, topology=topo, size=(1, 1, 1), extent=(1, 2, 3), halo=(3, 3, 3))

            model = HydrostaticFreeSurfaceModel(grid=grid, closure=hcabd_closure)
            @test model isa HydrostaticFreeSurfaceModel

            model = HydrostaticFreeSurfaceModel(grid=grid, momentum_advection=UpwindBiasedFifthOrder())
            @test model isa HydrostaticFreeSurfaceModel

            model = HydrostaticFreeSurfaceModel(grid=grid, closure=hcabd_closure)
            @test model isa HydrostaticFreeSurfaceModel

            model = HydrostaticFreeSurfaceModel(grid=grid, tracer_advection=UpwindBiasedFifthOrder())
            @test model isa HydrostaticFreeSurfaceModel
        end
    end

    @testset "Setting HydrostaticFreeSurfaceModel fields" begin
        @info "  Testing setting hydrostatic free surface model fields..."
        for arch in archs, FT in float_types
            N = (4, 4, 1)
            L = (2π, 3π, 5π)

            grid = RectilinearGrid(arch, FT, size=N, extent=L)
            model = HydrostaticFreeSurfaceModel(grid=grid)

            x, y, z = nodes((Face, Center, Center), model.grid, reshape=true)

            u₀(x, y, z) = x * y^2
            u_answer = @. x * y^2

            η₀ = rand(size(grid)...)
            η_answer = deepcopy(η₀)

            set!(model, u=u₀, η=η₀)

            u, v, w = model.velocities
            η = model.free_surface.η

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

        rectilinear_grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1), halo=(3, 3, 3))

        lat_lon_sector_grid = LatitudeLongitudeGrid(arch, size=(1, 1, 1), longitude=(0, 60), latitude=(15, 75), z=(-1, 0), precompute_metrics=true)
        lat_lon_strip_grid  = LatitudeLongitudeGrid(arch, size=(1, 1, 1), longitude=(-180, 180), latitude=(15, 75), z=(-1, 0), precompute_metrics=true)
        
        vertically_stretched_grid = RectilinearGrid(arch, size=(1, 1, 1), x=(0, 1), y=(0, 1), z=z_face_generator(), halo=(3, 3, 3))
        lat_lon_sector_grid_stretched = LatitudeLongitudeGrid(arch, size=(1, 1, 1), longitude=(0, 60), latitude=(15, 75), z=z_face_generator(), precompute_metrics=true)
        lat_lon_strip_grid_stretched  = LatitudeLongitudeGrid(arch, size=(1, 1, 1), longitude=(-180, 180), latitude=(15, 75), z=z_face_generator(), precompute_metrics=true)

        grids = (rectilinear_grid, lat_lon_sector_grid, lat_lon_strip_grid, lat_lon_sector_grid_stretched, lat_lon_strip_grid_stretched, vertically_stretched_grid)
        free_surfaces = (ExplicitFreeSurface(), ImplicitFreeSurface())

        for grid in grids
            for free_surface in free_surfaces
                topo = topology(grid)
                grid_type = typeof(grid).name.wrapper
                free_surface_type = typeof(free_surface).name.wrapper
                test_label = "[$arch, $grid_type, $topo, $free_surface_type]"                
                @testset "Time-stepping HydrostaticFreeSurfaceModels with different grids $test_label" begin
                    @info "  Testing time-stepping HydrostaticFreeSurfaceModels with different grids $test_label..."
                    @test time_step_hydrostatic_model_works(grid, free_surface=free_surface)
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
                         HydrostaticSphericalCoriolis(scheme=VectorInvariantEnergyConserving()),
                         HydrostaticSphericalCoriolis(scheme=VectorInvariantEnstrophyConserving()))

            @testset "Time-stepping HydrostaticFreeSurfaceModels [$arch, $(typeof(coriolis))]" begin
                @test time_step_hydrostatic_model_works(lat_lon_sector_grid, coriolis=coriolis)
                @test time_step_hydrostatic_model_works(lat_lon_strip_grid, coriolis=coriolis)
            end
        end

        for momentum_advection in (VectorInvariant(), CenteredSecondOrder(), WENO5())
            @testset "Time-stepping HydrostaticFreeSurfaceModels [$arch, $(typeof(momentum_advection))]" begin
                @info "  Testing time-stepping HydrostaticFreeSurfaceModels [$arch, $(typeof(momentum_advection))]..."
                @test time_step_hydrostatic_model_works(rectilinear_grid, momentum_advection=momentum_advection)
            end
        end

        momentum_advection = VectorInvariant()
        @testset "Time-stepping HydrostaticFreeSurfaceModels [$arch, $(typeof(momentum_advection))]" begin
            @info "  Testing time-stepping HydrostaticFreeSurfaceModels [$arch, $(typeof(momentum_advection))]..."
            @test time_step_hydrostatic_model_works(lat_lon_sector_grid, momentum_advection=momentum_advection)
        end

        for closure in (IsotropicDiffusivity(),
                        HorizontallyCurvilinearAnisotropicDiffusivity(),
                        HorizontallyCurvilinearAnisotropicDiffusivity(time_discretization=VerticallyImplicitTimeDiscretization()),
                        CATKEVerticalDiffusivity(),
                        CATKEVerticalDiffusivity(time_discretization=ExplicitTimeDiscretization()))

            @testset "Time-stepping Curvilinear HydrostaticFreeSurfaceModels [$arch, $(typeof(closure).name.wrapper)]" begin
                @info "  Testing time-stepping Curvilinear HydrostaticFreeSurfaceModels [$arch, $(typeof(closure).name.wrapper)]..."
                @test_skip time_step_hydrostatic_model_works(arch, vertically_stretched_grid, closure=closure)
                @test time_step_hydrostatic_model_works(lat_lon_sector_grid, closure=closure)
                @test time_step_hydrostatic_model_works(lat_lon_strip_grid, closure=closure)
            end
        end

        closure = IsotropicDiffusivity()
        @testset "Time-stepping Rectilinear HydrostaticFreeSurfaceModels [$arch, $(typeof(closure).name.wrapper)]" begin
            @info "  Testing time-stepping Rectilinear HydrostaticFreeSurfaceModels [$arch, $(typeof(closure).name.wrapper)]..."
            @test time_step_hydrostatic_model_works(rectilinear_grid, closure=closure)
        end

        @testset "PrescribedVelocityFields [$arch]" begin
            @info "  Testing PrescribedVelocityFields [$arch]..."
            
            grid = RectilinearGrid(arch, size=1, x = (0, 1), topology = (Periodic, Flat, Flat))
            
            u₀, v₀ = 0.1, 0.2
            
            U = Field(Face, Center, Center, arch, grid)
            V = Field(Center, Face, Center, arch, grid)

            CUDA.@allowscalar begin
                parent(U)[2, 1, 1] = u₀
                parent(V)[2, 1, 1] = v₀
            end

            u = PrescribedField(Face, Center, Center, U, grid)
            v = PrescribedField(Face, Center, Center, V, grid)

            CUDA.@allowscalar begin
                @test parent(u)[1, 1, 1] == u₀
                @test parent(u)[3, 1, 1] == u₀
                @test parent(v)[1, 1, 1] == v₀
                @test parent(v)[3, 1, 1] == v₀
            end
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

        @testset "HydrostaticFreeSurfaceModel with tracers and forcings [$arch]" begin
            @info "  Testing HydrostaticFreeSurfaceModel with tracers and forcings [$arch]..."
            hydrostatic_free_surface_model_tracers_and_forcings_work(arch)
        end
    end
end
