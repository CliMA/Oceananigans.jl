include("dependencies_for_runtests.jl")

Gaussian(x, y, L) = exp(-(x^2 + y^2) / 2L^2)
prescribed_velocities() = PrescribedVelocityFields(u=(λ, ϕ, z, t = 0) -> 0.1 * hack_cosd(ϕ))

function Δ_min(grid)
    Δx_min = minimum_xspacing(grid, Center(), Center(), Center())
    Δy_min = minimum_yspacing(grid, Center(), Center(), Center())
    return min(Δx_min, Δy_min)
end

function solid_body_tracer_advection_test(grid; P = XPartition, regions = 1)

    if architecture(grid) isa GPU
        devices = (0, 0)
    else
        devices = nothing
    end

    if grid isa RectilinearGrid
        L = 0.1
    else
        L = 24 # degrees
    end

    # Tracer patch parameters
    cᵢ(x, y, z) = Gaussian(x, 0, L)
    eᵢ(x, y, z) = Gaussian(x, y, L)

    mrg = MultiRegionGrid(grid, partition = P(regions), devices = devices)

    model = HydrostaticFreeSurfaceModel(grid = mrg,
                                        tracers = (:c, :e),
                                        velocities = prescribed_velocities(),
                                        free_surface = ExplicitFreeSurface(),
                                        momentum_advection = nothing,
                                        tracer_advection = WENO(),
                                        coriolis = nothing,
                                        buoyancy = nothing,
                                        closure  = nothing)

    set!(model, c=cᵢ, e=eᵢ)

    # Time-scale for tracer advection across the smallest grid cell; 0.1 is maximum velocity
    advection_time_scale = Δ_min(grid) / 0.1

    Δt = 0.1advection_time_scale

    for _ in 1:10
        time_step!(model, Δt)
    end

    return model.tracers
end

function solid_body_rotation_test(grid; P = XPartition, regions = 1)

    if architecture(grid) isa GPU
        devices = (0, 0)
    else
        devices = nothing
    end

    mrg = MultiRegionGrid(grid, partition = P(regions))

    free_surface = ExplicitFreeSurface(gravitational_acceleration = 1)
    coriolis     = HydrostaticSphericalCoriolis(rotation_rate = 1)

    model = HydrostaticFreeSurfaceModel(grid = mrg,
                                        momentum_advection = VectorInvariant(),
                                        free_surface = free_surface,
                                        coriolis = coriolis,
                                        tracers = :c,
                                        tracer_advection = WENO(),
                                        buoyancy = nothing,
                                        closure = nothing)

    g = model.free_surface.gravitational_acceleration
    R = grid.radius
    Ω = model.coriolis.rotation_rate

    uᵢ(λ, φ, z) = 0.1 * cosd(φ) * sind(λ)
    ηᵢ(λ, φ, z) = (R * Ω * 0.1 + 0.1^2 / 2) * sind(φ)^2 / g * sind(λ)
    cᵢ(λ, φ, z) = Gaussian(λ, φ - 5, 10)

    set!(model, u=uᵢ, η=ηᵢ, c=cᵢ)

    Δt = 0.1 * Δ_min(grid) / sqrt(g * grid.Lz)

    for _ in 1:10
        time_step!(model, Δt)
    end

    return merge(model.velocities, model.tracers, (; η = model.free_surface.η))
end

function diffusion_cosine_test(grid; P = XPartition, regions = 1, closure, field_name = :c)
    if architecture(grid) isa GPU
        devices = (0, 0)
    else
        devices = nothing
    end

    mrg = MultiRegionGrid(grid, partition = P(regions), devices = devices)

    # For MultiRegionGrids with regions > 1, the SplitExplicitFreeSurface extends the
    # halo region in the horizontal. Because the extented halo region size cannot exceed
    # the grid's interior size we pick here the number of substeps taking the grid's
    # size into consideration.
    free_surface = SplitExplicitFreeSurface(substeps = 8)

    model = HydrostaticFreeSurfaceModel(; grid = mrg,
                                        free_surface,
                                        closure,
                                        tracers = :c,
                                        coriolis = nothing,
                                        buoyancy=nothing)

    initial_condition(x, y, z) = cos(2x)

    expr = quote
        # we use set!(model, ...) so that initialize!(model) is called
        # initialize!(model) is required for SplitExplicitFreeSurface
        # so that the barotropic transport is initialized
        set!($model, $field_name = $initial_condition)
    end
    eval(expr)

    # Step forward with small time-step relative to diffusive time-scale
    Δt = 1e-6 * cell_diffusion_timescale(model)

    for _ in 1:10
        time_step!(model, Δt)
    end

    return fields(model)[field_name]
end

Nx = Ny = 32

partitioning = [XPartition]

for arch in archs
    grid_rect = RectilinearGrid(arch,
                                size = (Nx, Ny, 1),
                                halo = (3, 3, 3),
                                topology = (Periodic, Bounded, Bounded),
                                x = (0, 1),
                                y = (0, 1),
                                z = (0, 1))

    grid_lat = LatitudeLongitudeGrid(arch,
                                     size = (Nx, Ny, 1),
                                     halo = (3, 3, 3),
                                     latitude = (-80, 80),
                                     longitude = (-180, 180),
                                     z = (-1, 0),
                                     radius = 1)

    @testset "Testing multi region tracer advection" begin
        for grid in (grid_rect, grid_lat)

            # on a single grid
            cs, es = solid_body_tracer_advection_test(grid, regions=1)

            cs = Array(interior(cs))
            es = Array(interior(es))

            for regions in (2,), P in partitioning
                @info "  Testing $regions $(P)s on $(typeof(grid).name.wrapper) on the $arch"
                c, e = solid_body_tracer_advection_test(grid; P=P, regions=regions)

                c = interior(reconstruct_global_field(c))
                e = interior(reconstruct_global_field(e))

                @test all(isapprox(c, cs, atol=1e-20, rtol=1e-15))
                @test all(isapprox(e, es, atol=1e-20, rtol=1e-15))
            end
        end
    end

    @testset "Testing multi region solid body rotation" begin
        grid = LatitudeLongitudeGrid(arch,
                                     size = (Nx, Ny, 1),
                                     halo = (3, 3, 3),
                                     latitude = (-80, 80),
                                     longitude = (-160, 160),
                                     z = (-1, 0),
                                     radius = 1,
                                     topology=(Bounded, Bounded, Bounded))

        # on a single grid
        us, vs, ws, cs, ηs = solid_body_rotation_test(grid, regions=1)

        us = Array(interior(us))
        vs = Array(interior(vs))
        ws = Array(interior(ws))
        cs = Array(interior(cs))
        ηs = Array(interior(ηs))

        for regions in (2,), P in partitioning
            @info "  Testing $regions $(P)s on $(typeof(grid).name.wrapper) on the $arch"
            u, v, w, c, η = solid_body_rotation_test(grid; P=P, regions=regions)

            u = interior(reconstruct_global_field(u))
            v = interior(reconstruct_global_field(v))
            w = interior(reconstruct_global_field(w))
            c = interior(reconstruct_global_field(c))
            η = interior(reconstruct_global_field(η))

            @test all(isapprox(u, us, atol=1e-20, rtol=1e-15))
            @test all(isapprox(v, vs, atol=1e-20, rtol=1e-15))
            @test all(isapprox(w, ws, atol=1e-20, rtol=1e-15))
            @test all(isapprox(c, cs, atol=1e-20, rtol=1e-15))
            @test all(isapprox(η, ηs, atol=1e-20, rtol=1e-15))
        end
    end

    @testset "Testing multi region gaussian diffusion" begin
        grid  = RectilinearGrid(arch,
                                size = (Nx, Ny, 1),
                                halo = (3, 3, 3),
                                topology = (Bounded, Bounded, Bounded),
                                x = (0, 1),
                                y = (0, 1),
                                z = (0, 1))

        diff₂ = ScalarDiffusivity(ν = 1, κ = 1)
        diff₄ = ScalarBiharmonicDiffusivity(ν = 1e-5, κ = 1e-5)

        for field_name in (:u, :v, :c)
            for closure in (diff₂, diff₄)

                # on a single grid
                fs = diffusion_cosine_test(grid; closure, field_name, regions = 1)
                fs = Array(interior(fs))

                for regions in (2,), P in partitioning
                    @info "  Testing diffusion of $field_name on $regions $(P)s with $(typeof(closure).name.wrapper) on $arch"

                    f = diffusion_cosine_test(grid; closure, P, field_name, regions)
                    f = interior(reconstruct_global_field(f))

                    @test all(isapprox(f, fs, atol=1e-20, rtol=1e-15))
                end
            end
        end
    end
end
