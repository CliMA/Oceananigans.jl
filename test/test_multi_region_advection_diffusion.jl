using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: reconstruct_global_field
using Oceananigans.Grids: min_Δx, min_Δy
using Oceananigans.Operators: hack_cosd

# Tracer patch for visualization
Gaussian(x, y, L) = exp(-(x^2 + y^2) / 2L^2)

prescribed_velocities() = PrescribedVelocityFields(u=(λ, ϕ, z, t=0) -> 0.1 * hack_cosd(ϕ))

function Δ_min(grid) 
    Δx_min = min_Δx(grid)
    Δy_min = min_Δy(grid)
    return min(Δx_min, Δy_min)
end

function solid_body_tracer_advection_test(grid; P = XPartition, regions = 1)

    if architecture(grid) isa GPU
        devices = (0, 0)
    else
        devices = nothing
    end
    mrg = MultiRegionGrid(grid, partition = P(regions), devices = devices)

    if grid isa RectilinearGrid
        L = 0.1
    else
        L = 24 
    end

    model = HydrostaticFreeSurfaceModel(grid = mrg,
                                        tracers = (:c, :d, :e),
                                        velocities = prescribed_velocities(),
                                        free_surface = ExplicitFreeSurface(),
                                        momentum_advection = nothing,
                                        tracer_advection = WENO5(),
                                        coriolis = nothing,
                                        buoyancy = nothing,
                                        closure  = nothing)

    # Tracer patch for visualization
    Gaussian(x, y, L) = exp(-(x^2 + y^2) / 2L^2)

    # Tracer patch parameters
    cᵢ(x, y, z) = Gaussian(x, 0, L)
    dᵢ(x, y, z) = Gaussian(0, y, L)
    eᵢ(x, y, z) = Gaussian(x, y, L)

    set!(model, c=cᵢ, d=dᵢ, e=eᵢ)

    # Time-scale for tracer advection across the smallest grid cell
    advection_time_scale = Δ_min(grid) / 0.1
    
    Δt = 0.1advection_time_scale
    
    for step in 1:10
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
    mrg = MultiRegionGrid(grid, partition = P(regions), devices = devices)

    free_surface = ExplicitFreeSurface(gravitational_acceleration = 1)
    coriolis     = HydrostaticSphericalCoriolis(rotation_rate = 1)

    model = HydrostaticFreeSurfaceModel(grid = mrg,
                                        momentum_advection = VectorInvariant(),
                                        free_surface = free_surface,
                                        coriolis = coriolis,
                                        tracers = :c,
                                        tracer_advection = WENO5(),
                                        buoyancy = nothing,
                                        closure = nothing)

    g = model.free_surface.gravitational_acceleration
    R = grid.radius
    Ω = model.coriolis.rotation_rate

    uᵢ(λ, φ, z) = 0.1 * cosd(φ) * sin(λ)
    ηᵢ(λ, φ)    = (R * Ω * 0.1 + 0.1^2 / 2) * sind(φ)^2 / g * sin(λ)

    cᵢ(λ, φ, z) = Gaussian(λ, φ - 5, 10)

    set!(model, u=uᵢ, η=ηᵢ, c=cᵢ)

    Δt = 0.1 * Δ_min(grid)  / sqrt(g * grid.Lz) 

    for step in 1:10
        time_step!(model, Δt)
    end

    return merge(model.velocities, model.tracers, (; η = model.free_surface.η))
end

function diffusion_cosine_test(grid;  P = XPartition, regions = 1, closure, field_name = :c) 
    κ, m = 1, 2 # diffusivity and cosine wavenumber

    if architecture(grid) isa GPU
        devices = (0, 0)
    else
        devices = nothing
    end
    mrg = MultiRegionGrid(grid, partition = P(regions), devices = devices)

    model = HydrostaticFreeSurfaceModel(grid = mrg,
                                        coriolis = nothing,
                                        closure = closure,
                                        tracers = :c,
                                        buoyancy=nothing)

    init(x, y, z) = cos(m * x)
    f = fields(model)[field_name]

    @apply_regionally set!(f, init)
    
    update_state!(model)

    # Step forward with small time-step relative to diff. time-scale
    Δt = 1e-6 * grid.Lz^2 / κ
    for n = 1:10
        time_step!(model, Δt)
    end

    return f
end

Nx = 32; Ny = 32

for arch in archs

    grid_rect = RectilinearGrid(arch, size = (Nx, Ny, 1),
                                        halo = (3, 3, 3),
                                        topology = (Periodic, Bounded, Bounded),
                                        x = (0, 1),
                                        y = (0, 1),
                                        z = (0, 1))

    grid_lat = LatitudeLongitudeGrid(arch, size = (Nx, Ny, 1),
                                        halo = (3, 3, 3),
                                        radius = 1, latitude = (-80, 80),
                                        longitude = (-160, 160), z = (-1, 0))

    partitioning = [XPartition]
    
    @testset "Testing multi region tracer advection" begin
        for grid in [grid_rect, grid_lat]
        
            cs, ds, es = solid_body_tracer_advection_test(grid)
            
            cs = Array(interior(cs));
            ds = Array(interior(ds));
            es = Array(interior(es));

            for regions in [2, 4], P in partitioning
                @info "  Testing $regions $(P)s on $(typeof(grid).name.wrapper) on the $arch"
                c, d, e = solid_body_tracer_advection_test(grid; P = P, regions=regions)

                c = interior(reconstruct_global_field(c))
                d = interior(reconstruct_global_field(d))
                e = interior(reconstruct_global_field(e))

                @test all(c .≈ cs)
                @test all(d .≈ ds)
                @test all(e .≈ es)
            end
        end
    end

    @testset "Testing multi region solid body rotation" begin
        grid = grid_lat
        us, vs, ws, cs, ηs = solid_body_rotation_test(grid)
            
        us = Array(interior(us));
        vs = Array(interior(vs));
        ws = Array(interior(ws));
        cs = Array(interior(cs));
        ηs = Array(interior(ηs));
        
        for regions in [2, 4], P in partitioning
            @info "  Testing $regions $(P)s on $(typeof(grid).name.wrapper) on the $arch"
            u, v, w, c, η = solid_body_rotation_test(grid; P = P, regions=regions)

            u = interior(reconstruct_global_field(u))
            v = interior(reconstruct_global_field(v))
            w = interior(reconstruct_global_field(w))
            c = interior(reconstruct_global_field(c))
            η = interior(reconstruct_global_field(η))
                
            @test all(isapprox(u, us, atol=1e-20, rtol = 1e-15))
            @test all(isapprox(v, vs, atol=1e-20, rtol = 1e-15))
            @test all(isapprox(w, ws, atol=1e-20, rtol = 1e-15))
            @test all(isapprox(c, cs, atol=1e-20, rtol = 1e-15))
            @test all(isapprox(η, ηs, atol=1e-20, rtol = 1e-15))
        end
    end

    @testset "Testing multi region gaussian diffusion" begin
        grid  = RectilinearGrid(arch, size = (Nx, Ny, 1),
                                halo = (3, 3, 3),
                                topology = (Bounded, Bounded, Bounded),
                                x = (0, 1),
                                y = (0, 1),
                                z = (0, 1))
        
        diff₂ = ScalarDiffusivity(ν = 1, κ = 1)
        diff₄ = ScalarBiharmonicDiffusivity(ν = 1e-5, κ = 1e-5)

        for fieldname in [:u, :v, :c]
            for closure in [diff₂, diff₄]

                fs = diffusion_cosine_test(grid; closure, field_name = fieldname)
                fs = Array(interior(fs));

                for regions in [2, 4], P in partitioning
                    @info "  Testing diffusion of $fieldname on $regions $(P)s with $(typeof(closure).name.wrapper) on the $arch"

                    f = diffusion_cosine_test(grid; closure, P = P, field_name = fieldname, regions = regions)
                    f = interior(reconstruct_global_field(f))

                    @test all(f .≈ fs)
                end
            end
        end
    end
end
