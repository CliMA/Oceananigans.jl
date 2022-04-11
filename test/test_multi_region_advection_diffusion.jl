using Oceananigans.MultiRegion
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

function solid_body_tracer_advection_test(grid; regions = 1)

    if architecture(grid) isa GPU
        devices = (0, 0)
    else
        devices = nothing
    end
    mrg = MultiRegionGrid(grid, partition = XPartition(regions), devices = devices)

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

function solid_body_rotation_test(grid; regions = 1)

    if architecture(grid) isa GPU
        devices = (0, 0)
    else
        devices = nothing
    end
    mrg = MultiRegionGrid(grid, partition = XPartition(regions), devices = devices)

    free_surface = ExplicitFreeSurface(gravitational_acceleration = 1)
    coriolis     = HydrostaticSphericalCoriolis(rotation_rate = 1)

    model = HydrostaticFreeSurfaceModel(grid = mrg,
                                        momentum_advection = VectorInvariant(),
                                        free_surface = free_surface,
                                        coriolis = coriolis,
                                        tracers = :c,
                                        tracer_advection = WENO5(grid),
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

function diffusion_cosine_test(grid; regions = 1, closure, field_name = :c) 
    κ, m = 1, 2 # diffusivity and cosine wavenumber

    if architecture(grid) isa GPU
        devices = (0, 0)
    else
        devices = nothing
    end
    mrg = MultiRegionGrid(grid, partition = XPartition(regions), devices = devices)

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
                                        longitude = (-180, 180), z = (-1, 0))

    @testset "Testing multi region tracer advection" begin
        for grid in [grid_rect, grid_lat]
        
            cs, ds, es = solid_body_tracer_advection_test(grid)
            
            cs = interior(cs);
            ds = interior(ds);
            es = interior(es);

            for regions in [2, 4]
                @info "  Testing $regions partitions on $(typeof(grid).name.wrapper) on the $arch"
                c, d, e = solid_body_tracer_advection_test(grid; regions=regions)

                c = construct_regionally(interior, c)
                d = construct_regionally(interior, d)
                e = construct_regionally(interior, e)
                
                for region in 1:regions
                    init = Int(size(cs, 1) / regions) * (region - 1) + 1
                    fin  = Int(size(cs, 1) / regions) * region
                    @test all(Array(c[region]) .≈ Array(cs)[init:fin, :, :])
                    @test all(Array(d[region]) .≈ Array(ds)[init:fin, :, :])
                    @test all(Array(e[region]) .≈ Array(es)[init:fin, :, :])
                end
            end
        end
    end

    @testset "Testing multi region solid body rotation" begin
        grid = grid_lat
        us, vs, ws, cs, ηs = solid_body_rotation_test(grid)
            
        us = interior(us);
        vs = interior(vs);
        ws = interior(ws);
        cs = interior(cs);
        ηs = interior(ηs);
        
        for regions in [2, 4]
            @info "  Testing $regions partitions on $(typeof(grid).name.wrapper) on the $arch"
            u, v, w, c, η = solid_body_rotation_test(grid; regions=regions)

            u = construct_regionally(interior, u)
            v = construct_regionally(interior, v)
            w = construct_regionally(interior, w)
            c = construct_regionally(interior, c)
            η = construct_regionally(interior, η)
                
            for region in 1:regions
                init = Int(size(cs, 1) / regions) * (region - 1) + 1
                fin  = Int(size(cs, 1) / regions) * region
                @test all(Array(u[region]) .≈ Array(us)[init:fin, :, :])
                @test all(Array(v[region]) .≈ Array(vs)[init:fin, :, :])
                @test all(Array(w[region]) .≈ Array(ws)[init:fin, :, :])
                @test all(Array(c[region]) .≈ Array(cs)[init:fin, :, :])
                @test all(Array(η[region]) .≈ Array(ηs)[init:fin, :, :])
            end
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
                fs = interior(fs);

                for regions in [2, 4]
                    @info "  Testing diffusion of $fieldname on $regions partitions with $(typeof(closure).name.wrapper) on the $arch"

                    f = diffusion_cosine_test(grid; closure, field_name = fieldname, regions = regions)
                    f = construct_regionally(interior, f)

                    for region in 1:regions
                        init = Int(size(grid, 1) / regions) * (region - 1) + 1
                        fin  = Int(size(grid, 1) / regions) * region
                        @test all(Array(f[region])[1:(1+fin-init), :, :] .≈ Array(fs)[init:fin, :, :])
                    end
                end
            end
        end
    end
end
