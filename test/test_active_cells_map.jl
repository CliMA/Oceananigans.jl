include("dependencies_for_runtests.jl")

using Oceananigans.Operators: hack_cosd

function Δ_min(grid) 
    Δx_min = minimum_xspacing(grid, Center(), Center(), Center())
    Δy_min = minimum_yspacing(grid, Center(), Center(), Center())
    return min(Δx_min, Δy_min)
end

@inline Gaussian(x, y, L) = exp(-(x^2 + y^2) / L^2)

function solid_body_rotation_test(grid)

    free_surface = SplitExplicitFreeSurface(grid; substeps = 5, gravitational_acceleration = 1)
    coriolis     = HydrostaticSphericalCoriolis(rotation_rate = 1)

    model = HydrostaticFreeSurfaceModel(; grid,
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

    # Gaussian leads to values with O(1e-60),
    # too small for repetible testing. We cap it at 0.1
    cᵢ(λ, φ, z) = max(Gaussian(λ, φ - 5, 10), 0.1)
    vᵢ(λ, φ, z) = 0.1

    set!(model, u=uᵢ, η=ηᵢ, c=cᵢ)

    Δt = 0.1 * Δ_min(grid) / sqrt(g * grid.Lz) 
    
    simulation = Simulation(model; Δt, stop_iteration = 10)
    run!(simulation)

    return merge(model.velocities, model.tracers, (; η = model.free_surface.η))
end

Nx = 32
Ny = 32

for arch in archs
    @testset "Active cells map solid body rotation" begin
        underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, 10),
                                                halo = (4, 4, 4),
                                                latitude = (-80, 80),
                                                longitude = (-160, 160),
                                                z = (-10, 0),
                                                radius = 1,
                                                topology=(Bounded, Bounded, Bounded))

        # Make sure the bottom is the same
        bottom_height = zeros(Nx, Ny)
        for i in 1:Nx, j in 1:Ny
            bottom_height[i, j] = - rand() * 5 - 5
        end

        immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
        immersed_active_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map = true)

        ua, va, wa, ca, ηa = solid_body_rotation_test(immersed_active_grid)
        u, v, w, c, η      = solid_body_rotation_test(immersed_grid)

        ua = interior(on_architecture(CPU(), ua))
        va = interior(on_architecture(CPU(), va))
        wa = interior(on_architecture(CPU(), wa))
        ca = interior(on_architecture(CPU(), ca))
        ηa = interior(on_architecture(CPU(), ηa))

        u = interior(on_architecture(CPU(), u))
        v = interior(on_architecture(CPU(), v))
        w = interior(on_architecture(CPU(), w))
        c = interior(on_architecture(CPU(), c))
        η = interior(on_architecture(CPU(), η))

        atol = eps(eltype(immersed_grid))
        rtol = sqrt(eps(eltype(immersed_grid)))

        @test all(isapprox(u, ua; atol, rtol))
        @test all(isapprox(v, va; atol, rtol))
        @test all(isapprox(w, wa; atol, rtol))
        @test all(isapprox(c, ca; atol, rtol))
        @test all(isapprox(η, ηa; atol, rtol))
    end
end
