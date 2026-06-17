include("dependencies_for_runtests.jl")

using Oceananigans.Grids: MultiEnvelopeVerticalDiscretization, MultiEnvelopeGrid
using Oceananigans.Operators: σⁿ
using Oceananigans.Grids: static_column_depthᶜᶜᵃ, znode

@testset "LinearEnvelope formulation: metric and physical depth" begin
    for arch in test_architectures()
        Lz = 1000.0
        z = MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, length=11)); formulation=LinearEnvelope())
        grid = RectilinearGrid(arch; size=(8, 8, 10), x=(0, 1e5), y=(0, 1e5), z,
                               topology=(Bounded, Bounded, Bounded))

        bottom_height(x, y) = 700 + 200 * cos(2π * x / 1e5) * cos(2π * y / 1e5)   # 500–900 m, < Lz
        materialize_envelopes!(grid, bottom_height)

        σᵉ = on_architecture(CPU(), grid.z.σᶜᶜᵉ)
        Hcc = on_architecture(CPU(), grid.z.hᶜᶜ)

        # σᵉ = H/Lz, depth-independent (same for every k); physical resting depth h = H.
        for i in 1:8, j in 1:8
            x = (i - 0.5) * 1e5 / 8
            y = (j - 0.5) * 1e5 / 8
            H = bottom_height(x, y)
            @test Hcc[i, j, 1] ≈ H rtol=1e-12
            for k in 1:10
                @test σᵉ[i, j, k] ≈ H / Lz rtol=1e-12
            end
        end

        # σᵉ must equal the finite-difference Jacobian Δz/Δr of the resting coordinate (η = 0, σ_fs = 1).
        Δr = Lz / 10
        for i in 1:4:8, j in 1:4:8
            for k in 1:10
                zᵏ   = znode(i, j, k,   grid, Center(), Center(), Face())
                zᵏ⁺¹ = znode(i, j, k+1, grid, Center(), Center(), Face())
                @test (zᵏ⁺¹ - zᵏ) / Δr ≈ σⁿ(i, j, k, grid, Center(), Center(), Center()) rtol=1e-10
            end
            # bottom face sits at the physical envelope depth −H
            x = (i - 0.5) * 1e5 / 8; y = (j - 0.5) * 1e5 / 8
            @test znode(i, j, 1, grid, Center(), Center(), Face()) ≈ -bottom_height(x, y) rtol=1e-10
        end
    end
end

@testset "MultiEnvelope formulation: per-zone depth-dependent σᵉ" begin
    for arch in test_architectures()
        Lz = 1000.0
        z = MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, length=11));
                                                formulation=MultiEnvelope(level_counts=(4, 3, 3)))
        grid = RectilinearGrid(arch; size=(8, 8, 10), x=(0, 1e5), y=(0, 1e5), z,
                               topology=(Bounded, Bounded, Bounded))

        e1(x, y) = 250.0                                            # pycnocline envelope
        e2(x, y) = 600.0                                            # mid envelope
        e3(x, y) = 850 + 50 * cos(2π * x / 1e5) * cos(2π * y / 1e5) # bottom envelope, 800–900 < Lz
        materialize_envelopes!(grid, (e1, e2, e3))

        σᵉ = on_architecture(CPU(), grid.z.σᶜᶜᵉ)
        hcc = on_architecture(CPU(), grid.z.hᶜᶜ)

        # Reference zone thicknesses: zone1 (top 4 levels) 400 m, zone2 300 m, zone3 300 m.
        for i in 1:8, j in 1:8
            x = (i - 0.5) * 1e5 / 8; y = (j - 0.5) * 1e5 / 8
            σ_zone1 = (e1(x, y) - 0)        / 400
            σ_zone2 = (e2(x, y) - e1(x, y)) / 300
            σ_zone3 = (e3(x, y) - e2(x, y)) / 300
            for k in 7:10; @test σᵉ[i, j, k] ≈ σ_zone1 rtol=1e-12; end   # surface zone (top k)
            for k in 4:6;  @test σᵉ[i, j, k] ≈ σ_zone2 rtol=1e-12; end
            for k in 1:3;  @test σᵉ[i, j, k] ≈ σ_zone3 rtol=1e-12; end
            @test hcc[i, j, 1] ≈ e3(x, y) rtol=1e-12                     # physical depth = deepest envelope
            @test all(σᵉ[i, j, :] .> 0)                                 # monotonic map ⇒ σᵉ > 0
        end

        # Bottom face sits at the deepest envelope; surface face at 0.
        for i in 1:4:8, j in 1:4:8
            x = (i - 0.5) * 1e5 / 8; y = (j - 0.5) * 1e5 / 8
            @test znode(i, j, 1,  grid, Center(), Center(), Face()) ≈ -e3(x, y) rtol=1e-10
            @test znode(i, j, 11, grid, Center(), Center(), Face()) ≈ 0 atol=1e-10
        end
    end
end

@testset "LinearEnvelope conservation (horizontally-varying σᵉ)" begin
    for arch in test_architectures()
        Lz = 1000.0
        z = MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, length=11)); formulation=LinearEnvelope())
        grid = RectilinearGrid(arch; size=(8, 8, 10), x=(0, 1e5), y=(0, 1e5), z,
                               topology=(Bounded, Bounded, Bounded))

        bottom_height(x, y) = 700 + 200 * cos(2π * x / 1e5) * cos(2π * y / 1e5)
        materialize_envelopes!(grid, bottom_height)

        model = HydrostaticFreeSurfaceModel(grid;
                                            free_surface = SplitExplicitFreeSurface(grid; substeps=10),
                                            tracers = (:c, :constant), buoyancy = nothing,
                                            timestepper = :SplitRungeKutta3,
                                            vertical_coordinate = ZStarCoordinate())

        cᵢ(x, y, z) = 1 + 0.1 * sin(2π * x / 1e5)
        ηᵢ(x, y, z) = 0.2 * exp(-((x - 5e4)^2 + (y - 5e4)^2) / 4e8)
        set!(model, c=cᵢ, constant=1, η=ηᵢ)

        ∫c = Field(Integral(model.tracers.c)); compute!(∫c)
        ∫c₀ = Array(interior(∫c, 1, 1, 1))[1]
        w = model.velocities.w; Nz = size(grid, 3)

        for _ in 1:60
            time_step!(model, 20.0)
        end
        compute!(∫c)
        ∫cₙ = Array(interior(∫c, 1, 1, 1))[1]

        @test isapprox(∫cₙ, ∫c₀; rtol=1e-10)
        @test maximum(abs, Array(interior(model.tracers.constant)) .- 1) < 1e-10
        @test maximum(abs, Array(interior(w, :, :, Nz+1))) < 1e-10
    end
end

@testset "MultiEnvelope conservation (depth-dependent σᵉ from formulation)" begin
    for arch in test_architectures()
        Lz = 1000.0
        z = MultiEnvelopeVerticalDiscretization(collect(range(-Lz, 0, length=11));
                                                formulation=MultiEnvelope(level_counts=(4, 3, 3)))
        grid = RectilinearGrid(arch; size=(8, 8, 10), x=(0, 1e5), y=(0, 1e5), z,
                               topology=(Bounded, Bounded, Bounded))

        e1(x, y) = 250.0
        e2(x, y) = 600.0
        e3(x, y) = 850 + 50 * cos(2π * x / 1e5) * cos(2π * y / 1e5)
        materialize_envelopes!(grid, (e1, e2, e3))

        model = HydrostaticFreeSurfaceModel(grid;
                                            free_surface = SplitExplicitFreeSurface(grid; substeps=10),
                                            tracers = (:c, :constant), buoyancy = nothing,
                                            timestepper = :SplitRungeKutta3,
                                            vertical_coordinate = ZStarCoordinate())

        cᵢ(x, y, z) = 1 + 0.1 * sin(2π * x / 1e5)
        ηᵢ(x, y, z) = 0.2 * exp(-((x - 5e4)^2 + (y - 5e4)^2) / 4e8)
        set!(model, c=cᵢ, constant=1, η=ηᵢ)

        ∫c = Field(Integral(model.tracers.c)); compute!(∫c)
        ∫c₀ = Array(interior(∫c, 1, 1, 1))[1]
        w = model.velocities.w; Nz = size(grid, 3)

        for _ in 1:60
            time_step!(model, 20.0)
        end
        compute!(∫c)
        ∫cₙ = Array(interior(∫c, 1, 1, 1))[1]

        @test isapprox(∫cₙ, ∫c₀; rtol=1e-10)
        @test maximum(abs, Array(interior(model.tracers.constant)) .- 1) < 1e-10
        @test maximum(abs, Array(interior(w, :, :, Nz+1))) < 1e-10
    end
end
