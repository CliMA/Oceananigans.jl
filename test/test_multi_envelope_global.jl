include("dependencies_for_runtests.jl")

using Oceananigans.Grids: MultiEnvelopeGrid
using Oceananigans.Models: ZStarCoordinate

@testset "ME100 global multi-envelope builder" begin
    for arch in test_architectures()
        z = global_multi_envelope_z(; surface_levels=40, mid_levels=40, bottom_levels=20)
        grid = LatitudeLongitudeGrid(arch; size=(6, 6, 100),
                                     longitude=(-60, 60), latitude=(-40, 40), z,
                                     topology=(Bounded, Bounded, Bounded))

        @test grid isa MultiEnvelopeGrid
        @test size(grid, 3) == 100

        # surface-refined reference: shallowest spacing near the surface, much finer than the deepest
        Δr = on_architecture(CPU(), grid.z.Δᵃᵃᶜ)
        @test Δr[100] < Δr[1]               # top (surface) cell thinner than bottom cell
        @test Δr[100] < 50                  # fine surface reference resolution

        # Envelopes: pycnocline 250 m, mid 1500 m, bathymetry ~3000–5000 m (deep, strictly increasing).
        e1(λ, φ) = 250.0
        e2(λ, φ) = 1500.0
        e3(λ, φ) = 4000 + 1000 * cosd(2φ)
        materialize_envelopes!(grid, (e1, e2, e3))

        σᵉ = on_architecture(CPU(), grid.z.σᶜᶜᵉ)
        hcc = on_architecture(CPU(), grid.z.hᶜᶜ)
        for i in 1:6, j in 1:6
            @test all(σᵉ[i, j, k] > 0 for k in 1:100)   # monotonic map ⇒ positive Jacobian
            @test hcc[i, j, 1] > 1500                    # physical depth = deepest envelope
        end
    end
end

@testset "ME100 global conservation (LatitudeLongitudeGrid)" begin
    for arch in test_architectures()
        z = global_multi_envelope_z(; surface_levels=40, mid_levels=40, bottom_levels=20)
        grid = LatitudeLongitudeGrid(arch; size=(6, 6, 100),
                                     longitude=(-60, 60), latitude=(-40, 40), z,
                                     topology=(Bounded, Bounded, Bounded))

        e1(λ, φ) = 250.0
        e2(λ, φ) = 1500.0
        e3(λ, φ) = 4000 + 1000 * cosd(2φ)
        materialize_envelopes!(grid, (e1, e2, e3))

        model = HydrostaticFreeSurfaceModel(grid;
                                            free_surface = SplitExplicitFreeSurface(grid; substeps=20),
                                            tracers = (:c, :constant), buoyancy = nothing, coriolis = nothing,
                                            timestepper = :SplitRungeKutta3,
                                            vertical_coordinate = ZStarCoordinate())

        cᵢ(λ, φ, z) = 1 + 0.1 * sind(3λ)
        ηᵢ(λ, φ, z) = 0.2 * exp(-(λ^2 + φ^2) / 400)
        set!(model, c=cᵢ, constant=1, η=ηᵢ)

        ∫c = Field(Integral(model.tracers.c)); compute!(∫c)
        ∫c₀ = Array(interior(∫c, 1, 1, 1))[1]
        w = model.velocities.w; Nz = size(grid, 3)

        for _ in 1:20
            time_step!(model, 30.0)
        end
        compute!(∫c)
        ∫cₙ = Array(interior(∫c, 1, 1, 1))[1]

        @test isapprox(∫cₙ, ∫c₀; rtol=1e-9)
        @test maximum(abs, Array(interior(model.tracers.constant)) .- 1) < 1e-9
        @test maximum(abs, Array(interior(w, :, :, Nz+1))) < 1e-9
    end
end
