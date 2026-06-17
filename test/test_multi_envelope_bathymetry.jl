include("dependencies_for_runtests.jl")

using Oceananigans.Grids: MultiEnvelopeVerticalDiscretization
using Oceananigans.Models: ZStarCoordinate

@testset "shelf-safe envelopes from bathymetry" begin
    # Deep open ocean: envelopes sit at their targets.
    deep = [e(0, 0) for e in shelf_safe_envelopes((x, y) -> 4000.0, (250, 1500); minimum_thickness=10)]
    @test deep == [250.0, 1500.0, 4000.0]
    @test issorted(deep, lt = <)

    # Shallow shelf (bottom 80 m < 250 m): envelopes compress, strictly increasing, never collapse.
    shelf = [e(0, 0) for e in shelf_safe_envelopes((x, y) -> 80.0, (250, 1500); minimum_thickness=10)]
    @test issorted(shelf, lt = <)
    @test shelf[end] == 80.0
    @test minimum(diff(shelf)) ≥ 10 - 1e-9
end

@testset "materialize across a shelf keeps σᵉ > 0" begin
    for arch in test_architectures()
        z = MultiEnvelopeVerticalDiscretization(collect(range(-1000, 0, length=21));
                                                formulation=MultiEnvelope(level_counts=(8, 6, 6)))
        grid = RectilinearGrid(arch; size=(16, 4, 20), x=(0, 1e5), y=(0, 2.5e4), z,
                               topology=(Bounded, Periodic, Bounded))

        bathymetry(x, y) = 100 + 700 * (x / 1e5)   # 100 m shelf → 800 m deep, crosses the 250 m envelope
        materialize_envelopes!(grid, shelf_safe_envelopes(bathymetry, (250, 600); minimum_thickness=10))

        σᵉ = on_architecture(CPU(), grid.z.σᶜᶜᵉ)
        @test minimum(σᵉ[i, j, k] for i in 1:16, j in 1:4, k in 1:20) > 0
    end
end

@testset "envelope smoothing reduces roughness" begin
    z = MultiEnvelopeVerticalDiscretization(collect(range(-1000, 0, length=11));
                                            formulation=MultiEnvelope(level_counts=(4, 3, 3)))
    grid = RectilinearGrid(size=(16, 16, 10), x=(0, 1e5), y=(0, 1e5), z,
                           topology=(Bounded, Bounded, Bounded))
    f = Field((Center(), Center(), nothing), grid)
    set!(f, (x, y) -> 500 + 300 * sign(sin(6π * x / 1e5)))
    roughness_before = sum(abs2, Array(interior(f)) .- 500)
    smooth_envelope_field!(f, grid; passes=8)
    @test sum(abs2, Array(interior(f)) .- 500) < roughness_before
end

@testset "conservation across a shelf-crossing bathymetry" begin
    for arch in test_architectures()
        z = MultiEnvelopeVerticalDiscretization(collect(range(-1000, 0, length=21));
                                                formulation=MultiEnvelope(level_counts=(8, 6, 6)))
        grid = RectilinearGrid(arch; size=(16, 4, 20), x=(0, 1e5), y=(0, 2.5e4), z,
                               topology=(Bounded, Periodic, Bounded))
        bathymetry(x, y) = 100 + 700 * (x / 1e5)
        materialize_envelopes!(grid, shelf_safe_envelopes(bathymetry, (250, 600); minimum_thickness=10))

        model = HydrostaticFreeSurfaceModel(grid;
                                            free_surface = SplitExplicitFreeSurface(grid; substeps=10),
                                            tracers = (:c, :constant), buoyancy = nothing,
                                            timestepper = :SplitRungeKutta3,
                                            vertical_coordinate = ZStarCoordinate())
        set!(model, c=(x, y, z) -> 1 + 0.1 * sin(2π * x / 1e5), constant=1,
             η=(x, y, z) -> 0.05 * exp(-((x - 5e4)^2) / 1e8))

        ∫c = Field(Integral(model.tracers.c)); compute!(∫c)
        ∫c₀ = Array(interior(∫c, 1, 1, 1))[1]
        w = model.velocities.w; Nz = size(grid, 3)
        for _ in 1:40
            time_step!(model, 15.0)
        end
        compute!(∫c)
        @test isapprox(Array(interior(∫c, 1, 1, 1))[1], ∫c₀; rtol=1e-9)
        @test maximum(abs, Array(interior(model.tracers.constant)) .- 1) < 1e-9
        @test maximum(abs, Array(interior(w, :, :, Nz+1))) < 1e-9
    end
end
