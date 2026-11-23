include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, materialize_buoyancy_gradients

function isopycnal_diffusion_model(arch; buoyancy, closure)
    grid = RectilinearGrid(arch, size=(8, 8, 8), extent=(1, 1, 1), halo=(3, 3, 3), topology=(Periodic, Bounded, Bounded))
    model = HydrostaticFreeSurfaceModel(grid; buoyancy, closure, tracers=(:T, :S))
    Tᵢ(x, y, z) = 20 + 5z + sin(2π * x) * cos(π * y)
    Sᵢ(x, y, z) = 35 + 0.1 * cos(2π * x)
    set!(model, T=Tᵢ, S=Sᵢ)
    return model
end

@testset "Materialized buoyancy gradients" begin
    issd = IsopycnalSkewSymmetricDiffusivity(κ_symmetric=1e-3, κ_skew=1e-3)

    @test materialize_buoyancy_gradients(nothing) == false
    @test materialize_buoyancy_gradients(ScalarDiffusivity(κ=1)) == false
    @test materialize_buoyancy_gradients(issd) == true
    @test materialize_buoyancy_gradients((ScalarDiffusivity(κ=1), issd)) == true
    @test materialize_buoyancy_gradients(CATKEVerticalDiffusivity()) == true
    @test materialize_buoyancy_gradients(TKEDissipationVerticalDiffusivity()) == true

    for arch in archs
        @testset "Isopycnal diffusion materializes buoyancy gradients [$arch]" begin
            materialized = isopycnal_diffusion_model(arch; buoyancy=SeawaterBuoyancy(), closure=issd)
            @test !isnothing(materialized.buoyancy.gradients)

            plain = isopycnal_diffusion_model(arch; buoyancy=SeawaterBuoyancy(), closure=ScalarDiffusivity(κ=1e-3))
            @test isnothing(plain.buoyancy.gradients)

            # An explicitly constructed BuoyancyForce is left as given
            grid = materialized.grid
            inline_buoyancy = BuoyancyForce(grid, SeawaterBuoyancy(); materialize_gradients=false)
            inline = isopycnal_diffusion_model(arch; buoyancy=inline_buoyancy, closure=issd)
            @test isnothing(inline.buoyancy.gradients)

            for model in (materialized, inline)
                for _ in 1:3
                    time_step!(model, 1)
                end
            end

            @test Array(interior(materialized.tracers.T)) ≈ Array(interior(inline.tracers.T))
            @test Array(interior(materialized.tracers.S)) ≈ Array(interior(inline.tracers.S))
        end
    end
end
