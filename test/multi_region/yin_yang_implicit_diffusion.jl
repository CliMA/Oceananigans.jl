include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.Utils: getregion

@testset "Implicit vertical diffusion on Yin-Yang grids" for arch in archs, immersed in (false, true)
    sphere = YinYangGrid(arch, Float32; size=(48, 16, 4), halo=(3, 3, 3), z=(-4, 0))
    grid = immersed ? ImmersedBoundaryGrid(sphere, GridFittedBottom((λ, φ) -> -4)) : sphere
    closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), Float32; κ=1)
    model = HydrostaticFreeSurfaceModel(grid; closure, tracers=:c, buoyancy=nothing,
                                       momentum_advection=nothing, tracer_advection=nothing)
    set!(model, c=(λ, φ, z) -> 1 + cospi((z + 4) / 4))
    expected = CenterField(grid)
    damping = 1 / (1 + 4sinpi(1/8)^2)
    set!(expected, (λ, φ, z) -> 1 + damping * cospi((z + 4) / 4))

    for region in 1:2
        regional_model = getregion(model, region)
        implicit_step!(regional_model.tracers.c, regional_model.timestepper.implicit_solver,
                       regional_model.closure, regional_model.closure_fields, Val(1),
                       regional_model.clock, Oceananigans.fields(regional_model), 1f0)
        actual = on_architecture(CPU(), interior(regional_model.tracers.c))
        reference = on_architecture(CPU(), interior(getregion(expected, region)))
        @test actual ≈ reference rtol=5f-6
        @test mean(Float64, actual) ≈ 1 rtol=5f-6
    end
end
