include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

@testset "Bounded advection on Yin-Yang grids" for arch in archs
    grid = YinYangGrid(arch; size=(72, 24, 4), halo=(4, 4, 3), z=(-1000, 0))
    velocities = PrescribedVelocityFields(u=XFaceField(grid; boundary_conditions=nothing),
                                         v=YFaceField(grid; boundary_conditions=nothing),
                                         w=ZFaceField(grid; boundary_conditions=nothing))
    model = HydrostaticFreeSurfaceModel(grid; velocities,
        momentum_advection=nothing, tracer_advection=WENO(order=5, bounds=(0, 1)),
        tracers=:c, buoyancy=nothing, timestepper=:SplitRungeKutta3)
    set!(model, c=(λ, φ, z) -> ifelse(cosd(λ) * cosd(φ) > 1/2, 1, 0))
    time_step!(model, 1)
    limiter = model.advection.c.bounds.limiter
    @test minimum(limiter) ≥ 0
    @test maximum(limiter) ≤ 1
    @test minimum(limiter) < 1
    @test minimum(model.tracers.c) == 0
    @test maximum(model.tracers.c) == 1
end
