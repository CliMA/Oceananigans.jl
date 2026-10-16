include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: immersed_peripheral_node, immersed_inactive_node
using Oceananigans.Utils: getregion

@inline masked_initial_velocity(i, j, k, grid, velocity, loc) =
    ifelse(immersed_peripheral_node(i, j, k, grid, loc...), velocity[i, j, k], zero(grid))

@inline function masked_initial_transport(i, j, k, grid, velocity, loc)
    immersed = true
    for level in 1:grid.Nz
        immersed &= immersed_peripheral_node(i, j, level, grid, loc...)
    end
    return ifelse(immersed, velocity[i, j, 1], zero(grid))
end

@inline wet_constant_tracer_error(i, j, k, grid, tracer) =
    ifelse(immersed_inactive_node(i, j, k, grid, Center(), Center(), Center()), zero(grid), tracer[i, j, k] - 1)

@testset "Immersed multiregion initialization [$(typeof(arch))]" for arch in archs
    Nφ, H, R = 16, 1000, 1e6
    sphere = YinYangGrid(arch, Float32; size=(3Nφ, Nφ, 4), z=MutableVerticalDiscretization((-H, 0)), radius=R, halo=(4, 4, 4))
    bottom(λ, φ) = ifelse((abs(λ) < 15) & (φ > -35), zero(φ), oftype(φ, -1000))
    grid = ImmersedBoundaryGrid(sphere, GridFittedBottom(bottom))
    model = HydrostaticFreeSurfaceModel(grid; tracers=:c, buoyancy=nothing, momentum_advection=nothing,
        free_surface=SplitExplicitFreeSurface(grid; substeps=4, extend_halos=false),
        timestepper=:SplitRungeKutta3, vertical_coordinate=ZStarCoordinate())
    set!(model, u=(λ, φ, z) -> cosd(φ), v=0, c=1)
    u, v, _ = model.velocities
    U, V = model.free_surface.barotropic_velocities
    for (Lx, Ly, velocity, transport) in ((Face, Center, u, U), (Center, Face, v, V))
        loc = (Lx(), Ly(), Center())
        masked_velocity = Field(KernelFunctionOperation{Lx, Ly, Center}(masked_initial_velocity, grid, velocity, loc))
        masked_transport = Field(KernelFunctionOperation{Lx, Ly, Nothing}(masked_initial_transport, grid, transport, loc))
        compute!(masked_velocity)
        compute!(masked_transport)
        for region in 1:2
            @test maximum(abs, interior(getregion(masked_velocity, region))) == 0
            @test maximum(abs, interior(getregion(masked_transport, region))) == 0
        end
    end
    @test maximum(abs, interior(getregion(u, 1))) > 0.5

    Nt = 10
    for step in 1:Nt
        time_step!(model, 60f0)
    end
    error = Field(KernelFunctionOperation{Center, Center, Center}(wet_constant_tracer_error, grid, model.tracers.c))
    compute!(error)
    # The uniform tracer and the top transport velocity differ from 1 and 0 by rounding error alone: a few ulps
    # per step, and a few ulps of the vertical velocity H / Δ of the unit horizontal velocity across a cell of width Δ
    Δ = R * π / 2 / Nφ
    for region in 1:2
        @test maximum(abs, interior(getregion(error, region))) < Nt * eps(Float32)
        top_velocity = interior(getregion(model.transport_velocities.w, region), :, :, grid.Nz+1)
        @test maximum(abs, top_velocity) < 10 * eps(Float32) * H / Δ
    end
end
