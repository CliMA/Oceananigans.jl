include("dependencies_for_runtests.jl")

using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStar

function test_zstar_coordinate(model, Ni, Δt)
    
    ∫bᵢ = Field(Integral(model.tracers.b))
    w   = model.velocities.w
    Nz  = model.grid.Nz

    for _ in 1:Ni
        time_step!(model, Δt)
        ∫b = Field(Integral(model.tracers.b))

        @test ∫b[1, 1, 1] ≈ ∫bᵢ[1, 1, 1]
        @test maximum(interior(w, :, :, Nz+1)) < eps(eltype(w))
    end

    return nothing
end


@testset "Testing z-star coordinates" begin

    for arch in archs
        llg = LatitudeLongitudeGrid(arch; size = (10, 10, 10), latitude = (-10, 10), longitude = (-10, 10), z = (-10, 0))
        rtg = RectilinearGrid(arch; size = (10, 10, 10), x = (-10, 10), y = (-10, 10), z = (-10, 0))

        llgv = LatitudeLongitudeGrid(arch; size = (10, 10, 10), latitude = (-10, 10), longitude = (-10, 10), z = collect(-10:0))
        rtgv = RectilinearGrid(arch; size = (10, 10, 10), x = (-10, 10), y = (-10, 10), z = collect(-10:0))

        illg = ImmersedBoundaryGrid(llg, GridFittedBottom((x, y) -> - rand() - 5))
        irtg = ImmersedBoundaryGrid(rtg, GridFittedBottom((x, y) -> - rand() - 5))

        illgv = ImmersedBoundaryGrid(llgv, GridFittedBottom((x, y) -> - rand() - 5))
        irtgv = ImmersedBoundaryGrid(rtgv, GridFittedBottom((x, y) -> - rand() - 5))

        grids = [llg, rtg, llgv, rtgv, illg, irtg, illgv, irtgv]

        for grid in grids

            free_surface = SplitExplicitFreeSurface(grid; cfl = 0.75)
            model = HydrostaticFreeSurfaceModel(; grid, 
                                                  free_surface, 
                                                  tracers = (:b, :c), 
                                                  bouyancy = BuoyancTracer(),
                                                  generalized_vertical_coordinate = ZStar())

            bᵢ(x, y, z) = x < grid.Lx / 2 ? 0.06 : 0.01 

            set!(model, c = (x, y, z) -> rand(), b = bᵢ)

            test_zstar_coordinate(model, 100, 10)
        end
    end
end