include("dependencies_for_runtests.jl")

using Random
using Oceananigans.ImmersedBoundaries: PartialCellBottom

function test_zstar_coordinate(model, Ni, Δt)
    
    ∫bᵢ = Field(Integral(model.tracers.b))
    ∫cᵢ = Field(Integral(model.tracers.c))
    w   = model.velocities.w
    Nz  = model.grid.Nz

    # Testing that at each timestep
    # (1) tracers are conserved down to machine precision
    # (2) vertical velocities are zero at the top surface
    for _ in 1:Ni
        time_step!(model, Δt)
    end

    ∫b = Field(Integral(model.tracers.b))
    ∫c = Field(Integral(model.tracers.c))

    @test interior(∫b, 1, 1, 1) ≈ interior(∫bᵢ, 1, 1, 1)
    @test interior(∫c, 1, 1, 1) ≈ interior(∫cᵢ, 1, 1, 1)
    @test maximum(interior(w, :, :, Nz+1)) < eps(eltype(w))

    return nothing
end

function info_message(grid)
    msg1 = "Testing z-star coordinates on $(architecture(grid)) on a "
    msg2 = string(getnamewrapper(grid)) 
    msg3 = grid isa ImmersedBoundaryGrid ? " on a " * string(getnamewrapper(grid.underlying_grid)) : ""
    msg4 = grid.Δzᵃᵃᶠ.reference isa Number ? " with uniform spacing" : " with stretched spacing"
    msg5 = grid isa ImmersedBoundaryGrid ? " and $(string(getnamewrapper(grid.immersed_boundary))) immersed boundary" : ""

    return msg1 * msg2 * msg3 * msg4 * msg5
end

@testset "ZStar coordinate testset" begin

    z_uniform   = ZStarVerticalCoordinate(-20, 0)
    z_stretched = ZStarVerticalCoordinate(collect(-20:0))
    topologies  = ((Periodic, Periodic, Bounded), 
                   (Periodic, Bounded, Bounded),
                   (Bounded, Periodic, Bounded),
                   (Bounded, Bounded, Bounded)) 

    for arch in archs
        for topology in topologies
            Random.seed!(1234)

            rtg  = RectilinearGrid(arch; size = (10, 10, 10), x = (0, 100kilometers), y = (-10kilometers, 10kilometers), topology, z = z_uniform)
            rtgv = RectilinearGrid(arch; size = (10, 10, 10), x = (0, 100kilometers), y = (-10kilometers, 10kilometers), topology, z = z_stretched)
            
            irtg  = ImmersedBoundaryGrid(rtg,  GridFittedBottom((x, y) -> - rand() - 10))
            irtgv = ImmersedBoundaryGrid(rtgv, GridFittedBottom((x, y) -> - rand() - 10))
            prtg  = ImmersedBoundaryGrid(rtg, PartialCellBottom((x, y) -> - rand() - 10))
            prtgv = ImmersedBoundaryGrid(rtgv, PartialCellBottom((x, y) -> - rand() - 10))

            if topology[2] == Bounded
                llg  = LatitudeLongitudeGrid(arch; size = (10, 10, 10), latitude = (0, 1), longitude = (0, 1), topology, z = z_uniform)
                llgv = LatitudeLongitudeGrid(arch; size = (10, 10, 10), latitude = (0, 1), longitude = (0, 1), topology, z = z_stretched)

                illg  = ImmersedBoundaryGrid(llg,  GridFittedBottom((x, y) -> - rand() - 5))
                illgv = ImmersedBoundaryGrid(llgv, GridFittedBottom((x, y) -> - rand() - 5))
                pllg  = ImmersedBoundaryGrid(llg,  PartialCellBottom((x, y) -> - rand() - 5))
                pllgv = ImmersedBoundaryGrid(llgv, PartialCellBottom((x, y) -> - rand() - 5))

                grids = [llg, rtg, llgv, rtgv, illg, irtg, illgv, irtgv, pllg, prtg, pllgv, prtgv]
            else
                grids = [rtg, rtgv, irtg, irtgv, prtg, prtgv]
            end

            for grid in grids
                @testset info_message(grid) begin

                    # TODO: minimum_xspacing(grid) on a Immersed GPU grid with ZStarVerticalCoordinate
                    # fails because it uses too much parameter space. Figure out a way to reduce it 
                    free_surface = SplitExplicitFreeSurface(grid; substeps = 20)
                    model = HydrostaticFreeSurfaceModel(; grid, 
                                                        free_surface, 
                                                        tracers = (:b, :c), 
                                                        buoyancy = BuoyancyTracer())

                    bᵢ(x, y, z) = x < grid.Lx / 2 ? 0.06 : 0.01 

                    set!(model, c = (x, y, z) -> rand(), b = bᵢ)

                    test_zstar_coordinate(model, 100, 10)
                end
            end
        end
    end
end