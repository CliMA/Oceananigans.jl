include("dependencies_for_runtests.jl")

using Random
using Oceananigans: initialize!
using Oceananigans.ImmersedBoundaries: PartialCellBottom

function test_zstar_coordinate(model, Ni, Δt)
    
    bᵢ = deepcopy(model.tracers.b)
    cᵢ = deepcopy(model.tracers.c)

    ∫bᵢ = Field(Integral(bᵢ))
    ∫cᵢ = Field(Integral(cᵢ))
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
    @test maximum(abs, interior(w, :, :, Nz+1)) < eps(eltype(w))
    
    return nothing
end

function info_message(grid, free_surface)
    msg1 = "$(architecture(grid)) "
    msg2 = string(getnamewrapper(grid)) 
    msg3 = grid isa ImmersedBoundaryGrid ? " on a " * string(getnamewrapper(grid.underlying_grid)) : ""
    msg4 = grid.z.Δᵃᵃᶠ isa Number ? " with uniform spacing" : " with stretched spacing"
    msg5 = grid isa ImmersedBoundaryGrid ? " and $(string(getnamewrapper(grid.immersed_boundary))) immersed boundary" : ""
    msg6 = " using a " * string(getnamewrapper(free_surface))
    return msg1 * msg2 * msg3 * msg4 * msg5 * msg6
end

const C = Center
const F = Face

@testset "ZStar coordinate scaling tests" begin
    @info "testing the ZStar coordinate scalings"

    z = ZStarVerticalCoordinate(-20, 0)

    grid = RectilinearGrid(size = (2, 2, 20), 
                              x = (0, 2), 
                              y = (0, 1), 
                              z = z, 
                       topology = (Bounded, Periodic, Bounded))

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -10))

    model = HydrostaticFreeSurfaceModel(; grid, free_surface = SplitExplicitFreeSurface(grid; substeps = 20))

    @test znode(1, 1, 21, grid, C(), C(), F()) == 0
    @test dynamic_column_depthᶜᶜᵃ(1, 1, grid) == 10
    @test  static_column_depthᶜᶜᵃ(1, 1, grid) == 10

    set!(model, η = [1 1; 2 2])
    set!(model, u = (x, y, z) -> x, v = (x, y, z) -> y)
    update_state!(model)

    @test σⁿ(1, 1, 1, grid, C(), C(), C()) == 11 / 10
    @test σⁿ(2, 1, 1, grid, C(), C(), C()) == 12 / 10

    @test znode(1, 1, 21, grid, C(), C(), F()) == 1
    @test znode(2, 1, 21, grid, C(), C(), F()) == 2
    @test rnode(1, 1, 21, grid, C(), C(), F()) == 0
    @test dynamic_column_depthᶜᶜᵃ(1, 1, grid) == 11
    @test dynamic_column_depthᶜᶜᵃ(2, 1, grid) == 12
    @test  static_column_depthᶜᶜᵃ(1, 1, grid) == 10
    @test  static_column_depthᶜᶜᵃ(2, 1, grid) == 10
end

@testset "ZStar coordinate simulation testset" begin
    z_uniform   = ZStarVerticalCoordinate(-20, 0)
    z_stretched = ZStarVerticalCoordinate(collect(-20:0))
    topologies  = ((Periodic, Periodic, Bounded), 
                   (Periodic, Bounded, Bounded),
                   (Bounded, Periodic, Bounded),
                   (Bounded, Bounded, Bounded)) 

    for arch in archs
        for topology in topologies
            Random.seed!(1234)

            rtg  = RectilinearGrid(arch; size = (10, 10, 20), x = (0, 100kilometers), y = (-10kilometers, 10kilometers), topology, z = z_uniform)
            rtgv = RectilinearGrid(arch; size = (10, 10, 20), x = (0, 100kilometers), y = (-10kilometers, 10kilometers), topology, z = z_stretched)
            
            irtg  = ImmersedBoundaryGrid(rtg,   GridFittedBottom((x, y) -> rand() - 10))
            irtgv = ImmersedBoundaryGrid(rtgv,  GridFittedBottom((x, y) -> rand() - 10))
            prtg  = ImmersedBoundaryGrid(rtg,  PartialCellBottom((x, y) -> rand() - 10))
            prtgv = ImmersedBoundaryGrid(rtgv, PartialCellBottom((x, y) -> rand() - 10))

            if topology[2] == Bounded
                llg  = LatitudeLongitudeGrid(arch; size = (10, 10, 20), latitude = (0, 1), longitude = (0, 1), topology, z = z_uniform)
                llgv = LatitudeLongitudeGrid(arch; size = (10, 10, 20), latitude = (0, 1), longitude = (0, 1), topology, z = z_stretched)

                illg  = ImmersedBoundaryGrid(llg,   GridFittedBottom((x, y) -> rand() - 10))
                illgv = ImmersedBoundaryGrid(llgv,  GridFittedBottom((x, y) -> rand() - 10))
                pllg  = ImmersedBoundaryGrid(llg,  PartialCellBottom((x, y) -> rand() - 10))
                pllgv = ImmersedBoundaryGrid(llgv, PartialCellBottom((x, y) -> rand() - 10))

                # TODO: Partial cell bottom are broken at the moment and do not account for the Δz in the volumes
                # and vertical areas (see https://github.com/CliMA/Oceananigans.jl/issues/3958)
                # When this is issue is fixed we can add the partial cells to the testing.
                grids = [llg, rtg, llgv, rtgv, illg, irtg, illgv, irtgv] # , pllg, prtg, pllgv, prtgv]
            else
                grids = [rtg, rtgv, irtg, irtgv] #, prtg, prtgv]
            end

            for grid in grids
                split_free_surface    = SplitExplicitFreeSurface(grid; cfl = 0.75)
                
                implicit_free_surface = ImplicitFreeSurface()
                explicit_free_surface = ExplicitFreeSurface()
                
                for free_surface in [explicit_free_surface, implicit_free_surface, explicit_free_surface]
                    info_msg = info_message(grid, free_surface)
                    @testset "$info_msg" begin
                        @info "  Testing a $info_msg" 
                        # TODO: minimum_xspacing(grid) on a Immersed GPU grid with ZStarVerticalCoordinate
                        # fails because it uses too much parameter space. Figure out a way to reduce it 
                        model = HydrostaticFreeSurfaceModel(; grid, 
                                                            free_surface, 
                                                            tracers = (:b, :c), 
                                                            buoyancy = BuoyancyTracer())

                        bᵢ(x, y, z) = x < grid.Lx / 2 ? 0.06 : 0.01 

                        set!(model, c = (x, y, z) -> rand(), b = bᵢ)

                        Δt = free_surface isa ExplicitFreeSurface ? 10 : 5minutes
                        test_zstar_coordinate(model, 100, 5minutes)
                    end
                end
            end
        end
    end
end