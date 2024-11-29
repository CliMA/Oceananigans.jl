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
    @test maximum(interior(w, :, :, Nz+1)) < model.grid.Nz * eps(eltype(w))

    return nothing
end

function info_message(grid)
    msg1 = "Testing z-star coordinates on $(architecture(grid)) on a "
    msg2 = string(getnamewrapper(grid)) 
    msg3 = grid isa ImmersedBoundaryGrid ? " on a " * string(getnamewrapper(grid.underlying_grid)) : ""
    msg4 = grid.z.Δᶠ isa Number ? " with uniform spacing" : " with stretched spacing"
    msg5 = grid isa ImmersedBoundaryGrid ? " and $(string(getnamewrapper(grid.immersed_boundary))) immersed boundary" : ""

    return msg1 * msg2 * msg3 * msg4 * msg5
end

const C = Center
const F = Face

@testset "ZStar coordinate scaling tests" begin
    @info "testing the ZStar coordinate scalings"

    z = ZStarVerticalCoordinate(-20, 0)

    grid = RectilinearGrid(size = (2, 1, 20), 
                              x = (0, 2), 
                              y = (0, 1), 
                              z = z, 
                       topology = (Bounded, Periodic, Bounded))

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -10))

    model = HydrostaticFreeSurfaceModel(; grid, free_surface = SplitExplicitFreeSurface(grid; substeps = 20))

    @test znode(1, 1, 21, grid, C(), C(), F()) == 0
    @test dynamic_column_depthᶜᶜᵃ(1, 1, grid) == 10
    @test  static_column_depthᶜᶜᵃ(1, 1, grid) == 10

    set!(model, η = [1, 2])
    set!(model, u = (x, y, z) -> x)

    initialize!(model)
    update_state!(model)

    @test e₃ⁿ(1, 1, 1, grid, C(), C(), C()) == 11 / 10
    @test e₃ⁿ(2, 1, 1, grid, C(), C(), C()) == 12 / 10
    @test e₃⁻(1, 1, 1, grid, C(), C(), C()) == 1
    @test e₃⁻(2, 1, 1, grid, C(), C(), C()) == 1

    @test znode(1, 1, 21, grid, C(), C(), C()) == 1
    @test znode(2, 1, 21, grid, C(), C(), C()) == 2
    @test rnode(1, 1, 21, grid, C(), C(), F()) == 0
    @test dynamic_column_depthᶜᶜᵃ(1, 1, grid) == 11
    @test dynamic_column_depthᶜᶜᵃ(2, 1, grid) == 12
    @test  static_column_depthᶜᶜᵃ(1, 1, grid) == 10
    @test  static_column_depthᶜᶜᵃ(2, 1, grid) == 10

    @test ∂t_e₃(1, 1, 1, grid) == 1 / 10
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

                # Partial cell bottom are broken at the moment and do not account for the Δz in the volumes
                # and vertical areas (see https://github.com/CliMA/Oceananigans.jl/issues/3958)
                # When this is issue is fixed we can add the partial cells to the testing.
                grids = [llg, rtg, llgv, rtgv, illg, irtg, illgv, irtgv] # , pllg, prtg, pllgv, prtgv]
            else
                grids = [rtg, rtgv, irtg, irtgv] #, prtg, prtgv]
            end

            for grid in grids
                info_msg = info_message(grid)
                
                split_free_surface = SplitExplicitFreeSurface(grid; substeps = 20)
                implicit_free_surface = ImplicitFreeSurface()
                explicit_free_surface = ExplicitFreeSurface()
                
                for free_surface in [split_free_surface, implicit_free_surface, explicit_free_surface]
                    @testset "$info_msg" begin
                        @info "  $info_msg of $(free_surface)" 
                        # TODO: minimum_xspacing(grid) on a Immersed GPU grid with ZStarVerticalCoordinate
                        # fails because it uses too much parameter space. Figure out a way to reduce it 
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
end