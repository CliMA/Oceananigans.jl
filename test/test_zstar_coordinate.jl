include("dependencies_for_runtests.jl")

using Random
using Oceananigans: initialize!
using Oceananigans.ImmersedBoundaries: PartialCellBottom
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Models: ZStar

function test_zstar_coordinate(model, Ni, Δt)
    
    bᵢ = deepcopy(model.tracers.b)
    cᵢ = deepcopy(model.tracers.c)

    ∫bᵢ = Field(Integral(bᵢ))
    ∫cᵢ = Field(Integral(cᵢ))
    w   = model.velocities.w
    Nz  = model.grid.Nz

    for _ in 1:Ni
        time_step!(model, Δt)
    end

    ∫b = Field(Integral(model.tracers.b))
    ∫c = Field(Integral(model.tracers.c))
    
    # Testing that:
    # (1) tracers are conserved down to machine precision
    # (2) vertical velocities are zero at the top surface
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

    z = MutableVerticalDiscretization((-20, 0))

    grid = RectilinearGrid(size = (2, 2, 20), 
                              x = (0, 2), 
                              y = (0, 1), 
                              z = z, 
                       topology = (Bounded, Periodic, Bounded))

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -10))

    model = HydrostaticFreeSurfaceModel(; grid, 
                                          free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                          vertical_coordinate = ZStar())

    @test znode(1, 1, 21, grid, C(), C(), F()) == 0
    @test column_depthᶜᶜᵃ(1, 1, grid) == 10
    @test  static_column_depthᶜᶜᵃ(1, 1, grid) == 10

    set!(model, η = [1 1; 2 2])
    set!(model, u = (x, y, z) -> x, v = (x, y, z) -> y)
    update_state!(model)

    @test σⁿ(1, 1, 1, grid, C(), C(), C()) == 11 / 10
    @test σⁿ(2, 1, 1, grid, C(), C(), C()) == 12 / 10

    @test znode(1, 1, 21, grid, C(), C(), F()) == 1
    @test znode(2, 1, 21, grid, C(), C(), F()) == 2
    @test rnode(1, 1, 21, grid, C(), C(), F()) == 0
    @test column_depthᶜᶜᵃ(1, 1, grid) == 11
    @test column_depthᶜᶜᵃ(2, 1, grid) == 12
    @test  static_column_depthᶜᶜᵃ(1, 1, grid) == 10
    @test  static_column_depthᶜᶜᵃ(2, 1, grid) == 10
end

@testset "MutableVerticalDiscretization tests" begin
    @info "testing the MutableVerticalDiscretization in ZCoordinate mode"

    z = MutableVerticalDiscretization((-20, 0))

    # A mutable immersed grid
    mutable_grid = RectilinearGrid(size=(2, 2, 20), x=(0, 2), y=(0, 1), z=z)
    mutable_grid = ImmersedBoundaryGrid(mutable_grid, GridFittedBottom((x, y) -> -10))

    # A static immersed grid
    static_grid = RectilinearGrid(size=(2, 2, 20), x=(0, 2), y=(0, 1), z=(-20, 0))
    static_grid = ImmersedBoundaryGrid(static_grid, GridFittedBottom((x, y) -> -10))

    # Make sure a model with a MutableVerticalDiscretization but ZCoordinate still runs and 
    # the results are the same as a model with a static vertical discretization.
    mutable_model = HydrostaticFreeSurfaceModel(; grid=mutable_grid, free_surface=ImplicitFreeSurface())
    static_model  = HydrostaticFreeSurfaceModel(; grid=static_grid,  free_surface=ImplicitFreeSurface())

    uᵢ = rand(size(mutable_model.velocities.u)...)
    vᵢ = rand(size(mutable_model.velocities.v)...)

    set!(mutable_model; u=uᵢ, v=vᵢ)
    set!(static_model;  u=uᵢ, v=vᵢ)

    static_sim  = Simulation(static_model;  Δt=1e-3, stop_iteration=100)
    mutable_sim = Simulation(mutable_model; Δt=1e-3, stop_iteration=100)         
    
    run!(mutable_sim)
    run!(static_sim)

    # Check that fields are the same
    um, vm, wm = mutable_model.velocities
    us, vs, ws = static_model.velocities

    @test all(um.data .≈ us.data)
    @test all(vm.data .≈ vs.data)
    @test all(wm.data .≈ ws.data)
    @test all(um.data .≈ us.data)
end

@testset "ZStar coordinate simulation testset" begin
    z_uniform   = MutableVerticalDiscretization((-20, 0))
    z_stretched = MutableVerticalDiscretization(collect(-20:0))
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
                
                for free_surface in [split_free_surface, implicit_free_surface, explicit_free_surface]
                    
                    # TODO: There are parameter space issues with ImplicitFreeSurface and a immersed LatitudeLongitudeGrid
                    # For the moment we are skipping these tests.
                    if (arch isa GPU) && 
                       (free_surface isa ImplicitFreeSurface) && 
                       (grid isa ImmersedBoundaryGrid) && 
                       (grid.underlying_grid isa LatitudeLongitudeGrid)

                        @info "  Skipping $(info_message(grid, free_surface)) because of parameter space issues"
                        continue
                    end

                    info_msg = info_message(grid, free_surface)
                    @testset "$info_msg" begin
                        @info "  Testing a $info_msg" 
                        model = HydrostaticFreeSurfaceModel(; grid, 
                                                            free_surface, 
                                                            tracers = (:b, :c), 
                                                            buoyancy = BuoyancyTracer(),
                                                            vertical_coordinate = ZStar())

                        bᵢ(x, y, z) = x < grid.Lx / 2 ? 0.06 : 0.01 

                        set!(model, c = (x, y, z) -> rand(), b = bᵢ)

                        Δt = free_surface isa ExplicitFreeSurface ? 10 : 2minutes
                        test_zstar_coordinate(model, 100, Δt)
                    end
                end
            end
        end

        @info "  Testing a ZStar and Runge Kutta 3rd order time stepping"

        topology = topologies[2]
        rtg  = RectilinearGrid(arch; size = (10, 10, 20), x = (0, 100kilometers), y = (-10kilometers, 10kilometers), topology, z = z_uniform)
        llg  = LatitudeLongitudeGrid(arch; size = (10, 10, 20), latitude = (0, 1), longitude = (0, 1), topology, z = z_uniform)
        irtg = ImmersedBoundaryGrid(rtg,   GridFittedBottom((x, y) -> rand() - 10))
        illg = ImmersedBoundaryGrid(llg,   GridFittedBottom((x, y) -> rand() - 10))

        for grid in [rtg, llg, irtg, illg]

            split_free_surface = SplitExplicitFreeSurface(grid; cfl = 0.75)
            model = HydrostaticFreeSurfaceModel(; grid, 
                                                free_surface = split_free_surface, 
                                                tracers = (:b, :c), 
                                                timestepper = :SplitRungeKutta3,
                                                buoyancy = BuoyancyTracer(),
                                                vertical_coordinate = ZStar())

            bᵢ(x, y, z) = x < grid.Lx / 2 ? 0.06 : 0.01 

            set!(model, c = (x, y, z) -> rand(), b = bᵢ)

            Δt = 2minutes
            test_zstar_coordinate(model, 100, Δt)
        end
    end
end