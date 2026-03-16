include("dependencies_for_runtests.jl")
include("zstar_conservation_test_utils.jl")

@testset "ZStarCoordinate Explicit/SplitExplicit tracer conservation testset" begin
    z_stretched = MutableVerticalDiscretization(collect(-10:0))

    for arch in zstar_test_architectures()
        for topology in zstar_test_topologies(arch)
            grids = zstar_test_grids(arch, topology, z_stretched)

            # We test only SKR3 because AB2 is not conservative
            timestepper = :SplitRungeKutta3

            for grid in grids
                split_free_surface    = SplitExplicitFreeSurface(grid; substeps=8)
                explicit_free_surface = ExplicitFreeSurface()

                for free_surface in [explicit_free_surface, split_free_surface]

                    # These combination of parameters lead to the parameter error:
                    # Kernel invocation uses too much parameter memory.
                    if (arch isa Distributed{<:GPU})
                        if (grid isa ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid})
                            continue
                        end
                    end

                    info_msg = info_message(grid, free_surface, timestepper)
                    @testset "$info_msg" begin
                        @root @info "  Testing a $info_msg"
                        model = HydrostaticFreeSurfaceModel(deepcopy(grid);
                                                            free_surface,
                                                            tracers = (:b, :c, :constant),
                                                            timestepper,
                                                            buoyancy = BuoyancyTracer(),
                                                            vertical_coordinate = ZStarCoordinate())

                        bᵢ(x, y, z) = x < grid.Lx / 2 ? 0.06 : 0.01

                        set!(model, c = (x, y, z) -> rand(), b = bᵢ, constant = 1)

                        Δt = free_surface isa ExplicitFreeSurface ? 10 : 2minutes
                        test_zstar_coordinate(model, 100, Δt)
                    end
                end
            end
        end
    end
end
