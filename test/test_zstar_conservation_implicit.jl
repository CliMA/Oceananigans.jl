include("dependencies_for_runtests.jl")
include("zstar_conservation_test_utils.jl")

@testset "ZStarCoordinate Implicit tracer conservation testset" begin
    z_stretched = MutableVerticalDiscretization(collect(-10:0))

    for arch in zstar_test_architectures()
        for topology in zstar_test_topologies(arch)
            grids = zstar_test_grids(arch, topology, z_stretched)

            # We test only SKR3 because AB2 is not conservative
            timestepper = :SplitRungeKutta3

            for grid in grids
                # Preconditioned conjugate gradient solver does not satisfy local conservation stricly to machine precision.
                implicit_free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient, preconditioner=nothing)

                # These combination of parameters lead to the parameter error:
                # Kernel invocation uses too much parameter memory.
                if (arch isa Distributed{<:GPU})
                    if (grid isa LatitudeLongitudeGrid)
                        continue
                    end
                    if (grid isa ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid})
                        continue
                    end
                end

                free_surface = implicit_free_surface
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

                    Δt = 2minutes
                    test_zstar_coordinate(model, 60, Δt)
                end
            end
        end
    end
end
