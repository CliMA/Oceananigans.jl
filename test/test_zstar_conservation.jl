include("dependencies_for_runtests.jl")

using Random
using Oceananigans: initialize!
using Oceananigans.ImmersedBoundaries: PartialCellBottom
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Models: ZStarCoordinate, ZCoordinate
using Oceananigans.DistributedComputations: DistributedGrid, @root, @handshake

grid_type(::RectilinearGrid{F, X, Y}) where {F, X, Y} = "Rect{$X, $Y}"
grid_type(::LatitudeLongitudeGrid{F, X, Y}) where {F, X, Y} = "LatLon{$X, $Y}"

grid_type(g::ImmersedBoundaryGrid) = "Immersed" * grid_type(g.underlying_grid)

function test_zstar_coordinate(model, Ni, Δt, test_local_conservation=true)

    bᵢ = deepcopy(model.tracers.b)
    cᵢ = deepcopy(model.tracers.c)

    ∫bᵢ = Field(Integral(bᵢ))
    ∫cᵢ = Field(Integral(cᵢ))
    compute!(∫bᵢ)
    compute!(∫cᵢ)

    w   = model.velocities.w
    Nz  = model.grid.Nz

    for step in 1:Ni
        time_step!(model, Δt)

        ∫b = Field(Integral(model.tracers.b))
        ∫c = Field(Integral(model.tracers.c))
        compute!(∫b)
        compute!(∫c)

        condition = interior(∫b, 1, 1, 1) ≈ interior(∫bᵢ, 1, 1, 1)
        if !condition
            @info "Stopping early: buoyancy not conserved at step $step"
        end
        @test condition

        condition = interior(∫c, 1, 1, 1) ≈ interior(∫cᵢ, 1, 1, 1)
        if !condition
            @info "Stopping early: c tracer not conserved at step $step"
        end
        @test condition

        condition = maximum(abs, interior(w, :, :, Nz+1)) < eps(eltype(w))
        if !condition
            @info "Stopping early: nonzero vertical velocity at top at step $step"
        end
        @test condition

        # Constancy preservation test
        if test_local_conservation
            @test maximum(model.tracers.constant) ≈ 1
            @test minimum(model.tracers.constant) ≈ 1
        end
    end

    return nothing
end

function info_message(grid, free_surface, timestepper)
    msg1 = "$(summary(architecture(grid))) "
    msg2 = grid_type(grid)
    msg3 = " with $(timestepper)"
    msg4 = " using a " * string(getnamewrapper(free_surface))
    return msg1 * msg2 * msg3 * msg4
end

@testset "ZStarCoordinate tracer conservation testset" begin
    z_stretched = MutableVerticalDiscretization(collect(-10:0))

    for arch in archs
        topologies = if arch isa Distributed
            # tests become too long because we test too many architectures,
            # given that `FullyConnected` acts as a `Periodic` we don't need
            # to test different topologies
            [(Bounded, Bounded, Bounded)]
        else
           [(Periodic, Periodic, Bounded),
            (Bounded, Bounded, Bounded)]
        end

        @handshake @info arch

        for topology in topologies
            Random.seed!(1234)

            rtgv = RectilinearGrid(arch; size = (40, 40, 10), x = (0, 100kilometers), y = (-10kilometers, 10kilometers), topology, z = z_stretched)
            irtgv = ImmersedBoundaryGrid(deepcopy(rtgv),  GridFittedBottom((x, y) -> rand() - 10))
            prtgv = ImmersedBoundaryGrid(deepcopy(rtgv), PartialCellBottom((x, y) -> rand() - 10))

            if topology[2] == Bounded
                llgv = LatitudeLongitudeGrid(arch; size = (40, 40, 10), latitude = (0, 1), longitude = (0, 1), topology, z = z_stretched)

                illgv = ImmersedBoundaryGrid(deepcopy(llgv),  GridFittedBottom((x, y) -> rand() - 10))
                pllgv = ImmersedBoundaryGrid(deepcopy(llgv), PartialCellBottom((x, y) -> rand() - 10))

                # TODO: Partial cell bottom are broken at the moment and do not account for the Δz in the volumes
                # and vertical areas (see https://github.com/CliMA/Oceananigans.jl/issues/3958)
                # When this is issue is fixed we can add the partial cells to the testing.
                grids = [llgv, rtgv, illgv, irtgv] # , pllgv, prtgv]
            else
                grids = [rtgv, irtgv] #, prtgv]
            end

            # We test only SKR3 because AB2 is not conservative
            timestepper = :SplitRungeKutta3

            for grid in grids
                # Preconditioned conjugate gradient solver does not satisfy local conservation stricly to machine precision.
                implicit_free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient, preconditioner=nothing)
                split_free_surface    = SplitExplicitFreeSurface(grid; substeps=8)
                explicit_free_surface = ExplicitFreeSurface()

                for free_surface in [explicit_free_surface, split_free_surface, implicit_free_surface]

                    # These combination of parameters lead to the parameter error:
                    # Kernel invocation uses too much parameter memory.
                    if (arch isa Distributed{<:GPU})
                        if (grid isa LatitudeLongitudeGrid) && free_surface === implicit_free_surface
                            continue
                        end
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
                        test_zstar_coordinate(model, 100, Δt, !(free_surface isa ImplicitFreeSurface))
                    end
                end
            end
        end

        if arch isa Distributed{<:GPU}
            # Unfortunately tripolar grid tests fail on the GPU because of
            # parameter space memory. We skip also these test
            continue
        end

        for fold_topology in (RightCenterFolded, RightFaceFolded)
            @testset "$(fold_topology) TripolarGrid ZStarCoordinate tracer conservation tests" begin
                # Check that the grid is correctly partitioned in case of a distributed architecture
                if arch isa Distributed && ((arch.ranks[1] != 1 || arch.ranks[2] == 1) || (fold_topology == RightFaceFolded))
                    continue
                end

                @root @info "Testing a ZStarCoordinate coordinate with a $(fold_topology) Tripolar grid on $(summary(arch))..."

                underlying_grid = TripolarGrid(arch; size = (40, 40, 10), z = z_stretched, fold_topology)

                # Code credit:
                # https://github.com/PRONTOLab/GB-25/blob/682106b8487f94da24a64d93e86d34d560f33ffc/src/model_utils.jl#L65
                function mtn(λ, φ, λ₀, φ₀)
                    dφ = 5
                    dλ = 5
                    return exp(-(λ - λ₀)^2 / 2dλ^2 - (φ - φ₀)^2 / 2dφ^2)
                end
                function mtns(λ, φ)
                    λ₀ = 70
                    φ₀ = 55
                    return mtn(λ, φ, λ₀, φ₀) + mtn(λ, φ, λ₀ + 180, φ₀) + mtn(λ, φ, λ₀ + 360, φ₀)
                end

                zb = - 20
                h  = - zb + 10
                gaussian_islands(λ, φ) = zb + h * mtns(λ, φ)

                grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(gaussian_islands))
                free_surface = SplitExplicitFreeSurface(grid; substeps=8)

                model = HydrostaticFreeSurfaceModel(grid;
                                                    free_surface,
                                                    tracers = (:b, :c, :constant),
                                                    buoyancy = BuoyancyTracer(),
                                                    timestepper = :SplitRungeKutta3,
                                                    vertical_coordinate = ZStarCoordinate())

                bᵢ(x, y, z) = y < 0 ? 0.06 : 0.01

                # Instead of initializing with random velocities, infer them from a random initial streamfunction
                # to ensure the velocity field is divergence-free at initialization.
                ψ = Field{Center, Center, Center}(grid)
                set!(ψ, rand(size(ψ)...))
                uᵢ = ∂y(ψ)
                vᵢ = -∂x(ψ)

                set!(model, c = (x, y, z) -> rand(), u = uᵢ, v = vᵢ, b = bᵢ, constant = 1)

                Δt = 2minutes
                test_zstar_coordinate(model, 300, Δt)
            end
        end
    end
end
