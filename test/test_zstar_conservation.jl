include("dependencies_for_runtests.jl")

using Random
using Oceananigans: initialize!
using Oceananigans.ImmersedBoundaries: PartialCellBottom
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Models: ZStarCoordinate, ZCoordinate

grid_type(::RectilinearGrid{F, X, Y}) where {F, X, Y} = "Rect{$X, $Y}"
grid_type(::LatitudeLongitudeGrid{F, X, Y}) where {F, X, Y} = "LatLon{$X, $Y}"

grid_type(g::ImmersedBoundaryGrid) = "Immersed" * grid_type(g.underlying_grid)

function info_message(grid, free_surface, timestepper)
    msg1 = "$(typeof(architecture(grid))) "
    msg2 = grid_type(grid)
    msg3 = " with $(timestepper)"
    msg4 = " using a " * string(getnamewrapper(free_surface))
    return msg1 * msg2 * msg3 * msg4
end

# QuasiAdamsBashforth2 does not guarantee local conservation of tracers
test_local_conservation(timestepper) = timestepper != :QuasiAdamsBashforth2

@testset "ZStarCoordinate tracer conservation testset" begin
    z_stretched = MutableVerticalDiscretization(collect(-20:0))
    topologies  = ((Periodic, Periodic, Bounded),
                   (Periodic, Bounded, Bounded),
                   (Bounded, Periodic, Bounded),
                   (Bounded, Bounded, Bounded))

    for arch in archs
        for topology in topologies
            Random.seed!(1234)

            rtgv = RectilinearGrid(arch; size = (10, 10, 20), x = (0, 100kilometers), y = (-10kilometers, 10kilometers), topology, z = z_stretched)
            irtgv = ImmersedBoundaryGrid(deepcopy(rtgv),  GridFittedBottom((x, y) -> rand() - 10))
            prtgv = ImmersedBoundaryGrid(deepcopy(rtgv), PartialCellBottom((x, y) -> rand() - 10))

            if topology[2] == Bounded
                llgv = LatitudeLongitudeGrid(arch; size = (10, 10, 20), latitude = (0, 1), longitude = (0, 1), topology, z = z_stretched)

                illgv = ImmersedBoundaryGrid(deepcopy(llgv),  GridFittedBottom((x, y) -> rand() - 10))
                pllgv = ImmersedBoundaryGrid(deepcopy(llgv), PartialCellBottom((x, y) -> rand() - 10))

                # TODO: Partial cell bottom are broken at the moment and do not account for the Δz in the volumes
                # and vertical areas (see https://github.com/CliMA/Oceananigans.jl/issues/3958)
                # When this is issue is fixed we can add the partial cells to the testing.
                grids = [llgv, rtgv, illgv, irtgv] # , pllgv, prtgv]
            else
                grids = [rtgv, irtgv] #, prtgv]
            end

            for grid in grids
                split_free_surface    = SplitExplicitFreeSurface(grid; substeps=100)
                implicit_free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient)
                explicit_free_surface = ExplicitFreeSurface()

                for free_surface in [split_free_surface, explicit_free_surface]

                    # TODO: There are parameter space issues with ImplicitFreeSurface and a immersed LatitudeLongitudeGrid
                    # For the moment we are skipping these tests.
                    if (arch isa GPU) &&
                       (free_surface isa ImplicitFreeSurface) &&
                       (grid isa ImmersedBoundaryGrid) &&
                       (grid.underlying_grid isa LatitudeLongitudeGrid)

                        @info "  Skipping $(info_message(grid, free_surface)) because of parameter space issues"
                        continue
                    end

                    for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)

                        info_msg = info_message(grid, free_surface, timestepper)
                        if timestepper == :QuasiAdamsBashforth2
                            @info "  Skipping local conservation test for QuasiAdamsBashforth2 time stepping, which does not guarantee conservation of tracers."
                        else
                            @testset "$info_msg" begin
                                @info "  Testing a $info_msg"
                                model = HydrostaticFreeSurfaceModel(; grid = deepcopy(grid),
                                                                    free_surface,
                                                                    tracers = (:b, :c, :constant),
                                                                    timestepper,
                                                                    buoyancy = BuoyancyTracer(),
                                                                    vertical_coordinate = ZStarCoordinate())

                                bᵢ(x, y, z) = x < grid.Lx / 2 ? 0.06 : 0.01

                                set!(model, c = (x, y, z) -> rand(), b = bᵢ, constant = 1)

                                Δt = free_surface isa ExplicitFreeSurface ? 10 : 2minutes
                                test_zstar_coordinate(model, 100, Δt, test_local_conservation(timestepper))
                            end
                        end
                    end
                end
            end
        end
        
        @testset "RungeKutta5 ZStarCoordinate tracer conservation tests" begin
            @info "  Testing a ZStarCoordinate and Runge-Kutta 5th order time stepping"

            topology = topologies[2]
            rtg  = RectilinearGrid(arch; size=(10, 10, 20), x=(0, 100kilometers), y=(-10kilometers, 10kilometers), topology, z=z_stretched)
            llg  = LatitudeLongitudeGrid(arch; size=(10, 10, 20), latitude=(0, 1), longitude=(0, 1), topology, z=z_stretched)
            irtg = ImmersedBoundaryGrid(deepcopy(rtg), GridFittedBottom((x, y) -> rand()-10))
            illg = ImmersedBoundaryGrid(deepcopy(llg), GridFittedBottom((x, y) -> rand()-10))

            for grid in [rtg, llg] # , irtg, illg]
                split_free_surface = SplitExplicitFreeSurface(grid; substeps=50)
                model = HydrostaticFreeSurfaceModel(; grid,
                                                    free_surface = split_free_surface,
                                                    tracers = (:b, :c, :constant),
                                                    timestepper = :SplitRungeKutta5,
                                                    buoyancy = BuoyancyTracer(),
                                                    vertical_coordinate = ZStarCoordinate())

                bᵢ(x, y, z) = x < grid.Lx / 2 ? 0.06 : 0.01

                set!(model, c = (x, y, z) -> rand(), b = bᵢ, constant = 1)

                Δt = 2minutes
                test_zstar_coordinate(model, 100, Δt)
            end
        end

        @testset "TripolarGrid ZStarCoordinate tracer conservation tests" begin
            @info "Testing a ZStarCoordinate coordinate with a Tripolar grid on $(arch)..."

            grid = TripolarGrid(arch; size = (20, 20, 20), z = z_stretched)

            # Code credit:
            # https://github.com/PRONTOLab/GB-25/blob/682106b8487f94da24a64d93e86d34d560f33ffc/src/model_utils.jl#L65
            function mtn₁(λ, φ)
                λ₁ = 70
                φ₁ = 55
                dφ = 5
                return exp(-((λ - λ₁)^2 + (φ - φ₁)^2) / 2dφ^2)
            end

            function mtn₂(λ, φ)
                λ₁ = 70
                λ₂ = λ₁ + 180
                φ₂ = 55
                dφ = 5
                return exp(-((λ - λ₂)^2 + (φ - φ₂)^2) / 2dφ^2)
            end

            zb = - 20
            h  = - zb + 10
            gaussian_islands(λ, φ) = zb + h * (mtn₁(λ, φ) + mtn₂(λ, φ))

            grid = ImmersedBoundaryGrid(grid, GridFittedBottom(gaussian_islands))
            free_surface = SplitExplicitFreeSurface(grid; substeps=10)

            model = HydrostaticFreeSurfaceModel(; grid,
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
