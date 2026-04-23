include("dependencies_for_runtests.jl")
include("zstar_conservation_test_utils.jl")

@testset "ZStarCoordinate Tripolar tracer conservation testset" begin
    z_stretched = MutableVerticalDiscretization(collect(-10:0))

    for arch in zstar_test_architectures()
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

                grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(gaussian_islands))
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
                mean_xspacing = mean(xspacings(grid, Face(), Face(), Center()))
                mean_yspacing = mean(yspacings(grid, Face(), Face(), Center()))

                Δ = mean((mean_xspacing, mean_yspacing))
                U = 1

                # Set streamfunction amplitude to Δ * U to yield velocities of order U.
                set!(ψ, U * Δ * rand(size(ψ)...))
                fill_halo_regions!(ψ)

                uᵢ = ∂y(ψ)
                vᵢ = -∂x(ψ)

                set!(model, c = (x, y, z) -> rand(), u = uᵢ, v = vᵢ, b = bᵢ, constant = 1)

                Δt = 2minutes
                test_zstar_coordinate(model, 300, Δt)
            end
        end
    end
end
