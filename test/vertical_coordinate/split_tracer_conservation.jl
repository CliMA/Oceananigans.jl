include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "split_tracer_stepping_test_utils.jl"))

using Oceananigans.OrthogonalSphericalShellGrids: RightCenterFolded, RightFaceFolded

ridge(x₀, width, height, depth) = (x, y) -> - depth + height * exp(- ((x - x₀) / width)^2)

function zstar_basin_grids(arch)
    z = MutableVerticalDiscretization(collect(range(-60, 0, length=7)))

    rectilinear = RectilinearGrid(arch; size = (16, 8, 6), halo = (4, 4, 4), x = (0, 32kilometers), y = (0, 16kilometers),
                                  z, topology = (Bounded, Bounded, Bounded))

    latitude_longitude = LatitudeLongitudeGrid(arch; size = (16, 8, 6), halo = (4, 4, 4), longitude = (0, 0.32),
                                               latitude = (30, 30.16), z)

    return (ImmersedBoundaryGrid(rectilinear, GridFittedBottom(ridge(16kilometers, 4kilometers, 30, 60))),
            ImmersedBoundaryGrid(latitude_longitude, PartialCellBottom(ridge(0.16, 0.04, 30, 60))))
end

function tripolar_grid(arch, fold_topology)
    mountain(λ, φ, λ₀, φ₀) = exp(- (λ - λ₀)^2 / 2 / 5^2 - (φ - φ₀)^2 / 2 / 5^2)
    islands(λ, φ) = - 20 + 30 * (mountain(λ, φ, 70, 55) + mountain(λ, φ, 250, 55) + mountain(λ, φ, 430, 55))
    underlying_grid = TripolarGrid(arch; size = (20, 32, 5), z = MutableVerticalDiscretization(collect(-10:2:0)), fold_topology)
    return ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(islands))
end

function test_split_zstar_conservation(model, ratio, Δt, cycles)
    active = active_cells(model.grid)
    inventory₀ = tracer_inventory(model.tracers.c)
    η₀ = Array(interior(model.free_surface.displacement))

    for step in 1:cycles * ratio
        time_step!(model, Δt)
    end

    # The free surface must move for the test to exercise the z-star long step
    @test maximum(abs, Array(interior(model.free_surface.displacement)) .- η₀) > 1e-3

    # The accumulated transports telescope to η₁ - η₀, so the long-step vertical velocity vanishes at the top up to
    # the round-off of a column sum (a diagnostic of the identity; uniformity and inventory are the criteria below)
    w̄ = Array(interior(model.tracer_time_step_splitting.velocities.w))
    Nz = size(model.grid, 3)
    @test maximum(abs, w̄[:, :, Nz+1]) < 1e-10 * maximum(abs, w̄)

    @test maximum_uniform_deviation(model.tracers.constant, 1, active) < 1e-12
    @test abs(tracer_inventory(model.tracers.c) - inventory₀) / inventory₀ < 1e-12

    return nothing
end

@testset "Split tracer time stepping z-star conservation" begin
    for arch in archs
        for grid in zstar_basin_grids(arch), ratio in (1, 4, 8)
            @info "  Testing split tracer z-star conservation on $(summary(grid)) with ratio $ratio [$(typeof(arch))]..."
            model = baroclinic_adjustment_model(grid; ratio)
            test_split_zstar_conservation(model, ratio, 2minutes, 2)
        end

        for fold_topology in (RightCenterFolded, RightFaceFolded)
            @info "  Testing split tracer z-star conservation on a $fold_topology TripolarGrid [$(typeof(arch))]..."
            ratio = 4
            model = random_flow_model(tripolar_grid(arch, fold_topology); ratio)
            test_split_zstar_conservation(model, ratio, 2minutes, 2)
        end
    end
end
