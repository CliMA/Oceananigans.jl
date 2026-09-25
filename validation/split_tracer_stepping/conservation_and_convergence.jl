# Split tracer time stepping: conservation and error against `ratio = 1` (gates A1, A2, A5).
#
# For every grid and every ratio N this script reports, at the final time (a multiple of every N):
#   * the deviation of an initially uniform slow tracer, relative, over active cells;
#   * the inventory drift ∫c dV of a random slow tracer (∫σc dV under z-star), relative;
#   * the L² and L∞ errors of a smooth slow tracer against the run with `ratio = 1`;
#   * the maximum horizontal and vertical Courant numbers of the long step, N Δt |ū| / Δx and N Δt |w̄| / Δz.
#
# Usage: julia --project validation/split_tracer_stepping/conservation_and_convergence.jl [static|zstar|tripolar|all]

using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid, RightCenterFolded, RightFaceFolded

include(joinpath(@__DIR__, "..", "..", "test", "setup", "split_tracer_stepping_test_utils.jl"))

ratios = (1, 2, 4, 8, 16)

function run_split_model(build_model, grid, ratio, Δt, steps)
    model = build_model(deepcopy(grid), ratio)
    grid = model.grid
    active = active_cells(grid)
    inventory₀ = tracer_inventory(model.tracers.c)

    maximum_horizontal_courant = 0.0
    maximum_vertical_courant = 0.0

    for step in 1:steps
        time_step!(model, Δt)
        if step % ratio == 0 && !isnothing(model.tracer_time_step_splitting)
            horizontal, vertical = long_step_courant_numbers(model, ratio * Δt)
            maximum_horizontal_courant = max(maximum_horizontal_courant, horizontal)
            maximum_vertical_courant = max(maximum_vertical_courant, vertical)
        end
    end

    η = Array(interior(model.free_surface.displacement))
    free_surface_range = maximum(η) - minimum(η)
    uniform_deviation = maximum_uniform_deviation(model.tracers.constant, 1, active)
    inventory_drift = abs(tracer_inventory(model.tracers.c) - inventory₀) / abs(inventory₀)

    return (; model, active, free_surface_range, uniform_deviation, inventory_drift, maximum_horizontal_courant, maximum_vertical_courant)
end

function convergence_table(name, build_model, grid, Δt, steps)
    println()
    println("### ", name, " (Δt = ", prettytime(Δt), ", ", steps, " steps)")
    println()
    println("|  N | max tracer C (h) | max tracer C (v) | η range (m) | uniform deviation | inventory drift | L² error vs N=1 | L∞ error vs N=1 |")
    println("|---:|---:|---:|---:|---:|---:|---:|---:|")

    reference = nothing
    for ratio in ratios
        result = run_split_model(build_model, grid, ratio, Δt, steps)
        if ratio == 1
            reference = result
            L², L∞ = 0.0, 0.0
        else
            L², L∞ = relative_errors(result.model.tracers.smooth, reference.model.tracers.smooth, result.active)
        end
        @printf("| %2d | %.3f | %.3f | %.2e | %.2e | %.2e | %.2e | %.2e |\n", ratio,
                result.maximum_horizontal_courant, result.maximum_vertical_courant, result.free_surface_range,
                result.uniform_deviation, result.inventory_drift, L², L∞)
    end

    # The unsplit model, for comparison with the split model at N = 1
    unsplit = run_split_model((grid, ratio) -> build_model(grid, nothing), grid, 1, Δt, steps)
    L², L∞ = relative_errors(unsplit.model.tracers.smooth, reference.model.tracers.smooth, unsplit.active)
    @printf("| unsplit | - | - | %.2e | %.2e | %.2e | %.2e | %.2e |\n",
            unsplit.free_surface_range, unsplit.uniform_deviation, unsplit.inventory_drift, L², L∞)

    return nothing
end

ridge(x₀, width, height, depth) = (x, y) -> - depth + height * exp(- ((x - x₀) / width)^2)

function static_z_grids()
    rectilinear = RectilinearGrid(size = (16, 8, 6), halo = (4, 4, 4), x = (0, 32kilometers), y = (0, 16kilometers), z = (-60, 0),
                                  topology = (Bounded, Bounded, Bounded))
    rectilinear_ridge = ImmersedBoundaryGrid(rectilinear, GridFittedBottom(ridge(16kilometers, 4kilometers, 30, 60)))

    latitude_longitude = LatitudeLongitudeGrid(size = (16, 8, 6), halo = (4, 4, 4), longitude = (0, 0.32), latitude = (30, 30.16), z = (-60, 0))
    latitude_longitude_ridge = ImmersedBoundaryGrid(latitude_longitude, GridFittedBottom(ridge(0.16, 0.04, 30, 60)))

    return ("Rectilinear basin with an immersed ridge, static z" => rectilinear_ridge,
            "LatitudeLongitude basin with an immersed ridge, static z" => latitude_longitude_ridge)
end

function zstar_grids()
    z = MutableVerticalDiscretization(collect(range(-60, 0, length=7)))
    rectilinear = RectilinearGrid(size = (16, 8, 6), halo = (4, 4, 4), x = (0, 32kilometers), y = (0, 16kilometers), z = z,
                                  topology = (Bounded, Bounded, Bounded))
    rectilinear_ridge = ImmersedBoundaryGrid(rectilinear, GridFittedBottom(ridge(16kilometers, 4kilometers, 30, 60)))

    latitude_longitude = LatitudeLongitudeGrid(size = (16, 8, 6), halo = (4, 4, 4), longitude = (0, 0.32), latitude = (30, 30.16), z = z)
    latitude_longitude_ridge = ImmersedBoundaryGrid(latitude_longitude, GridFittedBottom(ridge(0.16, 0.04, 30, 60)))

    return ("Rectilinear basin with an immersed ridge, z-star" => rectilinear_ridge,
            "LatitudeLongitude basin with an immersed ridge, z-star" => latitude_longitude_ridge)
end

function tripolar_grids(z)
    # Gaussian islands, as in test/vertical_coordinate/conservation_tripolar.jl
    mountain(λ, φ, λ₀, φ₀) = exp(- (λ - λ₀)^2 / 2 / 5^2 - (φ - φ₀)^2 / 2 / 5^2)
    islands(λ, φ) = - 20 + 30 * (mountain(λ, φ, 70, 55) + mountain(λ, φ, 250, 55) + mountain(λ, φ, 430, 55))

    return Tuple(string(fold_topology, " TripolarGrid with immersed islands") =>
                 ImmersedBoundaryGrid(TripolarGrid(; size = (20, 32, 5), z, fold_topology), GridFittedBottom(islands))
                 for fold_topology in (RightCenterFolded, RightFaceFolded))
end

mode = isempty(ARGS) ? "all" : ARGS[1]

if mode ∈ ("static", "all")
    println("\n## A1: static z, oscillating overturning circulation (free surface at rest)")
    for (name, grid) in static_z_grids()
        build(grid, ratio) = overturning_model(grid; ratio, speed = 3, period = 30minutes)
        convergence_table(name, build, grid, 1minute, 32)
    end

    println("\n## A1 (supplementary): static z, baroclinic adjustment with a moving free surface")
    println("The unsplit model does not conserve ∫c dV here either: with a static z the tracer flux through the moving surface is not zero.")
    for (name, grid) in static_z_grids()
        build(grid, ratio) = baroclinic_adjustment_model(grid; ratio, vertical_coordinate = ZCoordinate())
        convergence_table(name, build, grid, 2minutes, 32)
    end
end

if mode ∈ ("zstar", "all")
    println("\n## A2: z-star, baroclinic adjustment with a moving free surface")
    for (name, grid) in zstar_grids()
        build(grid, ratio) = baroclinic_adjustment_model(grid; ratio)
        convergence_table(name, build, grid, 2minutes, 32)
    end

    println("\n## A2 (larger time step): z-star, baroclinic adjustment with a stronger front")
    for (name, grid) in zstar_grids()
        build(grid, ratio) = baroclinic_adjustment_model(grid; ratio, buoyancy_contrast = 0.1)
        convergence_table(name, build, grid, 3minutes, 32)
    end
end

if mode ∈ ("tripolar", "all")
    println("\n## A5: TripolarGrid with immersed islands, z-star, random flow across the fold")
    for (name, grid) in tripolar_grids(MutableVerticalDiscretization(collect(-10:2:0)))
        build(grid, ratio) = random_flow_model(grid; ratio)
        convergence_table(name * ", z-star", build, grid, 2minutes, 32)
    end

    println("\n## A5: TripolarGrid with immersed islands, static z, oscillating overturning (at rest near the fold)")
    for (name, grid) in tripolar_grids((-10, 0))
        build(grid, ratio) = overturning_model(grid; ratio, speed = 1, period = 30minutes, northern_rows_at_rest = 4)
        convergence_table(name * ", static z", build, grid, 2minutes, 32)
    end

    println("\n## A5 (supplementary): TripolarGrid with immersed islands, static z, random flow across the fold")
    println("The unsplit model does not conserve ∫c dV here either: with a static z the tracer flux through the moving surface is not zero.")
    for (name, grid) in tripolar_grids((-10, 0))
        build(grid, ratio) = random_flow_model(grid; ratio, vertical_coordinate = ZCoordinate())
        convergence_table(name * ", static z", build, grid, 2minutes, 32)
    end
end
