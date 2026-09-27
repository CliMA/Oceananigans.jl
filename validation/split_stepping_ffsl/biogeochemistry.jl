# Split tracer stepping with FluxFormSemiLagrangian slow NPZD tracers and Strang sub-cycled sources (gate C, BGC).
#
# The non-divergent z-star gyre of `conservation_and_error.jl gyre` (steady flow around the immersed ridge, so that the
# long-step Courant number is sustained while the net horizontal outflow of every cell stays small) carries the
# MinimalNPZD tracers (with sinking detritus) as the slow group, advected horizontally with FFSL over the long step,
# with M = 4 source sub-steps per half step. Phytoplankton starts sharp and low (1e-3) on one side of the basin. The script reports, maximized over long steps,
# the drift of total nitrogen ∫σ(N + P + Z + D) dV and the most negative value of any NPZD tracer, for two
# vertical schemes (both adaptive implicit): first-order upwind and WENO5.
#
# Usage: julia --project validation/split_stepping_ffsl/biogeochemistry.jl

include(joinpath(@__DIR__, "common.jl"))

function npzd_ffsl_model(grid; ratio, vertical_scheme, sinking_speed = 100 / day, biogeochemistry_substeps = 4)
    biogeochemistry = MinimalNPZD(grid; sinking_speed)
    ffsl = FluxFormSemiLagrangian(; vertical_scheme)
    splitting = TracerTimeStepSplitting(tracers = (:N, :P, :Z, :D), ratio = ratio, biogeochemistry_substeps = biogeochemistry_substeps)

    model = HydrostaticFreeSurfaceModel(grid; biogeochemistry,
                                        tracer_advection = (b = WENO(order=5), N = ffsl, P = ffsl, Z = ffsl, D = ffsl),
                                        vertical_coordinate = ZStarCoordinate(),
                                        momentum_advection = nothing,
                                        free_surface = SplitExplicitFreeSurface(grid; substeps = 8),
                                        timestepper = :SplitRungeKutta3,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :N, :P, :Z, :D),
                                        tracer_time_step_splitting = splitting)

    x = first(nodes(grid, Center(), Center(), Center()))
    x₀ = (minimum(x) + maximum(x)) / 2
    u, v = streamfunction_gyre_velocities(grid, 0.35)
    Random.seed!(1234)
    set!(model, u = u, v = v, b = 0, η = (x, y, z) -> exp(- (x - x₀ / 2)^2 / (x₀ / 2)^2) / 10,
         N = (x, y, z) -> 4 + rand(),
         P = (x, y, z) -> x < x₀ ? 0.5 + 0.1 * rand() : 1e-3,
         Z = (x, y, z) -> 0.2 + 0.05 * rand(),
         D = (x, y, z) -> 0.3 + 0.1 * rand())

    return model
end

function run_npzd(grid, ratio, vertical_scheme, Δt, steps)
    model = npzd_ffsl_model(grid; ratio, vertical_scheme)
    active = active_cells(model.grid)
    nitrogen₀ = total_nitrogen(model)
    nitrogen_drift = 0.0
    minimum_value = Inf
    swept_courant = 0.0
    sinking_courant = 0.0
    outflow_courant = 0.0

    for step in 1:steps
        time_step!(model, Δt)
        if step % ratio == 0
            nitrogen_drift = max(nitrogen_drift, abs(total_nitrogen(model) - nitrogen₀) / nitrogen₀)
            for name in (:N, :P, :Z, :D)
                minimum_value = min(minimum_value, minimum(Array(interior(model.tracers[name]))[active]))
            end
            swept_courant = max(swept_courant, maximum(swept_courant_numbers(model)))
            outflow_courant = max(outflow_courant, horizontal_outflow_courant_number(model, ratio * Δt))
            _, vertical = long_step_courant_numbers(model, ratio * Δt; drift = model.biogeochemistry.sinking_velocity.w)
            sinking_courant = max(sinking_courant, vertical)
        end
    end

    return (; nitrogen_drift, minimum_value, swept_courant, sinking_courant, outflow_courant)
end

function main()
    grid = rectilinear_basin()
    Δt = 3minutes
    steps = 640

    println()
    println("## MinimalNPZD slow group with FFSL horizontal advection and Strang sub-cycled sources (M = 4)")
    println()
    println("Rectilinear basin 16×8×6 with an immersed ridge, z-star, non-divergent gyre, Δt = ", prettytime(Δt), ", ", steps, " steps, sinking 100 m/day")
    println()
    println("| vertical scheme (adaptive implicit, cfl 0.5) |  N | C swept (h) | C (v, with sinking) | D | total N drift | min(N, P, Z, D) |")
    println("|---|---:|---:|---:|---:|---:|---:|")

    for (label, vertical_scheme) in (("UpwindBiased(order=1)", () -> UpwindBiased(order=1, time_discretization = adaptive_implicit())),
                                     ("WENO(order=5)", () -> WENO(order=5, time_discretization = adaptive_implicit())))
        for ratio in (1, 4, 16, 32, 64)
            result = run_npzd(grid, ratio, vertical_scheme(), Δt, steps)
            @printf("| %s | %2d | %.2f | %.2f | %.2f | %.2e | %.3e |\n", label, ratio, result.swept_courant, result.sinking_courant, result.outflow_courant,
                    result.nitrogen_drift, result.minimum_value)
            flush(stdout)
        end
    end
    return nothing
end

abspath(PROGRAM_FILE) == @__FILE__() && main()
