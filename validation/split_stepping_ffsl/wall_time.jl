# Indicative CPU wall time per simulated day of split tracer stepping with WENO5 or FFSL slow tracers (gate C).
#
# The non-divergent gyre of `conservation_and_error.jl` with `extra` additional slow passive tracers,
# for the unsplit WENO5 model and the split models at the maximum stable N of each slow scheme reported by
# `conservation_and_error.jl gyre`. The first long step is excluded (compilation).
#
# Usage: julia --project validation/split_stepping_ffsl/wall_time.jl [maximum stable N for WENO5] [maximum stable N for FFSL]

include(joinpath(@__DIR__, "common.jl"))

weno_ratio = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 8
ffsl_ratio = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 32

function timed_model(grid; ratio, slow, extra)
    extra_names = Tuple(Symbol(:t, n) for n in 1:extra)
    slow_names = (:c, :constant, :smooth, extra_names...)
    advection = merge(tracer_advection(slow, :WENO), NamedTuple{extra_names}(Tuple(slow_advection(slow) for _ in extra_names)))

    model = HydrostaticFreeSurfaceModel(grid; free_surface = SplitExplicitFreeSurface(grid; substeps = 8),
                                        momentum_advection = nothing,
                                        tracer_advection = advection,
                                        vertical_coordinate = ZStarCoordinate(),
                                        timestepper = :SplitRungeKutta3,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :fast, slow_names...),
                                        tracer_time_step_splitting = split_configuration(ratio, slow_names))

    # The boundary currents scale with the jump of ψ across one cell, so the speed scales with the resolution
    u, v = streamfunction_gyre_velocities(grid, 0.35 * 16 / size(grid, 1))
    set!(model, u = u, v = v, b = 0, fast = 1, c = (x, y, z) -> rand(), constant = 1, smooth = 1)
    foreach(name -> set!(model.tracers[name], (x, y, z) -> rand()), extra_names)

    return model
end

function wall_time_per_day(grid, ratio, slow, extra; Δt = 3minutes, cycles = 20)
    model = timed_model(grid; ratio, slow, extra)
    N = isnothing(ratio) ? 32 : ratio
    for step in 1:N
        time_step!(model, Δt)
    end
    steps = cycles * N
    elapsed = @elapsed for step in 1:steps
        time_step!(model, Δt)
    end
    return elapsed / (steps * Δt) * day
end

println()
println("## Indicative CPU wall time per simulated day (non-divergent gyre, threads = ", Threads.nthreads(), ")")

# The finer grid uses a proportionally smaller Δt and a slower gyre, so that the Courant numbers of the boundary currents
# (and the stable N) are unchanged
for (label, grid, Δt) in (("16×8×6", rectilinear_basin(), 3minutes), ("64×32×24", rectilinear_basin(size = (64, 32, 24)), 45))
    for extra in (0, 20)
        println()
        println("### ", label, " grid, Δt = ", prettytime(Δt), ", ", 3 + extra, " slow tracers (+ 2 fast tracers)")
        println()
        println("| configuration | wall time per simulated day (s) | speed-up vs unsplit |")
        println("|---|---:|---:|")
        unsplit = wall_time_per_day(grid, nothing, :WENO, extra; Δt)
        @printf("| unsplit WENO5 | %.2f | 1.00 |\n", unsplit)
        for (slow, ratio) in ((:WENO, weno_ratio), (:FFSL, ffsl_ratio))
            t = wall_time_per_day(grid, ratio, slow, extra; Δt)
            @printf("| split %s, N = %d | %.2f | %.2f |\n", slow == :WENO ? "WENO5" : "FFSL", ratio, t, unsplit / t)
        end
        flush(stdout)
    end
end
