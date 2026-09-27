# Discriminating runs for split tracer stepping with FluxFormSemiLagrangian slow tracers (gate C): do the failures seen in
# `conservation_and_error.jl` and `biogeochemistry.jl` come from the long step or from the schemes themselves?
#
#   1. Ridge gyre (divergent flow into the immersed ridge): split N = 32 vs an unsplit FFSL model with Δt = 96 minutes
#      (the length of the N = 32 long step), and the thickness left by the horizontal step in the blow-up cells.
#   2. Positivity of a passive step-function tracer (0.5 / 1e-3) in the non-divergent gyre at C ≈ 2.4: split N = 32 vs unsplit.
#   3. NPZD positivity vs the adaptive implicit cfl of the vertical upwind scheme, and an unsplit all-WENO5 NPZD model.
#   4. Swept Courant numbers on the fold face itself (y-face j = Ny + 1) and in the adjacent rows of the TripolarGrids.
#
# Usage: julia --project validation/split_stepping_ffsl/discriminators.jl

include(joinpath(@__DIR__, "biogeochemistry.jl"))

using Oceananigans.Operators: Axᶠᶜᶜ, Ayᶜᶠᶜ

grid = rectilinear_basin()
active = active_cells(grid)

function smooth_overshoot(model, bounds)
    s = Array(interior(model.tracers.smooth))[active]
    lower, upper = bounds
    return max(0, maximum(s) - upper, lower - minimum(s)) / (upper - lower)
end

println()
println("## 1. Ridge gyre: split N = 32 vs unsplit FFSL with Δt = 96 minutes")
println()
println("| run | time (hours) | smooth overshoot | uniform deviation |")
println("|---|---:|---:|---:|")
for (label, ratio, Δt, substeps, steps, every) in (("unsplit FFSL, Δt = 96 min", nothing, 96minutes, 200, 20, 4),
                                                  ("split FFSL, N = 32, Δt = 3 min", 32, 3minutes, 8, 640, 128))
    model = gyre_basin_model(grid; ratio, substeps)
    bounds = extrema(Array(interior(model.tracers.smooth))[active])
    for n in 1:steps
        time_step!(model, Δt)
        n % every == 0 && @printf("| %s | %.1f | %.2e | %.2e |\n", label, n * Δt / hour, smooth_overshoot(model, bounds),
                                  maximum_uniform_deviation(model.tracers.constant, 1, active))
    end
end

model = gyre_basin_model(grid; ratio = 32)
for n in 1:32
    time_step!(model, 3minutes)
end
T = 96minutes
ū, v̄, _ = model.tracer_time_step_splitting.velocities
println()
println("Thickness after the horizontal step of the first long step, relative to the initial one, 1 - T ∇ₕ⋅(Aₕū) / V:")
for (i, j, k) in ((11, 1, 2), (12, 1, 1))
    V = Vᶜᶜᶜ(i, j, k, grid)
    outflow = T * (Axᶠᶜᶜ(i+1, j, k, grid) * ū[i+1, j, k] - Axᶠᶜᶜ(i, j, k, grid) * ū[i, j, k] +
                   Ayᶜᶠᶜ(i, j+1, k, grid) * v̄[i, j+1, k] - Ayᶜᶠᶜ(i, j, k, grid) * v̄[i, j, k]) / V
    @printf("  cell (%d, %d, %d): %.2f (western neighbour active: %s)\n", i, j, k, 1 - outflow, active[i-1, j, k])
end

println()
println("## 2. Passive step-function tracer (0.5 / 1e-3) in the non-divergent gyre, FFSL with vertical upwind (AVID)")
println()
println("| run | swept C | minimum over long steps |")
println("|---|---:|---:|")
for (label, ratio, Δt, substeps, steps) in (("split N = 1", 1, 3minutes, 8, 640), ("split N = 32", 32, 3minutes, 8, 640),
                                            ("split N = 64", 64, 3minutes, 8, 640), ("unsplit, Δt = 96 min", nothing, 96minutes, 200, 20))
    scheme = FluxFormSemiLagrangian(vertical_scheme = UpwindBiased(order=1, time_discretization = adaptive_implicit()))
    splitting = isnothing(ratio) ? nothing : TracerTimeStepSplitting(tracers = :p, ratio = ratio)
    model = HydrostaticFreeSurfaceModel(grid; free_surface = SplitExplicitFreeSurface(grid; substeps), momentum_advection = nothing,
                                        tracer_advection = (b = WENO(order=5), p = scheme), vertical_coordinate = ZStarCoordinate(),
                                        timestepper = :SplitRungeKutta3, buoyancy = BuoyancyTracer(), tracers = (:b, :p),
                                        tracer_time_step_splitting = splitting)
    u, v = streamfunction_gyre_velocities(grid, 0.35)
    set!(model, u = u, v = v, b = 0, p = (x, y, z) -> x < 16kilometers ? 0.5 : 1e-3)
    N = isnothing(ratio) ? 1 : ratio
    lowest = Inf
    for n in 1:steps
        time_step!(model, Δt)
        n % N == 0 && (lowest = min(lowest, minimum(Array(interior(model.tracers.p))[active])))
    end
    @printf("| %s | %.2f | %.3e |\n", label, maximum(swept_courant_numbers(model)), lowest)
end

println()
println("## 3. NPZD positivity vs the adaptive implicit cfl of the vertical upwind scheme (FFSL slow NPZD, M = 4)")
println()
println("| cfl | N | total N drift | min(N, P, Z, D) |")
println("|---:|---:|---:|---:|")
for cfl in (0.5, 0.2, 0.05), ratio in (16, 32, 64)
    vertical_scheme = UpwindBiased(order=1, time_discretization = AdaptiveVerticallyImplicitDiscretization(; cfl))
    result = run_npzd(grid, ratio, vertical_scheme, 3minutes, 640)
    @printf("| %.2f | %d | %.2e | %.3e |\n", cfl, ratio, result.nitrogen_drift, result.minimum_value)
end

biogeochemistry = MinimalNPZD(grid; sinking_speed = 100 / day)
weno = WENO(order=5, time_discretization = adaptive_implicit())
model = HydrostaticFreeSurfaceModel(grid; biogeochemistry, tracer_advection = weno, momentum_advection = nothing,
                                    vertical_coordinate = ZStarCoordinate(), free_surface = SplitExplicitFreeSurface(grid; substeps = 8),
                                    timestepper = :SplitRungeKutta3, buoyancy = BuoyancyTracer(), tracers = (:b, :N, :P, :Z, :D))
u, v = streamfunction_gyre_velocities(grid, 0.35)
Random.seed!(1234)
set!(model, u = u, v = v, b = 0, N = (x, y, z) -> 4 + rand(), P = (x, y, z) -> x < 16kilometers ? 0.5 + 0.1 * rand() : 1e-3,
     Z = (x, y, z) -> 0.2 + 0.05 * rand(), D = (x, y, z) -> 0.3 + 0.1 * rand())
for n in 1:640
    time_step!(model, 3minutes)
end
lowest = minimum(minimum(Array(interior(model.tracers[name]))[active]) for name in (:N, :P, :Z, :D))
println()
@printf("Unsplit NPZD with WENO5 (horizontal and adaptive implicit vertical), no FFSL: min(N, P, Z, D) = %.3e\n", lowest)

println()
println("## 4. Swept Courant numbers at the fold of the TripolarGrids (maximum over long steps, 10 long steps)")
println()
println("| fold topology | N | max C swept | max |sʸ| on the fold face j = Ny + 1 | max |sʸ| on face Ny | max |sʸ| on face Ny - 1 | max |sˣ| on rows Ny - 1, Ny |")
println("|---|---:|---:|---:|---:|---:|---:|")
for fold_topology in (RightCenterFolded, RightFaceFolded), ratio in (32, 64)
    tripolar = tripolar_basin(; fold_topology)
    Nx, Ny, Nz = size(tripolar)
    model = tripolar_rotation_model(tripolar; ratio)
    workspace = flux_form_semi_lagrangian_workspace(model.advection)
    fold_face = face = below = rows = swept = 0.0
    for n in 1:10ratio
        time_step!(model, 3hours)
        if n % ratio == 0
            sʸ = workspace.sʸ
            sˣ = workspace.sˣ
            fold_face = max(fold_face, maximum(abs(sʸ[i, Ny+1, k]) for i in 1:Nx, k in 1:Nz))
            face = max(face, maximum(abs(sʸ[i, Ny, k]) for i in 1:Nx, k in 1:Nz))
            below = max(below, maximum(abs(sʸ[i, Ny-1, k]) for i in 1:Nx, k in 1:Nz))
            rows = max(rows, maximum(abs(sˣ[i, j, k]) for i in 1:Nx, j in Ny-1:Ny, k in 1:Nz))
            swept = max(swept, maximum(swept_courant_numbers(model)))
        end
    end
    @printf("| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f |\n", fold_topology, ratio, swept, fold_face, face, below, rows)
end
