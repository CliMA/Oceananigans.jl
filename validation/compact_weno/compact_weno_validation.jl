# CompactWENO validation: vertical advection of a wave packet on a stretched grid,
# CompactWENO vs explicit WENO5 (uniform coefficients — current Oceananigans practice).
#
# Part A (runs today) uses the reference methods in compact_weno_methods.jl:
#   - convergence table on an exponentially stretched column
#   - final-profile comparison figure  → compact_weno_final_profiles.png
#   - moving-solution animation        → compact_weno_advection.mp4
#
# Part B (runs once CompactWENO exists in Oceananigans) repeats the experiment with
# HydrostaticFreeSurfaceModel + PrescribedVelocityFields and cross-checks the model
# solution against the reference implementation on the identical grid. Flip:
validate_with_oceananigans = true

include(joinpath(@__DIR__, "compact_weno_methods.jl"))

using Printf
using CairoMakie

#####
##### Experiment: wave packet advected upward through the coarsening region
#####

const STOP_TIME = 0.45
const TOTAL_RATIO = 1.1          # Δmax/Δmin of the exponential mapping

# carrier wavelength 0.1 → ~4.6 points per wavelength in the coarse region at N = 96
wave_packet(z) = exp(-((z - 0.22) / 0.06)^2) * sin(2π * (z - 0.22) / 0.1)
exact_solution(z, t) = wave_packet(z - t)

function reference_solutions(zᶠ; stop_time=STOP_TIME, snapshots=nothing)
    Δ = diff(zᶠ)
    N = length(Δ)
    c̄₀ = quadrature_cell_averages(wave_packet, zᶠ)
    table = stretched_coefficient_table(zᶠ)

    closure(c) = i -> i == 1 ? 0.0 : c[min(i - 1, N)]
    compact_faces(c) = compact_face_values(c, table; weights=:nonlinear, closure_value=closure(c))
    weno_faces(c) = explicit_weno_face_values(c; weights=:nonlinear, closure_value=closure(c))

    compact_snapshots = snapshots === nothing ? nothing : snapshots.compact
    weno_snapshots = snapshots === nothing ? nothing : snapshots.weno

    compact = advect_column(c̄₀, Δ, compact_faces; stop_time, snapshots=compact_snapshots)
    weno = advect_column(c̄₀, Δ, weno_faces; stop_time, snapshots=weno_snapshots)
    return compact, weno
end

solution_error(c̄, zᶠ, t) = maximum(abs.(c̄ .- quadrature_cell_averages(z -> exact_solution(z, t), zᶠ)))

#####
##### Part A.1: convergence on the stretched column
#####

println("== Wave-packet convergence, exponential $(TOTAL_RATIO):1 column, t = $STOP_TIME ==")
resolutions = (64, 96, 128, 192, 256, 384)
compact_errors = Float64[]
weno_errors = Float64[]

for N in resolutions
    zᶠ = exponential_faces(N, TOTAL_RATIO)
    compact, weno = reference_solutions(zᶠ)
    push!(compact_errors, solution_error(compact, zᶠ, STOP_TIME))
    push!(weno_errors, solution_error(weno, zᶠ, STOP_TIME))
end

@printf("   %6s   %11s (ord)   %11s (ord)   %8s\n", "N", "CompactWENO", "WENO5-expl", "ratio")
for (n, N) in enumerate(resolutions)
    order(errors) = n == 1 ? NaN : log(errors[n-1] / errors[n]) / log(resolutions[n] / resolutions[n-1])
    @printf("   %6d   %11.3e (%4.2f)   %11.3e (%4.2f)   %7.1f×\n",
            N, compact_errors[n], order(compact_errors),
            weno_errors[n], order(weno_errors),
            weno_errors[n] / compact_errors[n])
end

#####
##### Part A.2: final-profile figure and animation
#####

# On the 20:1 mapping: N ≤ 96 destroys the packet for both schemes; 128 is the
# clearest contrast (compact 0.68 vs WENO 0.46, exact 0.81); ≥ 192 both ≈ exact
N = 128
zᶠ = exponential_faces(N, TOTAL_RATIO)
zᶜ = (zᶠ[1:end-1] .+ zᶠ[2:end]) ./ 2
snapshots = (compact = [], weno = [])
compact, weno = reference_solutions(zᶠ; snapshots)

exact_averages = quadrature_cell_averages(z -> exact_solution(z, STOP_TIME), zᶠ)
zdense = range(0, 1, length=2000)

@printf("\n   N = %d amplitude retention:  exact %.3f   CompactWENO %.3f   WENO5 %.3f\n",
        N, maximum(abs.(exact_averages)), maximum(abs.(compact)), maximum(abs.(weno)))

fig = Figure(size=(950, 700))
ax1 = Axis(fig[1, 1]; xlabel="z", ylabel="c",
           title="Wave packet at t = $STOP_TIME, exponential $(TOTAL_RATIO):1 grid, N = $N")
lines!(ax1, zdense, exact_solution.(zdense, STOP_TIME); color=:gray, label="exact")
scatterlines!(ax1, zᶜ, compact; markersize=5, label="CompactWENO")
scatterlines!(ax1, zᶜ, weno; markersize=5, label="WENO5 (uniform coefficients)")
axislegend(ax1; position=:lt)

ax2 = Axis(fig[2, 1]; xlabel="z", ylabel="|error|", yscale=log10)
lines!(ax2, zᶜ, max.(abs.(compact .- exact_averages), 1e-16); label="CompactWENO")
lines!(ax2, zᶜ, max.(abs.(weno .- exact_averages), 1e-16); label="WENO5 (uniform coefficients)")
axislegend(ax2; position=:lt)

figure_path = joinpath(@__DIR__, "compact_weno_final_profiles.png")
save(figure_path, fig)
println("   saved $figure_path")

frame = Observable(1)
time_title = @lift @sprintf("t = %.3f", snapshots.compact[$frame][1])
compact_frame = @lift snapshots.compact[$frame][2]
weno_frame = @lift snapshots.weno[$frame][2]
exact_frame = @lift exact_solution.(zdense, snapshots.compact[$frame][1])

animation = Figure(size=(950, 450))
ax = Axis(animation[1, 1]; xlabel="z", ylabel="c", title=time_title)
ylims!(ax, -1.2, 1.2)
lines!(ax, zdense, exact_frame; color=:gray, label="exact")
scatterlines!(ax, zᶜ, compact_frame; markersize=5, label="CompactWENO")
scatterlines!(ax, zᶜ, weno_frame; markersize=5, label="WENO5 (uniform coefficients)")
axislegend(ax; position=:lt)

animation_path = joinpath(@__DIR__, "compact_weno_advection.mp4")
record(animation, animation_path, 1:4:length(snapshots.compact); framerate=24) do n
    frame[] = n
end
println("   saved $animation_path")

#####
##### Part B: the same experiment through Oceananigans (needs CompactWENO implemented)
#####

if validate_with_oceananigans
    using Oceananigans

    function oceananigans_column_solution(scheme, zᶠ; stop_time=STOP_TIME, cfl=0.6)
        Nz = length(zᶠ) - 1
        grid = RectilinearGrid(size=Nz, z=zᶠ, topology=(Flat, Flat, Bounded), halo = 7)
        model = HydrostaticFreeSurfaceModel(grid;
                                            velocities = PrescribedVelocityFields(w=1),
                                            tracer_advection = scheme,
                                            tracers = :c,
                                            buoyancy = nothing,
                                            timestepper = :SplitRungeKutta3)

        initial_averages = quadrature_cell_averages(wave_packet, zᶠ)
        set!(model, c=reshape(initial_averages, 1, 1, Nz))

        simulation = Simulation(model; Δt=cfl * minimum_zspacing(grid), stop_time)
        run!(simulation)
        return model # Array(interior(model.tracers.c, 1, 1, :))
    end

    function oceananigans_validation(model_resolutions=(64, 96, 128, 192))
        println("\n== Oceananigans column advection, exponential $(TOTAL_RATIO):1, t = $STOP_TIME ==")
        schemes = (CompactWENO(), WENO(order=5))
        scheme_labels = ("CompactWENO", "WENO(order=5)")
        model_errors = [Float64[] for _ in schemes]

        for Nz in model_resolutions
            zᶠ = exponential_faces(Nz, TOTAL_RATIO)
            for (s, scheme) in enumerate(schemes)
                c = oceananigans_column_solution(scheme, zᶠ)
                push!(model_errors[s], solution_error(c, zᶠ, STOP_TIME))
            end
        end

        @printf("   %6s   %14s (ord)   %14s (ord)\n", "N", scheme_labels...)
        for (n, Nz) in enumerate(model_resolutions)
            @printf("   %6d", Nz)
            for s in 1:2
                order = n == 1 ? NaN :
                        log(model_errors[s][n-1] / model_errors[s][n]) /
                        log(model_resolutions[n] / model_resolutions[n-1])
                @printf("   %14.3e (%4.2f)", model_errors[s][n], order)
            end
            println()
        end

        # Cross-check: the model's CompactWENO against the reference implementation on
        # the same grid. Time discretizations differ slightly (SplitRungeKutta3 vs
        # SSPRK3), so agreement is expected to ~the temporal error, not machine precision.
        Nz = 128
        zᶠ = exponential_faces(Nz, TOTAL_RATIO)
        reference_compact, _ = reference_solutions(zᶠ)
        sol = oceananigans_column_solution(CompactWENO(), zᶠ)
        discrepancy = maximum(abs.(sol .- reference_compact))
        reference_error = solution_error(reference_compact, zᶠ, STOP_TIME)
        @printf("\n   model vs reference (N = %d): max discrepancy = %.3e (reference error %.3e)\n",
                Nz, discrepancy, reference_error)
        println(discrepancy < 10 * reference_error ?
                "   CROSS-CHECK PASS (discrepancy within the scheme's own error)" :
                "   CROSS-CHECK FAIL — implementation disagrees with the reference")
    end

    oceananigans_validation()
else
    println("\n(Set validate_with_oceananigans = true to run the Oceananigans comparison ",
            "once CompactWENO is implemented.)")
end
