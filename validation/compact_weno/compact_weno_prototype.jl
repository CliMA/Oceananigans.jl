# Phase-1 prototype gates for CompactWENO (CRWENO-type) vertical reconstruction.
# Design: docs/plans/2026-06-05-compact-vertical-advection-design.md
# All reconstruction methods live in compact_weno_methods.jl (the porting reference).
#
# Validates, standalone (stdlib only):
#   1. nonuniform candidate coefficients and optimal weights via per-face linear solves
#   2. reconstruction convergence on uniform / exponentially-mapped / fixed-ratio grids,
#      with uniform-grid smoothness indicators
#   3. optimal-weight positivity vs stretching ratio
#   4. non-oscillatory step advection vs explicit WENO5
#   5. tridiagonal conditioning with nonlinear weights

include(joinpath(@__DIR__, "compact_weno_methods.jl"))

using Printf

#####
##### Test profile (closed-form antiderivative → exact cell averages)
#####

profile(z) = sin(2π * z) + 0.4 * cos(6π * z + 0.7)
profile_antiderivative(z) = -cos(2π * z) / 2π + 0.4 * sin(6π * z + 0.7) / 6π

exact_cell_averages(zᶠ) =
    [(profile_antiderivative(zᶠ[j+1]) - profile_antiderivative(zᶠ[j])) / (zᶠ[j+1] - zᶠ[j])
     for j in 1:length(zᶠ)-1]

#####
##### Study 1: uniform-grid sanity
#####

function uniform_sanity()
    println("== 1. Uniform-grid sanity: numerical solves vs analytic CRWENO5 ==")
    table = stretched_coefficient_table(uniform_faces(16))
    i = 8
    expected_a = (2/3, 1/3, 2/3)
    expected_γ = (1/6, 5/6, 1/6)
    expected_optimal = (1/5, 1/2, 3/10)
    ok = all(abs.(table.a[i] .- expected_a) .< 1e-12) &&
         all(abs.(table.γ[i] .- expected_γ) .< 1e-12) &&
         all(abs.(table.optimal[i] .- expected_optimal) .< 1e-12)
    @printf("   a       = (%.10f, %.10f, %.10f)  expected (2/3, 1/3, 2/3)\n", table.a[i]...)
    @printf("   γ       = (%.10f, %.10f, %.10f)  expected (1/6, 5/6, 1/6)\n", table.γ[i]...)
    @printf("   optimal = (%.10f, %.10f, %.10f)  expected (1/5, 1/2, 3/10)\n", table.optimal[i]...)

    # the right-biased tables must mirror the left-biased ones on a uniform grid
    mirror = stretched_coefficient_table(uniform_faces(16); bias=:right)
    mirror_ok = all(abs.(mirror.a[i] .- expected_a) .< 1e-12) &&
                all(abs.(mirror.γ[i] .- expected_γ) .< 1e-12) &&
                all(abs.(mirror.optimal[i] .- expected_optimal) .< 1e-12)
    println("   right-biased mirror: ", mirror_ok ? "PASS" : "FAIL")

    println("   ", ok && mirror_ok ? "PASS" : "FAIL")
    return ok && mirror_ok
end

#####
##### Study 2: optimal-weight positivity vs stretching
#####

function positivity_study()
    println("\n== 2. Optimal-weight positivity ==")
    println("   geometric grids, N = 64:")
    first_negative_ratio = Inf
    for ratio in 1.0:0.05:1.5
        table = stretched_coefficient_table(geometric_faces(64, ratio))
        smallest = minimum(minimum(table.optimal[i]) for i in 4:63)
        ratio < first_negative_ratio && smallest < 0 && (first_negative_ratio = ratio)
        @printf("      r = %.2f   minimum optimal weight = %+.4f\n", ratio, smallest)
    end
    println("   exponential mappings, N = 64:")
    for total_ratio in (5, 10, 20, 50)
        table = stretched_coefficient_table(exponential_faces(64, total_ratio))
        smallest = minimum(minimum(table.optimal[i]) for i in 4:63)
        @printf("      Δmax/Δmin = %3d   minimum optimal weight = %+.4f\n", total_ratio, smallest)
    end
    return first_negative_ratio
end

#####
##### Study 3: reconstruction convergence
#####

const VARIANT_LABELS = ("CW-str-nl", "CW-str-opt", "CW-unif-nl", "WENO5-nl", "WENO5-opt")

function convergence_study(label, face_generator; resolutions=(16, 32, 64, 128, 256, 512))
    println("\n   -- $label --")
    errors = [Float64[] for _ in 1:5]

    for N in resolutions
        zᶠ = face_generator(N)
        c̄ = exact_cell_averages(zᶠ)
        closure(i) = profile(zᶠ[i])
        stretched = stretched_coefficient_table(zᶠ)
        uniform = uniform_coefficient_table(N)

        ĉs = (compact_face_values(c̄, stretched; weights=:nonlinear, closure_value=closure),
              compact_face_values(c̄, stretched; weights=:linear,    closure_value=closure),
              compact_face_values(c̄, uniform;   weights=:nonlinear, closure_value=closure),
              explicit_weno_face_values(c̄; weights=:nonlinear, closure_value=closure),
              explicit_weno_face_values(c̄; weights=:linear,    closure_value=closure))

        for (v, ĉ) in enumerate(ĉs)
            push!(errors[v], maximum(abs(ĉ[i] - profile(zᶠ[i])) for i in 4:N-1))
        end
    end

    @printf("   %6s", "N")
    foreach(l -> @printf("   %10s (ord)", l), VARIANT_LABELS)
    println()
    for (n, N) in enumerate(resolutions)
        @printf("   %6d", N)
        for v in 1:5
            order = n == 1 ? NaN : log2(errors[v][n-1] / errors[v][n])
            @printf("   %10.3e (%4.2f)", errors[v][n], order)
        end
        println()
    end

    finest_pair_order(v) = log2(errors[v][end-1] / errors[v][end])
    return finest_pair_order(1), errors
end

#####
##### Study 4: step advection (SSPRK3, w = 1, inflow = 1)
#####

function advect_step(zᶠ; scheme, stop_time=0.4, cfl=0.3)
    Δ = diff(zᶠ)
    N = length(Δ)
    c̄₀ = [clamp((0.3 - zᶠ[j]) / Δ[j], 0.0, 1.0) for j in 1:N]
    table = scheme == :compact ? stretched_coefficient_table(zᶠ) : nothing

    function face_values(c)
        closure(i) = i == 1 ? 1.0 : c[min(i - 1, N)]
        scheme == :compact ?
            compact_face_values(c, table; weights=:nonlinear, closure_value=closure) :
            explicit_weno_face_values(c; weights=:nonlinear, closure_value=closure)
    end

    c̄ = advect_column(c̄₀, Δ, face_values; stop_time, cfl)
    return maximum(c̄) - 1.0, -minimum(c̄)
end

function step_advection_study()
    println("\n== 4. Step advection: overshoot / undershoot at t = 0.4 (N = 128) ==")
    results = Dict()
    for (grid_label, zᶠ) in (("uniform", uniform_faces(128)),
                             ("exponential 10:1", exponential_faces(128, 10)))
        for scheme in (:compact, :explicit)
            overshoot, undershoot = advect_step(zᶠ; scheme)
            results[(grid_label, scheme)] = max(overshoot, undershoot)
            @printf("   %-18s %-9s   overshoot = %+.3e   undershoot = %+.3e\n",
                    grid_label, scheme, overshoot, undershoot)
        end
    end
    return results
end

#####
##### Study 5: conditioning of the nonlinear tridiagonal system
#####

function conditioning_study()
    println("\n== 5. Conditioning (∞-norm) of the nonlinear compact system ==")
    worst = 0.0
    for (label, zᶠ) in (("uniform, N=128, step", uniform_faces(128)),
                        ("exponential 10:1, N=128, step", exponential_faces(128, 10)),
                        ("geometric r=1.2, N=64, step", geometric_faces(64, 1.2)))
        Δ = diff(zᶠ)
        N = length(Δ)
        c̄ = [clamp((0.3 - zᶠ[j]) / Δ[j], 0.0, 1.0) for j in 1:N]
        table = stretched_coefficient_table(zᶠ)
        T, _ = assemble_compact_system(c̄, table; weights=:nonlinear, closure_value=i -> 0.0)
        κ = cond(Matrix(T), Inf)
        worst = max(worst, κ)
        @printf("   %-32s cond = %.2e\n", label, κ)

        rough = [sin(37.0 * j) + 0.3 * sin(101.0 * j) for j in 1:N]
        T, _ = assemble_compact_system(rough, table; weights=:nonlinear, closure_value=i -> 0.0)
        κ = cond(Matrix(T), Inf)
        worst = max(worst, κ)
        @printf("   %-32s cond = %.2e\n", label * " (rough)", κ)
    end
    return worst
end

#####
##### Run all studies and evaluate the Phase-1 gate
#####

sanity_ok = uniform_sanity()

first_negative_ratio = positivity_study()

println("\n== 3. Reconstruction convergence (max error, interior faces) ==")
convergence_study("uniform", uniform_faces)
expo5_order,  _ = convergence_study("exponential 5:1",  N -> exponential_faces(N, 5))
expo10_order, expo10_errors = convergence_study("exponential 10:1", N -> exponential_faces(N, 10))
expo20_order, expo20_errors = convergence_study("exponential 20:1", N -> exponential_faces(N, 20))

# Fixed per-cell ratio is NOT a convergence setting: Δmax → (1 - 1/r) L as N grows,
# so no scheme can converge in max norm. Shown only to verify all variants saturate
# identically (the saturation is the grid's, not the scheme's).
convergence_study("geometric r=1.1 (fixed-ratio; saturates by construction)",
                  N -> geometric_faces(N, 1.1);
                  resolutions=(16, 32, 64, 128, 256))

# Cost of uniform-grid smoothness indicators: nonlinear vs forced-optimal weights,
# stretched candidates in both, at the finest resolution on the harshest mapping.
uniform_smoothness_cost = expo20_errors[1][end] / expo20_errors[2][end]
practice_improvement = expo10_errors[4][end] / expo10_errors[1][end]

step_results = step_advection_study()

worst_condition_number = conditioning_study()

println("\n== Phase-1 gate ==")
gates = (
    ("uniform coefficients reproduce analytic CRWENO5", sanity_ok),
    ("optimal weights positive for geometric r ≤ 1.2", first_negative_ratio > 1.2),
    ("observed order ≥ 4 on exponential 10:1", expo10_order ≥ 4),
    ("observed order ≥ 4 on exponential 20:1", expo20_order ≥ 4),
    ("uniform-β cost ≤ 4× vs optimal weights (exponential 20:1, finest N)",
     uniform_smoothness_cost ≤ 4),
    ("≥ 10× error reduction vs uniform-coefficient explicit WENO5 (exponential 10:1, finest N)",
     practice_improvement ≥ 10),
    ("step overshoot/undershoot < 1e-2 on both grids",
     all(step_results[(g, :compact)] < 1e-2 for g in ("uniform", "exponential 10:1"))),
    ("condition number < 1e4", worst_condition_number < 1e4),
)
for (description, passed) in gates
    @printf("   [%s] %s\n", passed ? "PASS" : "FAIL", description)
end
@printf("\n   uniform-β cost (nonlinear / optimal-weight error): %.1f×\n", uniform_smoothness_cost)
@printf("   improvement vs current-practice explicit WENO5:     %.0f×\n", practice_improvement)
println(all(last.(gates)) ? "\nALL GATES PASS" : "\nGATE FAILURES — revisit design")
