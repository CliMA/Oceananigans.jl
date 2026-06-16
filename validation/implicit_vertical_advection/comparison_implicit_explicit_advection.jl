# AdaptiveImplicitVerticalAdvection (AIVA) validation: stability + correctness.
#
# Stretched 1D column with steady, uniform vertical velocity w = w₀. The tracer
# equation reduces to pure translation c(z, t) = c₀(z − w₀ t).
#
# Four runs at the same final time:
#   1. Explicit UpwindBiased(5), large Δt (CFL ≈ 4 in thin cells) — blows up.
#   2. AdaptiveVerticallyImplicitDiscretization with fixed `maximum_explicit_cfl`, same large Δt.
#   3. AdaptiveVerticallyImplicitDiscretization with `implicit_fraction`, same large Δt.
#   4. Reference: explicit UpwindBiased(5), small Δt (CFL ≈ 0.5).
#
# Compared against the analytical Gaussian translated by w₀·T.

using Printf
using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.TimeSteppers: adaptive_implicit_vertical_advection_diagnostics
using Oceananigans.Grids: minimum_zspacing, zspacings

#####
##### Grid
#####

const Nz = 70
const Lz = 1.0
const refinement = 2.0

ξ(k) = (k - 1) / Nz * 2
zᶠ = [Lz * (tanh(refinement * ξ(k)) / tanh(refinement) - 1) for k in 1:Nz/2+1] / 2 .+ 0.5
zᶠ = [zᶠ..., (0.5 .+ (0.5 .- reverse(zᶠ[1:end-1])))...] .- 1

grid = RectilinearGrid(CPU(); size=Nz, z=zᶠ, topology=(Flat, Flat, Bounded), halo=8)

#####
##### Setup
#####

const w₀ = 1.0
const c₀ = -0.8
const σc = 0.08

initial_tracer(z) = exp(-(z - c₀)^2 / σc^2)
prescribed_velocity(z, t) = w₀

velocities = PrescribedVelocityFields(w=prescribed_velocity)

explicit_scheme = WENO(order=5)
aiva_cfl_scheme = WENO(order=5, time_discretization=AdaptiveVerticallyImplicitDiscretization(maximum_explicit_cfl=1.2))
aiva_fraction_scheme = WENO(order=5, time_discretization=AdaptiveVerticallyImplicitDiscretization(implicit_fraction=0.5))

build_model(advection) = HydrostaticFreeSurfaceModel(grid;
                                                     velocities,
                                                     tracers=:c,
                                                     tracer_advection=advection,
                                                     timestepper=:SplitRungeKutta3,
                                                     buoyancy=nothing,
                                                     closure=nothing)

explicit_model = build_model(explicit_scheme)
aiva_cfl_model = build_model(aiva_cfl_scheme)
aiva_fraction_model = build_model(aiva_fraction_scheme)
reference_model = build_model(explicit_scheme)

set!(explicit_model, c=initial_tracer)
set!(aiva_cfl_model, c=initial_tracer)
set!(aiva_fraction_model, c=initial_tracer)
set!(reference_model, c=initial_tracer)

#####
##### Time-stepping
#####

Δzₘᵢₙ = minimum_zspacing(grid)
Δt = 3.0 * Δzₘᵢₙ / w₀
Δτ = 0.5 * Δzₘᵢₙ / w₀

final_time = (-0.15 - c₀) / w₀
Nₛ = round(Int, final_time / Δt)
Nᵣ = round(Int, final_time / Δτ)
Nsub = round(Int, Δt / Δτ)

@info "Δz_min  = $(Δzₘᵢₙ), w₀ = $(w₀)"
@info "AIVA Δt = $(Δt) (CFL ≈ $(w₀ * Δt / Δzₘᵢₙ)), $Nₛ steps"
@info "Ref  Δτ = $(Δτ) (CFL ≈ $(w₀ * Δτ / Δzₘᵢₙ)), $Nᵣ steps"

#####
##### Containers and initial sampling
#####

zᶜ = collect(znodes(aiva_cfl_model.tracers.c))
Δzᶜ = vec(collect(zspacings(grid, Center(), Center(), Center())))

explicit_history = zeros(Nz, Nₛ + 1)
aiva_cfl_history = zeros(Nz, Nₛ + 1)
aiva_fraction_history = zeros(Nz, Nₛ + 1)
reference_history = zeros(Nz, Nₛ + 1)
aiva_cfl_mass = zeros(Nₛ + 1)
aiva_fraction_mass = zeros(Nₛ + 1)
reference_mass = zeros(Nₛ + 1)

sample!(history, model) = (history .= Array(interior(model.tracers.c))[1, 1, :])

sample!(view(explicit_history, :, 1), explicit_model)
sample!(view(aiva_cfl_history, :, 1), aiva_cfl_model)
sample!(view(aiva_fraction_history, :, 1), aiva_fraction_model)
sample!(view(reference_history, :, 1), reference_model)
aiva_cfl_mass[1] = sum(aiva_cfl_history[:, 1] .* Δzᶜ)
aiva_fraction_mass[1] = sum(aiva_fraction_history[:, 1] .* Δzᶜ)
reference_mass[1] = sum(reference_history[:, 1] .* Δzᶜ)

#####
##### Time-step loop (large Δt for AIVA / explicit, substep for reference)
#####

blowup_step = Ref(-1)
blowup_threshold = 1e3 * maximum(abs, explicit_history[:, 1])

for n in 1:Nₛ
    if blowup_step[] < 0
        try
            time_step!(explicit_model, Δt)
            sample!(view(explicit_history, :, n + 1), explicit_model)
            cn = explicit_history[:, n + 1]
            if !all(isfinite, cn) || maximum(abs, cn) > blowup_threshold
                blowup_step[] = n
            end
        catch
            blowup_step[] = n
            explicit_history[:, n + 1:end] .= NaN
        end
    end

    time_step!(aiva_cfl_model, Δt)
    sample!(view(aiva_cfl_history, :, n + 1), aiva_cfl_model)
    aiva_cfl_mass[n + 1] = sum(aiva_cfl_history[:, n + 1] .* Δzᶜ)

    time_step!(aiva_fraction_model, Δt)
    sample!(view(aiva_fraction_history, :, n + 1), aiva_fraction_model)
    aiva_fraction_mass[n + 1] = sum(aiva_fraction_history[:, n + 1] .* Δzᶜ)

    for _ in 1:Nsub
        time_step!(reference_model, Δτ)
    end
    sample!(view(reference_history, :, n + 1), reference_model)
    reference_mass[n + 1] = sum(reference_history[:, n + 1] .* Δzᶜ)
end

#####
##### Analytical reference and diagnostics
#####

Tₛ = Nₛ * Δt
exact_tracer = [exp(-(zᶜ[k] - c₀ - w₀ * Tₛ)^2 / σc^2) for k in 1:Nz]

function centroid(c, z, dz)
    m = sum(c .* dz)
    return sum(c .* z .* dz) / m, m
end

zᴬᶜ, mᴬᶜ = centroid(aiva_cfl_history[:, end], zᶜ, Δzᶜ)
zᴬᶠ, mᴬᶠ = centroid(aiva_fraction_history[:, end], zᶜ, Δzᶜ)
zᴿ, mᴿ = centroid(reference_history[:, end], zᶜ, Δzᶜ)
zᴱ, mᴱ = centroid(exact_tracer, zᶜ, Δzᶜ)

l²(c1, c2) = sqrt(sum((c1 .- c2).^2 .* Δzᶜ) / sum(Δzᶜ))
l∞(c1, c2) = maximum(abs, c1 .- c2)

l²ᴱ = l²(exact_tracer, zeros(Nz))
l²ᴬᶜᴱ = l²(aiva_cfl_history[:, end], exact_tracer)
l²ᴬᶠᴱ = l²(aiva_fraction_history[:, end], exact_tracer)
l²ᴿᴱ = l²(reference_history[:, end], exact_tracer)
l²ᴬᶜᴿ = l²(aiva_cfl_history[:, end], reference_history[:, end])
l²ᴬᶠᴿ = l²(aiva_fraction_history[:, end], reference_history[:, end])
l∞ᴬᶜᴱ = l∞(aiva_cfl_history[:, end], exact_tracer)
l∞ᴬᶠᴱ = l∞(aiva_fraction_history[:, end], exact_tracer)
l∞ᴿᴱ = l∞(reference_history[:, end], exact_tracer)

cfl_diagnostics = adaptive_implicit_vertical_advection_diagnostics(aiva_cfl_scheme)
fraction_diagnostics = adaptive_implicit_vertical_advection_diagnostics(aiva_fraction_scheme)

#####
##### Summary
#####

println("\n=== AIVA stability + correctness validation ===")
@printf("grid: Nz=%d, Δz_min=%.3e, Δz_max=%.3e\n", Nz, minimum(Δzᶜ), maximum(Δzᶜ))
@printf("AIVA Δt = %.4e (CFL ≈ %.2f), reference Δτ = %.4e (CFL ≈ %.2f)\n", Δt, w₀ * Δt / Δzₘᵢₙ, Δτ, w₀ * Δτ / Δzₘᵢₙ)
@printf("Final time = %.4e (%d AIVA steps = %d reference steps)\n", Tₛ, Nₛ, Nₛ * Nsub)

println("\n--- Stability ---")
if blowup_step[] > 0
    @printf("Explicit %s at AIVA Δt: BLEW UP at step %d (t = %.3e)\n", summary(explicit_scheme), blowup_step[], blowup_step[] * Δt)
else
    println("Explicit ", summary(explicit_scheme), " at AIVA Δt: survived (NOT the intended outcome — increase Δt or w₀).")
end
@printf("AIVA fixed-CFL mode: completed %d steps, max|c| = %.3e (initial %.3e)\n", Nₛ, maximum(abs, aiva_cfl_history[:, end]), maximum(abs, aiva_cfl_history[:, 1]))
@printf("AIVA percentile mode: completed %d steps, max|c| = %.3e (initial %.3e)\n", Nₛ, maximum(abs, aiva_fraction_history[:, end]), maximum(abs, aiva_fraction_history[:, 1]))

println("\n--- Correctness (vs analytical translation) ---")
@printf("Analytical:      max|c| = %.3e, mass = %.4e, centroid = %.4f (target %.4f)\n", maximum(abs, exact_tracer), mᴱ, zᴱ, c₀ + w₀ * Tₛ)
@printf("Reference:       max|c| = %.3e, mass = %.4e, centroid = %.4f, L²-rel = %.3e, L∞ = %.3e\n", maximum(abs, reference_history[:, end]), mᴿ, zᴿ, l²ᴿᴱ / l²ᴱ, l∞ᴿᴱ)
@printf("AIVA fixed-CFL:  max|c| = %.3e, mass = %.4e, centroid = %.4f, L²-rel = %.3e, L∞ = %.3e\n", maximum(abs, aiva_cfl_history[:, end]), mᴬᶜ, zᴬᶜ, l²ᴬᶜᴱ / l²ᴱ, l∞ᴬᶜᴱ)
@printf("AIVA percentile: max|c| = %.3e, mass = %.4e, centroid = %.4f, L²-rel = %.3e, L∞ = %.3e\n", maximum(abs, aiva_fraction_history[:, end]), mᴬᶠ, zᴬᶠ, l²ᴬᶠᴱ / l²ᴱ, l∞ᴬᶠᴱ)
@printf("AIVA fixed-CFL vs reference:  L² = %.3e\n", l²ᴬᶜᴿ)
@printf("AIVA percentile vs reference: L² = %.3e\n", l²ᴬᶠᴿ)

println("\n--- Percentile diagnostics ---")
@printf("Fixed-CFL mode:  resolved threshold = %.3e, realized implicit fraction = %.3f, median CFL_w = %.3e, max CFL_w = %.3e\n",
        cfl_diagnostics.resolved_cfl, cfl_diagnostics.realized_implicit_fraction, cfl_diagnostics.median_cfl, cfl_diagnostics.max_cfl)
@printf("Percentile mode: resolved threshold = %.3e, realized implicit fraction = %.3f, median CFL_w = %.3e, max CFL_w = %.3e\n",
        fraction_diagnostics.resolved_cfl, fraction_diagnostics.realized_implicit_fraction, fraction_diagnostics.median_cfl, fraction_diagnostics.max_cfl)

println("\n--- Mass conservation (∫c dz) ---")
@printf("Initial:        fixed-CFL = %.6e, percentile = %.6e, reference = %.6e\n", aiva_cfl_mass[1], aiva_fraction_mass[1], reference_mass[1])
@printf("Final:          fixed-CFL = %.6e, percentile = %.6e, reference = %.6e\n", aiva_cfl_mass[end], aiva_fraction_mass[end], reference_mass[end])
@printf("Drift (final/initial − 1): fixed-CFL = %+.3e, percentile = %+.3e, reference = %+.3e\n",
        aiva_cfl_mass[end] / aiva_cfl_mass[1] - 1,
        aiva_fraction_mass[end] / aiva_fraction_mass[1] - 1,
        reference_mass[end] / reference_mass[1] - 1)

#####
##### Plot
#####

using CairoMakie

fig = Figure(size=(900, 1100))

ax1 = Axis(fig[1, 1], xlabel="t", ylabel="z", title="AIVA percentile mode c(z, t)")
heatmap!(ax1, 1:size(aiva_fraction_history, 2), zᶜ, aiva_fraction_history')

ax2 = Axis(fig[2, 1], xlabel="c", ylabel="z", title=@sprintf("Final profile (t = %.3f, w₀·t = %.3f)", Tₛ, w₀ * Tₛ))
lines!(ax2, aiva_fraction_history[:, 1], zᶜ, color=:black, linestyle=:dash, label="initial")
lines!(ax2, exact_tracer, zᶜ, color=:gray, linewidth=3, label="analytical")
lines!(ax2, reference_history[:, end], zᶜ, color=:green, label=@sprintf("reference (CFL ≈ %.2f)", w₀ * Δτ / Δzₘᵢₙ))
lines!(ax2, aiva_cfl_history[:, end], zᶜ, color=:blue, label="AIVA fixed-CFL")
lines!(ax2, aiva_fraction_history[:, end], zᶜ, color=:orange, label="AIVA percentile")
axislegend(ax2, position=:rb)

cmaxᴬᶜ = [maximum(abs, aiva_cfl_history[:, n + 1]) for n in 0:Nₛ]
cmaxᴬᶠ = [maximum(abs, aiva_fraction_history[:, n + 1]) for n in 0:Nₛ]
cmaxˣ = [maximum(abs, explicit_history[:, n + 1]) for n in 0:Nₛ]
cmaxᴿ = [maximum(abs, reference_history[:, n + 1]) for n in 0:Nₛ]
ax3 = Axis(fig[3, 1], xlabel="AIVA step", ylabel="max |c|")
lines!(ax3, 0:Nₛ, cmaxᴬᶜ, label="AIVA fixed-CFL")
lines!(ax3, 0:Nₛ, cmaxᴬᶠ, label="AIVA percentile")
lines!(ax3, 0:Nₛ, cmaxˣ, label="explicit (large Δt)")
lines!(ax3, 0:Nₛ, cmaxᴿ, label="explicit (small Δt)")
ylims!(ax3, 0.5, 1.5)
axislegend(ax3, position=:rb)
