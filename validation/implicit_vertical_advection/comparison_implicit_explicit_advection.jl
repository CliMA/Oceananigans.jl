# AdaptiveImplicitVerticalAdvection (AIVA) validation: stability + correctness.
#
# Stretched 1D column with steady, uniform vertical velocity w = w₀. The tracer
# equation reduces to pure translation c(z, t) = c₀(z − w₀ t).
#
# Three runs at the same final time:
#   1. Explicit UpwindBiased(5), large Δt (CFL ≈ 4 in thin cells) — blows up.
#   2. AIVA wrapping the same explicit scheme, same large Δt      — survives.
#   3. Reference: explicit UpwindBiased(5), small Δt (CFL ≈ 0.5).
#
# Compared against the analytical Gaussian translated by w₀·T.

using Printf
using Oceananigans
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection
using Oceananigans.Grids: minimum_zspacing, zspacings

#####
##### Grid
#####

const Nz = 64
const Lz = 1.0
const refinement = 3.0

ξ(k) = (k - 1) / Nz
zᶠ = [Lz * (tanh(refinement * ξ(k)) / tanh(refinement) - 1) for k in 1:Nz+1]

grid = RectilinearGrid(CPU(); size=Nz, z=zᶠ, topology=(Flat, Flat, Bounded))

#####
##### Setup
#####

const w₀ = 1.0
const c₀ = -0.55
const σc = 0.08

initial_tracer(z) = exp(-(z - c₀)^2 / σc^2)
prescribed_velocity(z, t) = w₀

velocities = PrescribedVelocityFields(w=prescribed_velocity)

explicit_scheme = UpwindBiased(order=5)
aiva_scheme     = AdaptiveImplicitVerticalAdvection(explicit_scheme=explicit_scheme, cfl=0.5)

build_model(advection) = HydrostaticFreeSurfaceModel(grid;
                                                     velocities,
                                                     tracers = :c,
                                                     tracer_advection = advection,
                                                     buoyancy = nothing,
                                                     closure = nothing)

explicit_model  = build_model(explicit_scheme)
aiva_model      = build_model(aiva_scheme)
reference_model = build_model(explicit_scheme)

set!(explicit_model,  c=initial_tracer)
set!(aiva_model,      c=initial_tracer)
set!(reference_model, c=initial_tracer)

#####
##### Time-stepping
#####

Δzₘᵢₙ = minimum_zspacing(grid)
Δt    = 7.5 * Δzₘᵢₙ / w₀   # CFL ≈ 7.5
Δτ    = 0.5 * Δzₘᵢₙ / w₀   # CFL ≈ 0.5 (reference)

# Place final centroid at z ≈ −0.15 (well below the top stretched zone)
final_time = (-0.15 - c₀) / w₀
Nₛ   = round(Int, final_time / Δt)
Nᵣ   = round(Int, final_time / Δτ)
Nsub = round(Int, Δt / Δτ)

@info "Δz_min = $(Δzₘᵢₙ), w₀ = $(w₀)"
@info "AIVA Δt = $(Δt) (CFL ≈ $(w₀*Δt/Δzₘᵢₙ)), $Nₛ steps"
@info "Ref  Δτ = $(Δτ) (CFL ≈ $(w₀*Δτ/Δzₘᵢₙ)), $Nᵣ steps"

#####
##### Containers and initial sampling
#####

zᶜ  = collect(znodes(aiva_model.tracers.c))
Δzᶜ = vec(collect(zspacings(grid, Center(), Center(), Center())))

explicit_history  = zeros(Nz, Nₛ + 1)
aiva_history      = zeros(Nz, Nₛ + 1)
reference_history = zeros(Nz, Nₛ + 1)
aiva_mass         = zeros(Nₛ + 1)
reference_mass    = zeros(Nₛ + 1)

sample!(history, model) = (history .= Array(interior(model.tracers.c))[1, 1, :])

sample!(view(explicit_history,  :, 1), explicit_model)
sample!(view(aiva_history,      :, 1), aiva_model)
sample!(view(reference_history, :, 1), reference_model)
aiva_mass[1]      = sum(aiva_history[:, 1]      .* Δzᶜ)
reference_mass[1] = sum(reference_history[:, 1] .* Δzᶜ)

#####
##### Time-step loop (large Δt for AIVA / explicit, substep for reference)
#####

blowup_step      = Ref(-1)
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

    time_step!(aiva_model, Δt)
    sample!(view(aiva_history, :, n + 1), aiva_model)
    aiva_mass[n + 1] = sum(aiva_history[:, n + 1] .* Δzᶜ)

    for _ in 1:Nsub
        time_step!(reference_model, Δτ)
    end
    sample!(view(reference_history, :, n + 1), reference_model)
    reference_mass[n + 1] = sum(reference_history[:, n + 1] .* Δzᶜ)
end

#####
##### Analytical reference and diagnostics
#####

Tₛ           = Nₛ * Δt
exact_tracer = [exp(-(zᶜ[k] - c₀ - w₀ * Tₛ)^2 / σc^2) for k in 1:Nz]

function centroid(c, z, dz)
    m = sum(c .* dz)
    return sum(c .* z .* dz) / m, m
end

zᴬ, mᴬ = centroid(aiva_history[:, end],      zᶜ, Δzᶜ)
zᴿ, mᴿ = centroid(reference_history[:, end], zᶜ, Δzᶜ)
zᴱ, mᴱ = centroid(exact_tracer,              zᶜ, Δzᶜ)

l²(c1, c2) = sqrt(sum((c1 .- c2).^2 .* Δzᶜ) / sum(Δzᶜ))
l∞(c1, c2) = maximum(abs, c1 .- c2)

l²ᴱ  = l²(exact_tracer, zeros(Nz))
l²ᴬᴱ = l²(aiva_history[:, end],      exact_tracer)
l²ᴿᴱ = l²(reference_history[:, end], exact_tracer)
l²ᴬᴿ = l²(aiva_history[:, end],      reference_history[:, end])
l∞ᴬᴱ = l∞(aiva_history[:, end],      exact_tracer)
l∞ᴿᴱ = l∞(reference_history[:, end], exact_tracer)

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
@printf("AIVA at large Δt: completed %d steps, max|c| = %.3e (initial %.3e)\n", Nₛ, maximum(abs, aiva_history[:, end]), maximum(abs, aiva_history[:, 1]))

println("\n--- Correctness (vs analytical translation) ---")
@printf("Analytical: max|c| = %.3e, mass = %.4e, centroid = %.4f (target %.4f)\n", maximum(abs, exact_tracer), mᴱ, zᴱ, c₀ + w₀ * Tₛ)
@printf("Reference:  max|c| = %.3e, mass = %.4e, centroid = %.4f, L²-rel = %.3e, L∞ = %.3e\n", maximum(abs, reference_history[:, end]), mᴿ, zᴿ, l²ᴿᴱ / l²ᴱ, l∞ᴿᴱ)
@printf("AIVA:       max|c| = %.3e, mass = %.4e, centroid = %.4f, L²-rel = %.3e, L∞ = %.3e\n", maximum(abs, aiva_history[:, end]), mᴬ, zᴬ, l²ᴬᴱ / l²ᴱ, l∞ᴬᴱ)
@printf("AIVA vs reference: L² = %.3e\n", l²ᴬᴿ)

println("\n--- Mass conservation (∫c dz) ---")
@printf("Initial:        AIVA = %.6e, reference = %.6e\n", aiva_mass[1],   reference_mass[1])
@printf("Final:          AIVA = %.6e, reference = %.6e\n", aiva_mass[end], reference_mass[end])
@printf("Drift (final/initial − 1): AIVA = %+.3e, reference = %+.3e\n", aiva_mass[end] / aiva_mass[1] - 1, reference_mass[end] / reference_mass[1] - 1)
@printf("Min/max during AIVA run: %.6e / %.6e\n", minimum(aiva_mass), maximum(aiva_mass))

#####
##### Plot
#####

using CairoMakie

fig = Figure(size=(900, 1000))

# AIVA Hovmöller
ax1 = Axis(fig[1, 1], xlabel="t", ylabel="z", title="AIVA c(z, t)")
heatmap!(ax1, 1:size(aiva_history, 2), zᶜ, aiva_history')

# Final-time profiles
ax2 = Axis(fig[2, 1], xlabel="c", ylabel="z", title=@sprintf("Final profile (t = %.3f, w₀·t = %.3f)", Tₛ, w₀ * Tₛ))
lines!(ax2, aiva_history[:, 1],        zᶜ, color=:black, linestyle=:dash, label="initial")
lines!(ax2, exact_tracer,              zᶜ, color=:gray,  linewidth=3, label="analytical")
lines!(ax2, reference_history[:, end], zᶜ, color=:green, label=@sprintf("reference (CFL ≈ %.2f)", w₀ * Δτ / Δzₘᵢₙ))
lines!(ax2, aiva_history[:, end],      zᶜ, color=:blue,  label=@sprintf("AIVA (CFL ≈ %.1f)", w₀ * Δt / Δzₘᵢₙ))
axislegend(ax2, position=:rb)

# max|c| vs step (log)
cmaxᴬ = [maximum(abs, aiva_history[:, n + 1])     for n in 0:Nₛ]
cmaxˣ = [maximum(abs, explicit_history[:, n + 1]) for n in 0:Nₛ]
cmaxᴿ = [maximum(abs, reference_history[:, n + 1]) for n in 0:Nₛ]
ax3 = Axis(fig[3, 1], xlabel="AIVA step", ylabel="max |c|", yscale=log10)
lines!(ax3, 0:Nₛ, cmaxᴬ, label="AIVA (large Δt)")
lines!(ax3, 0:Nₛ, cmaxˣ, label="explicit (large Δt)")
lines!(ax3, 0:Nₛ, cmaxᴿ, label="explicit (small Δt)")
ylims!(ax3, 0.85, 1.01)
axislegend(ax3, position=:rb)
