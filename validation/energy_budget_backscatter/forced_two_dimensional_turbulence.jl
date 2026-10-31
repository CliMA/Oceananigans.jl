# Forced two-dimensional turbulence: the simplest flow in which grid-scale viscosity removes energy that should
# cascade upscale. A fine run (truth) is compared to coarse runs with and without EnergyBudgetBackscatter.
# The flow is stirred near the coarse grid scale and damped at large scales by linear drag.
#
# Usage: julia --project forced_two_dimensional_turbulence.jl   (needs CairoMakie and FFTW)

using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: EnergyConserving
using CairoMakie
using FFTW
using Printf
using Random
using Statistics

L = 1024kilometers
H = 1000
μ = 1 / 50days                    # linear drag
ω = 2π / 30days                   # rotation rate of the stirring pattern
F₀ = parse(Float64, get(ENV, "STIRRING_AMPLITUDE", "2e-7"))   # rms stirring acceleration (m s⁻²)
τ₄ = parse(Float64, get(ENV, "BIHARMONIC_DAYS", "1")) * day   # biharmonic damping time at the grid scale Δ / π
stop_time = parse(Float64, get(ENV, "STOP_YEARS", "2")) * 365days

#####
##### Stirring: a streamfunction in the ring 15 ≤ |k| ≤ 17 (wavelength ≈ 4 coarse cells), the same at every resolution
#####

modes = [(k, l) for k in -17:17 for l in 0:17 if 15 ≤ hypot(k, l) ≤ 17 && (l > 0 || k > 0)]
phases = 2π * rand(Xoshiro(1), length(modes))

# Velocity forcing -∂yψ, ∂xψ of ψ = Σ A cos(θ + ωt), split into the coefficients of cos(ωt) and sin(ωt)
function stirring(x, y, component, part)
    F = zero(x)
    for ((k, l), φ) in zip(modes, phases)
        θ = 2π * (k * x + l * y) / L + φ
        amplitude = F₀ / sqrt(length(modes)) / hypot(k, l)
        F += amplitude * (component == :u ? l : -k) * (part == :cos ? sin(θ) : cos(θ))
    end
    return F
end

@inline u_stirring_and_drag(i, j, k, grid, clock, fields, p) =
    @inbounds p.Fᶜ[i, j, k] * cos(p.ω * clock.time) + p.Fˢ[i, j, k] * sin(p.ω * clock.time) - p.μ * fields.u[i, j, k]

@inline v_stirring_and_drag(i, j, k, grid, clock, fields, p) =
    @inbounds p.Fᶜ[i, j, k] * cos(p.ω * clock.time) + p.Fˢ[i, j, k] * sin(p.ω * clock.time) - p.μ * fields.v[i, j, k]

function run_turbulence(Δ; backscatter=false)
    N = round(Int, L / Δ)
    grid = RectilinearGrid(size=(N, N, 1), halo=(4, 4, 4), x=(0, L), y=(0, L), z=(-H, 0),
                           topology=(Periodic, Periodic, Bounded))

    forcings = map((u=(XFaceField, u_stirring_and_drag), v=(YFaceField, v_stirring_and_drag))) do (location, func)
        Fᶜ, Fˢ = location(grid), location(grid)
        component = location === XFaceField ? :u : :v
        set!(Fᶜ, (x, y, z) -> stirring(x, y, component, :cos))
        set!(Fˢ, (x, y, z) -> stirring(x, y, component, :sin))
        Forcing(func, discrete_form=true, parameters=(; Fᶜ, Fˢ, ω, μ))
    end

    biharmonic_viscosity = HorizontalScalarBiharmonicDiffusivity(ν = (Δ / π)^4 / τ₄)
    closure = backscatter ? (biharmonic_viscosity, EnergyBudgetBackscatter()) : biharmonic_viscosity

    model = HydrostaticFreeSurfaceModel(grid; closure, forcing=forcings, buoyancy=nothing, tracers=(),
                                        momentum_advection = VectorInvariant(vorticity_scheme=EnergyConserving()))

    simulation = Simulation(model; Δt = 15minutes * Δ / 4kilometers, stop_time)

    u, v, w = model.velocities
    times, energies, velocities = Float64[], Float64[], []

    function record(sim)
        push!(times, time(sim))
        push!(energies, (mean(interior(u) .^ 2) + mean(interior(v) .^ 2)) / 2)
        time(sim) > stop_time / 2 && push!(velocities, (Array(interior(u, :, :, 1)), Array(interior(v, :, :, 1))))
        return nothing
    end

    progress(sim) = @info @sprintf("Δ = %d km%s: %s, KE = %.2e m² s⁻²", Δ / kilometers,
                                   backscatter ? " with backscatter" : "", prettytime(sim), energies[end])

    add_callback!(simulation, record, TimeInterval(5days))
    add_callback!(simulation, progress, TimeInterval(100days))
    run!(simulation)

    ζ = Field(∂x(v) - ∂y(u))
    compute!(ζ)

    return (; times, energies, velocities, ζ)
end

# Isotropic spectrum E(K), with K in cycles per domain, normalized so that ∑ E(K) is the mean kinetic energy
function isotropic_spectrum(velocities)
    N = size(first(velocities)[1], 1)
    E = mean(abs2.(fft(u)) .+ abs2.(fft(v)) for (u, v) in velocities) ./ (2 * N^4)
    K = [round(Int, hypot(k, l)) for k in fftfreq(N, N), l in fftfreq(N, N)]
    return [sum(E[K .== n]) for n in 1:N÷2]
end

cases = [("4 km (truth)",       4kilometers,  false),
         ("16 km",              16kilometers, false),
         ("16 km backscatter",  16kilometers, true)]

results = [run_turbulence(Δ; backscatter) for (_, Δ, backscatter) in cases]

#####
##### Figure
#####

fig = Figure(size=(1200, 800))
energy_axis = Axis(fig[1, 1:2], xlabel="t (days)", ylabel="KE (m² s⁻²)")
spectrum_axis = Axis(fig[1, 3], xscale=log10, yscale=log10, xlabel="K (cycles per domain)", ylabel="E(K)",
                     title="Second-half mean")

for ((label, _, _), result) in zip(cases, results)
    lines!(energy_axis, result.times ./ day, result.energies; label)
    lines!(spectrum_axis, isotropic_spectrum(result.velocities); label)
    second_half = result.times .> stop_time / 2
    @info @sprintf("%s: second-half mean KE = %.2e m² s⁻²", label, mean(result.energies[second_half]))
end

vlines!(spectrum_axis, [16], color=:gray, linestyle=:dash)
axislegend(energy_axis, position=:lt)

ζmax = maximum(abs, interior(first(results).ζ))
for (n, ((label, _, _), result)) in enumerate(zip(cases, results))
    ζ_axis = Axis(fig[2, n], title="ζ, $label", aspect=1)
    heatmap!(ζ_axis, result.ζ, colormap=:balance, colorrange=(-ζmax, ζmax))
end

save(@sprintf("forced_two_dimensional_turbulence_%gday.png", τ₄ / day), fig)
