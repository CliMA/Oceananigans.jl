# Figure for the output of baroclinic_channel.jl. The truth is also shown block-averaged onto the coarse grid,
# the part of the truth a coarse model could represent.
#
# Usage: julia --project plot_baroclinic_channel.jl, with the same BIHARMONIC_DAYS and WENO as the run.

using Oceananigans
using Oceananigans.Units
using CairoMakie
using FFTW
using Printf
using Statistics

τ₄ = parse(Float64, get(ENV, "BIHARMONIC_DAYS", "0.1")) * day
weno = parse(Bool, get(ENV, "WENO", "false"))
prefix = @sprintf("baroclinic_channel_%gday", τ₄ / day) * (weno ? "_weno" : "")

surface(case, name) = FieldTimeSeries(prefix * "_" * case * "_surface.jld2", name)

# Mean over r × r blocks of cells
coarse_grain(a, r) = [mean(a[(i-1)*r+1:i*r, (j-1)*r+1:j*r]) for i in 1:size(a, 1)÷r, j in 1:size(a, 2)÷r]

eddy_kinetic_energy(u, v) = (mean((u .- mean(u, dims=1)) .^ 2) + mean((v .- mean(v, dims=1)) .^ 2)) / 2

# Isotropic spectrum E(K), with K in cycles per domain, normalized so that ∑ E(K) is the mean surface kinetic energy
function isotropic_spectrum(velocities)
    Nλ, Nφ = size(first(velocities)[1])
    E = mean(abs2.(fft(u .- mean(u, dims=1))) .+ abs2.(fft(v .- mean(v, dims=1))) for (u, v) in velocities) ./ (2 * (Nλ * Nφ)^2)
    K = [round(Int, hypot(k, l)) for k in fftfreq(Nλ, Nλ), l in fftfreq(Nφ, Nφ)]
    return [sum(E[K .== n]) for n in 1:min(Nλ, Nφ)÷2]
end

function load(case; coarsening=1)
    ut, vt, ζt = surface(case, "u"), surface(case, "v"), surface(case, "ζ")
    Nλ, Nφ = size(ut.grid, 1), size(ut.grid, 2)
    velocities = [(coarse_grain(interior(ut[n], :, :, 1), coarsening), coarse_grain(interior(vt[n], :, 1:Nφ, 1), coarsening))
                  for n in eachindex(ut.times)]

    times = ut.times
    second_half = times .> times[end] / 2
    energies = [eddy_kinetic_energy(u, v) for (u, v) in velocities]
    ζ = coarse_grain(interior(ζt[end], :, 1:Nφ, 1), coarsening)

    V₁ = FieldTimeSeries(prefix * "_" * case * "_averages.jld2", "V₁")[end]
    Ψ = interior(Field(Average(V₁, dims=1)), 1, :, 1)

    return (; times, energies, spectrum = isotropic_spectrum(velocities[second_half]),
              mean_energy = mean(energies[second_half]), ζ, Ψ, φ = φnodes(V₁.grid, Face()))
end

coarsening = size(surface("truth", "u").grid, 1) ÷ size(surface("coarse", "u").grid, 1)

colors = Makie.wong_colors()

cases = [("truth",                  colors[1], load("truth")),
         ("truth coarse-grained",   colors[1], load("truth"; coarsening)),
         ("coarse",                 colors[2], load("coarse")),
         ("coarse backscatter",     colors[3], load("coarse_backscatter"))]

fig = Figure(size=(2000, 1100))
energy_axis = Axis(fig[1, 1:2], xlabel="t (days)", ylabel="surface EKE (m² s⁻²)")
spectrum_axis = Axis(fig[1, 3:4], xscale=log10, yscale=log10, xlabel="K (cycles per domain)", ylabel="E(K)",
                     title="Second-half mean, surface")
overturning_axis = Axis(fig[2, 1:4], xlabel="latitude (°)", ylabel="Ψ (m² s⁻¹)", title="Second-half mean overturning")

for (label, color, result) in cases
    linestyle = label == "truth coarse-grained" ? :dash : :solid
    lines!(energy_axis, result.times ./ day, result.energies; label, color, linestyle)
    lines!(spectrum_axis, result.spectrum; label, color, linestyle)
    label == "truth coarse-grained" || lines!(overturning_axis, result.φ, result.Ψ; label, color)
    @info @sprintf("%s: second-half mean surface EKE = %.2e m² s⁻², peak |Ψ| = %.2f m² s⁻¹",
                   label, result.mean_energy, maximum(abs, result.Ψ))
end

axislegend(energy_axis, position=:lt)
axislegend(overturning_axis, position=:rb)

for (n, (label, _, result)) in enumerate(cases)
    ζ_axis = Axis(fig[3, n], title="surface ζ, $label")
    ζmax = maximum(abs, result.ζ)
    heatmap!(ζ_axis, result.ζ, colormap=:balance, colorrange=(-ζmax, ζmax))
end

save(prefix * ".png", fig)
