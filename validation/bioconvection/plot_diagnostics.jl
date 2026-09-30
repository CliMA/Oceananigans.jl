# Diagnostic figures for gyrotactic_bioconvection.jl, generated from its JLD2 output:
#   1. growth of the horizontally averaged w² with the fitted exponential
#   2. horizontally averaged concentration profiles vs the up-swimming equilibrium
#   3. final concentration field with the gyrotactic orientation vectors
#
# Run with: julia --project validation/bioconvection/plot_diagnostics.jl

using Oceananigans
using CairoMakie
using Statistics
using Printf

H  = 5e-3
n₀ = 1e12
Pe = 10.0

filename = joinpath(@__DIR__, "gyrotactic_bioconvection.jld2")
nt = FieldTimeSeries(filename, "n")
wt = FieldTimeSeries(filename, "w")
pt = FieldTimeSeries(filename, "pˣ")
times = nt.times

data_color = "#0072B2"  # Okabe-Ito blue
fit_color  = "#D55E00"  # Okabe-Ito vermillion

# Figure 1: exponential growth of mean w²
energies = [mean(interior(wt[i]).^2) for i in eachindex(times)]
positive = findall(>(0), energies)
peak_energy = maximum(energies)
window = [i for i in positive if 1e-3 * peak_energy < energies[i] < 0.1 * peak_energy]
t = times[window]
ε = log.(energies[window])
slope = cov(t, ε) / var(t)
σ = slope / 2

fig1 = Figure(size = (700, 450))
ax1 = Axis(fig1[1, 1], yscale = log10,
           xlabel = "t (s)", ylabel = "mean w² (m² s⁻²)",
           title = "Growth of the gyrotactic overturning instability")
vspan!(ax1, t[1], t[end], color = (:gray, 0.12))
scatterlines!(ax1, times[positive], energies[positive],
              color = data_color, markersize = 6, label = "simulation")
fit_energies = exp.(mean(ε) .+ slope .* (t .- mean(t)))
lines!(ax1, t, fit_energies, color = fit_color, linestyle = :dash, linewidth = 3,
       label = @sprintf("fit: σ = %.3f s⁻¹", σ))
axislegend(ax1, position = :rb, framevisible = false)
save(joinpath(@__DIR__, "gyrotactic_bioconvection_growth.png"), fig1)

# Figure 2: horizontally averaged profiles vs the up-swimming equilibrium
xc, yc, zc = nodes(nt)
profile_times = [60, 120, 180, 240, 300]
ramp = cgrad(:blues)

fig2 = Figure(size = (600, 500))
ax2 = Axis(fig2[1, 1], xscale = log10,
           xlabel = "⟨n⟩ₓ / n₀", ylabel = "z (m)",
           title = "Cell concentration profiles vs equilibrium exp(Vₛz/D)")
equilibrium(z) = Pe * exp(Pe * z / H) / (1 - exp(-Pe))
lines!(ax2, equilibrium.(zc), zc, color = :black, linestyle = :dash, linewidth = 2.5,
       label = "analytic equilibrium")
for (j, target) in enumerate(profile_times)
    i = argmin(abs.(times .- target))
    profile = vec(mean(interior(nt[i], :, 1, :), dims = 1)) ./ n₀
    lines!(ax2, max.(profile, 1e-4), zc, linewidth = 2.5,
           color = ramp[0.35 + 0.65 * (j - 1) / (length(profile_times) - 1)],
           label = @sprintf("t = %d s", round(Int, times[i])))
end
xlims!(ax2, 5e-3, 30)
axislegend(ax2, position = :lb, framevisible = false)
save(joinpath(@__DIR__, "gyrotactic_bioconvection_profiles.png"), fig2)

# Figure 3: final n/n₀ with gyrotactic orientation vectors p̂
xp, yp, zp = nodes(pt)
pˣfinal = interior(pt[length(times)], :, 1, :)
pᶻfinal = sqrt.(max.(0, 1 .- pˣfinal.^2))
nfinal = interior(nt[length(times)], :, 1, :) ./ n₀

istride, kstride = 8, 6
isub = 1:istride:length(xp)
ksub = 3:kstride:length(zp)-6  # stop below the lid so arrow tips stay inside the frame
xa = repeat(xp[isub], outer = length(ksub))
za = repeat(zp[ksub], inner = length(isub))
ua = vec(pˣfinal[isub, ksub])
va = vec(pᶻfinal[isub, ksub])

fig3 = Figure(size = (750, 450))
ax3 = Axis(fig3[1, 1], xlabel = "x (m)", ylabel = "z (m)",
           title = @sprintf("n/n₀ and swimming orientation p̂ at t = %.0f s", times[end]))
hm = heatmap!(ax3, xc, zc, nfinal, colormap = :dense, colorrange = (0, 3))
Colorbar(fig3[1, 2], hm, label = "n / n₀")
arrows2d!(ax3, xa, za, ua, va, lengthscale = 3.5e-4, color = fit_color)
save(joinpath(@__DIR__, "gyrotactic_bioconvection_orientation.png"), fig3)

@info "Wrote growth, profiles, and orientation figures"
