# Diagnostic figures for migrating_swimmers_stratified.jl, generated from its JLD2 output:
#   1. migration Hovmöller and downward jet speed vs the observed 1-2 cm/s
#   2. before/after stratification profiles
#   3. effective diffusivity: flux-gradient (primary) vs sorted-profile (check)
#   4. energetics: BPE, APE, irreversible mixing rate, mixing efficiency
#   5. eddy scale: x-spectra of w at the aggregation depth vs animal and swarm scales
#
# The flux-gradient κeff sees only resolved fluxes while the BPE/sorted diagnostics also
# absorb WENO's implicit dissipation on b; the two bracket the true mixing.
#
# Run with: julia --project validation/bioconvection/plot_migrating_swimmers.jl

using Oceananigans
using CairoMakie
using JLD2
using Statistics
using Printf

Lx = 0.5
Lz = 1.2
N² = 0.01    # interface buoyancy frequency squared
δ  = 0.1     # pycnocline half-thickness
zᵖ = -Lz/2   # pycnocline depth
Δb = √π * N² * δ
κb = 1e-7
κsalt = 1.5e-9   # molecular salt diffusivity, the normalization of Houghton et al. Fig. 3b
n₀ = 1.5e6
σx = 0.04
animal_length = 0.01

data_color = "#0072B2"  # Okabe-Ito blue
fit_color  = "#D55E00"  # Okabe-Ito vermillion
band_color = (:gray, 0.15)

prefix = joinpath(@__DIR__, "migrating_swimmers_stratified")

Bt  = FieldTimeSeries(prefix * "_profiles.jld2", "B")
Wt  = FieldTimeSeries(prefix * "_profiles.jld2", "W")
WBt = FieldTimeSeries(prefix * "_profiles.jld2", "WB")
N̄t  = FieldTimeSeries(prefix * "_profiles.jld2", "N̄")
profile_times = Bt.times

grid = Bt.grid
Nz = size(grid, 3)
zc = znodes(grid, Center())
zf = znodes(grid, Face())
Δz = zc[2] - zc[1]

energetics = jldopen(prefix * "_energetics.jld2")
bpe_times = energetics["times"]
bpe = energetics["background_potential_energies"]
ape = energetics["available_potential_energies"]
dissipation = energetics["dissipation_rates"]
Φᵢ = energetics["diffusive_bpe_fluxes"]
jet_times = energetics["jet_times"]
jet_speeds = energetics["jet_speeds"]
close(energetics)

# Figure 1: migration Hovmöller and jet speed
concentration = Array{Float64}(undef, length(profile_times), Nz)
for (i, _) in enumerate(profile_times)
    concentration[i, :] = interior(N̄t[i], 1, 1, :) ./ n₀
end

fig1 = Figure(size = (1000, 420))
ax1a = Axis(fig1[1, 1], xlabel = "t (min)", ylabel = "z (m)",
            title = "⟨n⟩ₓ / n₀: upward migration and lid accumulation")
hm1 = heatmap!(ax1a, profile_times ./ 60, zc, concentration, colormap = :dense,
               colorrange = (0, 0.5))
Colorbar(fig1[1, 2], hm1)
ax1b = Axis(fig1[1, 3], xlabel = "t (min)", ylabel = "max downward w (cm/s)",
            title = "Aggregation-scale jet")
hspan!(ax1b, 1.0, 2.0, color = band_color)
lines!(ax1b, jet_times ./ 60, 100 .* jet_speeds, color = data_color, linewidth = 2)
text!(ax1b, 0.98, 0.02, text = "observed 1-2 cm/s", space = :relative,
      align = (:right, :bottom), color = :gray)
save(joinpath(@__DIR__, "migrating_swimmers_jet.png"), fig1)

# Figure 2: pycnocline erosion in the coordinates of Houghton et al. Fig. 3a —
# normalized density vs height above the interface, over their measurement window
snapshot_targets = [0.0, profile_times[end] / 2, profile_times[end]]
ramp = cgrad(:blues)

fig2 = Figure(size = (900, 480))
ax2a = Axis(fig2[1, 1], xlabel = "⟨b⟩ / (Δb/2)", ylabel = "height above pycnocline (m)",
            title = "Mean buoyancy")
ax2b = Axis(fig2[1, 2], xlabel = "N²(z) / N²₀", title = "Local stratification")
for (j, target) in enumerate(snapshot_targets)
    i = argmin(abs.(profile_times .- target))
    b̄ = interior(Bt[i], 1, 1, :)
    stratification = [(b̄[k] - b̄[k-1]) / Δz / N² for k in 2:Nz]
    color = ramp[0.35 + 0.65 * (j - 1) / (length(snapshot_targets) - 1)]
    label = @sprintf("t = %.0f min", profile_times[i] / 60)
    lines!(ax2a, b̄ ./ (Δb/2), zc .- zᵖ; color, linewidth = 2.5, label)
    lines!(ax2b, stratification, zf[2:Nz] .- zᵖ; color, linewidth = 2.5, label)
end
vlines!(ax2b, [1.0], color = :gray, linestyle = :dash)
for ax in (ax2a, ax2b)
    ylims!(ax, -0.4, 0.35)
end
xlims!(ax2a, -1.5, 1.5)
axislegend(ax2a, position = :lt, framevisible = false)
save(joinpath(@__DIR__, "migrating_swimmers_stratification.png"), fig2)

# Figure 3: effective diffusivity, flux-gradient (primary) vs sorted-profile (check)
faces = 2:Nz  # interior faces
κeff = fill(NaN, length(profile_times), length(faces))
for (i, _) in enumerate(profile_times)
    b̄ = interior(Bt[i], 1, 1, :)
    w̄ = interior(Wt[i], 1, 1, :)
    wb = interior(WBt[i], 1, 1, :)
    for (j, k) in enumerate(faces)
        ∂zb̄ = (b̄[k] - b̄[k-1]) / Δz
        # mask near-zero gradients: the flux-gradient ratio blows up in mixed patches
        ∂zb̄ < 0.05N² && continue
        b̄ᶠ = (b̄[k] + b̄[k-1]) / 2
        w′b′ = wb[k] - w̄[k] * b̄ᶠ
        κeff[i, j] = -w′b′ / ∂zb̄ + κb
    end
end

# sorted-profile check: invert the 1D heat equation for the sorted buoyancy b★
bt = FieldTimeSeries(prefix * "_fields.jld2", "b")
field_times = bt.times
Nx = size(grid, 1)
b★ = Array{Float64}(undef, length(field_times), Nz)
for (i, _) in enumerate(field_times)
    sorted = sort!(vec(Array(interior(bt[i], :, 1, :))))
    b★[i, :] = vec(mean(reshape(sorted, Nx, Nz), dims = 1))
end
κsorted = fill(NaN, length(field_times) - 1, length(faces))
for i in 1:length(field_times)-1
    Δt = field_times[i+1] - field_times[i]
    ∂tb★ = (b★[i+1, :] .- b★[i, :]) ./ Δt
    flux = cumsum(∂tb★) .* Δz
    for (j, k) in enumerate(faces)
        ∂zb★ = ((b★[i, k] + b★[i+1, k]) / 2 - (b★[i, k-1] + b★[i+1, k-1]) / 2) / Δz
        ∂zb★ < 0.05N² && continue
        κsorted[i, j] = flux[k-1] / ∂zb★
    end
end

mean_ignoring_nan(v) = (s = filter(!isnan, v); isempty(s) ? NaN : mean(s))
mean_κsorted = [mean_ignoring_nan(κsorted[:, j]) for j in eachindex(faces)]

# Houghton et al. obtain κeff by fitting the before/after density profiles with the 1D
# diffusion equation; the same event-integrated inversion of ⟨b⟩ is robust where the
# time-mean of instantaneous flux-gradient ratios is not (near-threshold gradients).
b̄initial = interior(Bt[1], 1, 1, :)
b̄final = interior(Bt[length(profile_times)], 1, 1, :)
T = profile_times[end] - profile_times[1]
∫Δb̄ = cumsum(b̄final .- b̄initial) .* Δz
κhoughton = fill(NaN, length(faces))
for (j, k) in enumerate(faces)
    ∂zb̃ = ((b̄initial[k] - b̄initial[k-1]) + (b̄final[k] - b̄final[k-1])) / 2Δz
    ∂zb̃ < 0.1N² && continue
    κhoughton[j] = ∫Δb̄[k-1] / (T * ∂zb̃)
end
peak_enhancement = maximum(filter(!isnan, κhoughton)) / κsalt

# right panel in the coordinates of Houghton et al. Fig. 3b: κeff/κsalt vs height
# above the pycnocline (gradients vanish in the homogeneous layers, so κeff is only
# defined near the interface)
fig3 = Figure(size = (1100, 480))
ax3a = Axis(fig3[1, 1], xlabel = "t (min)", ylabel = "z (m)",
            title = "log₁₀(κeff / κb), flux-gradient")
hm3 = heatmap!(ax3a, profile_times ./ 60, zf[faces], log10.(max.(κeff ./ κb, 0.1)),
               colormap = :thermal, colorrange = (0, 3))
Colorbar(fig3[1, 2], hm3)
ax3b = Axis(fig3[1, 3], xlabel = "κeff / κsalt", ylabel = "height above pycnocline (m)",
            xscale = log10,
            title = @sprintf("Event-integrated κeff, peak = %.0f κsalt", peak_enhancement))
lines!(ax3b, max.(κhoughton ./ κsalt, 10), zf[faces] .- zᵖ, color = data_color,
       linewidth = 2.5, label = "profile-fit (Houghton method)")
lines!(ax3b, max.(mean_κsorted ./ κsalt, 10), zf[faces] .- zᵖ, color = fit_color,
       linewidth = 2.5, label = "sorted-profile, time mean")
vlines!(ax3b, [κb / κsalt], color = :gray, linestyle = :dash, label = "model κb")
ylims!(ax3b, -0.4, 0.35)
xlims!(ax3b, 10, 1e5)
axislegend(ax3b, position = :rb, framevisible = false)
save(joinpath(@__DIR__, "migrating_swimmers_diffusivity.png"), fig3)

# Figure 4: energetics and mixing efficiency
trapezoid(t, y) = sum((y[i] + y[i+1]) / 2 * (t[i+1] - t[i]) for i in 1:length(t)-1)
diffusive_reference = bpe[1] .+ [0; cumsum((Φᵢ[1:end-1] .+ Φᵢ[2:end]) ./ 2 .* diff(bpe_times))]

Φd = similar(bpe)
Φd[2:end-1] = (bpe[3:end] .- bpe[1:end-2]) ./ (bpe_times[3:end] .- bpe_times[1:end-2]) .- Φᵢ[2:end-1]
Φd[1] = Φd[2]; Φd[end] = Φd[end-1]
smoothing = 5
smoothed_Φd = [mean(Φd[max(1, i-smoothing):min(end, i+smoothing)]) for i in eachindex(Φd)]

mixing = trapezoid(bpe_times, max.(Φd, 0))
dissipated = trapezoid(bpe_times, dissipation)
efficiency = mixing / (mixing + dissipated)

fig4 = Figure(size = (1000, 450))
ax4a = Axis(fig4[1, 1], xlabel = "t (min)", ylabel = "energy (m² s⁻²)",
            title = "Potential energy budget")
lines!(ax4a, bpe_times ./ 60, bpe .- bpe[1], color = data_color, linewidth = 2.5,
       label = "BPE - BPE₀")
lines!(ax4a, bpe_times ./ 60, diffusive_reference .- bpe[1], color = :gray,
       linestyle = :dash, linewidth = 2, label = "diffusion only")
lines!(ax4a, bpe_times ./ 60, ape, color = fit_color, linewidth = 2.5, label = "APE")
axislegend(ax4a, position = :lt, framevisible = false)
ax4b = Axis(fig4[1, 2], xlabel = "t (min)", ylabel = "rate (m² s⁻³)",
            title = @sprintf("Mixing rate Φd and dissipation ε, η = %.2f", efficiency))
lines!(ax4b, bpe_times ./ 60, smoothed_Φd, color = data_color, linewidth = 2, label = "Φd")
lines!(ax4b, bpe_times ./ 60, dissipation, color = fit_color, linewidth = 2, label = "⟨ε⟩")
lines!(ax4b, bpe_times ./ 60, Φᵢ, color = :gray, linestyle = :dash, label = "Φᵢ")
axislegend(ax4b, position = :rt, framevisible = false)
save(joinpath(@__DIR__, "migrating_swimmers_energetics.png"), fig4)

# Figure 5: eddy scale ~ aggregation scale, not animal scale
wt = FieldTimeSeries(prefix * "_fields.jld2", "w")
nt = FieldTimeSeries(prefix * "_fields.jld2", "n")
xc = xnodes(grid, Center())
wavenumbers = 1:(Nx ÷ 2 - 1)
wavelengths = Lx ./ wavenumbers

spectrum_targets = [0.2, 0.4, 0.6, 0.8] .* field_times[end]
fig5 = Figure(size = (1100, 480))
ax5a = Axis(fig5[1, 1], xlabel = "wavelength (m)", ylabel = "|ŵ|² (m² s⁻²)",
            xscale = log10, yscale = log10,
            title = "x-spectra of w at the aggregation depth")
for (j, target) in enumerate(spectrum_targets)
    i = argmin(abs.(field_times .- target))
    n̄profile = vec(mean(Array(interior(nt[i], :, 1, :)), dims = 1))
    kagg = argmax(n̄profile)
    wrow = Array(interior(wt[i], :, 1, kagg))
    ŵ = [abs(sum(wrow .* exp.(-2π * im * m .* xc ./ Lx)))^2 / Nx^2 for m in wavenumbers]
    lines!(ax5a, wavelengths, max.(ŵ, 1e-14), linewidth = 2,
           color = ramp[0.35 + 0.65 * (j - 1) / (length(spectrum_targets) - 1)],
           label = @sprintf("t = %.1f min", field_times[i] / 60))
end
vlines!(ax5a, [σx], color = fit_color, linestyle = :dash, label = "σx (aggregation)")
vlines!(ax5a, [animal_length], color = :gray, linestyle = :dot, label = "animal length")
axislegend(ax5a, position = :lt, framevisible = false)

middle = argmin(abs.(field_times .- field_times[end] / 2))
ωt = FieldTimeSeries(prefix * "_fields.jld2", "ω")
ax5b = Axis(fig5[1, 2], xlabel = "x (m)", ylabel = "z (m)",
            title = @sprintf("ω and n/n₀ = 0.1 contour, t = %.1f min", field_times[middle] / 60))
xω, yω, zω = nodes(ωt)
hm5 = heatmap!(ax5b, xω, zω, Array(interior(ωt[middle], :, 1, :)),
               colormap = :vik, colorrange = (-1, 1))
contour!(ax5b, xc, zc, Array(interior(nt[middle], :, 1, :)) ./ n₀,
         levels = [0.1], color = :black, linewidth = 1.5)
Colorbar(fig5[1, 3], hm5, label = "ω (s⁻¹)")
save(joinpath(@__DIR__, "migrating_swimmers_eddy_scale.png"), fig5)

@info "Wrote jet, stratification, diffusivity, energetics, and eddy-scale figures"
