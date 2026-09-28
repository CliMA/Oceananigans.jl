# Figures of the doubly periodic f-plane experiments of oriented_coriolis_periodic_experiments.jl: adjustment from
# white noise, decaying turbulence, forced turbulence with drag, and the energy and enstrophy budgets.
# The energy- and enstrophy-conserving schemes coincide on an f-plane, so the first represents both.
#
# Usage: PERIODIC_OUTPUT=<directory of the .jld2 files> FIGURE_DIRECTORY=<output> julia periodic_figures.jl

using JLD2
using Statistics
using Printf
using CairoMakie

include("spectra.jl")

const f₀ = 1e-4
const day = 86400.0

directory = get(ENV, "PERIODIC_OUTPUT", ".")
output = get(ENV, "FIGURE_DIRECTORY", ".")
preview = get(ENV, "PREVIEW_DIRECTORY", output)

function load_experiment(experiment)
    for prefix in ("oriented_coriolis", "chiral_coriolis")
        file = joinpath(directory, "$(prefix)_$(experiment).jld2")
        isfile(file) && return load(file, "outputs")
    end
    error("no output for $experiment in $directory")
end

oriented_key(outputs) = haskey(outputs, "OrientedCoriolis") ? "OrientedCoriolis" : "ChiralCoriolis"
cases(outputs) = (("Four-point", outputs["EnergyConserving"]), ("C-D", outputs["CDScheme"]), ("Oriented", outputs[oriented_key(outputs)]))

colors = Dict("Four-point" => :royalblue, "C-D" => :darkorange, "Oriented" => :crimson)

kinetic_energy(s) = (mean(abs2, s.u) + mean(abs2, s.v)) / 2
enstrophy(s) = mean(abs2, s.ζ)
time_mean(snapshots, variable, window) = mean(getproperty(s, variable) for s in snapshots if window[1] ≤ s.t ≤ window[2])
running_mean(a, n) = [mean(a[max(1, m - n ÷ 2):min(length(a), m + n ÷ 2)]) for m in eachindex(a)]

const spectrum_axis = (xscale=log10, yscale=log10, xlabel="Wavenumber K",
                       xticks=([π/16, π/8, π/4, π/2, π], ["π/16", "π/8", "π/4", "π/2", "π"]))

function field_panel!(position, field, title; colorrange, colormap=:balance)
    ax = Axis(position; title, aspect=DataAspect())
    hidedecorations!(ax)
    return heatmap!(ax, field; colormap, colorrange)
end

#####
##### Adjustment from white-noise velocity
#####

adjustment = cases(load_experiment("adjustment"))
window = (10day, 20day)
v̄ = Dict(name => time_mean(output.snapshots, :v, window) for (name, output) in adjustment)
ū = Dict(name => time_mean(output.snapshots, :u, window) for (name, output) in adjustment)
E₀ = Dict(name => kinetic_energy(output.snapshots[1]) for (name, output) in adjustment)
retained = Dict(name => (mean(abs2, ū[name]) + mean(abs2, v̄[name])) / 2 / E₀[name] for (name, _) in adjustment)

fig = Figure(size=(1150, 800), fontsize=15)
range_v = maximum(abs, v̄["Four-point"]) / 2
heatmaps = [field_panel!(fig[1, column], v̄[name], @sprintf("%s: time-mean v, K̄/E₀ = %.1e", name, retained[name]);
                        colorrange=(-range_v, range_v)) for (column, (name, _)) in enumerate(adjustment)]
Colorbar(fig[1, 4], last(heatmaps); label="v [m s⁻¹]")

ax = Axis(fig[2, 1]; title="(d) Kinetic energy", xlabel="Time [days]", ylabel="K / E₀ (1-day mean)", yscale=log10)
for (name, output) in adjustment
    K = kinetic_energy.(output.snapshots) ./ E₀[name]
    t = [s.t for s in output.snapshots] ./ day
    lines!(ax, t, running_mean(K, 24); color=colors[name], label=name)
end
axislegend(ax; position=:rt)

for (column, (title, fields, dims)) in enumerate((("(e) Time-mean v along x", v̄, 1), ("(f) Time-mean u along y", ū, 2)))
    panel = Axis(fig[2, column + 1]; title, ylabel="Power [m² s⁻²]", spectrum_axis...)
    for (name, _) in adjustment
        power = directional_spectrum(fields[name][:, :, 1], dims)
        lines!(panel, index_wavenumbers(size(fields[name], 1))[2:end], power[2:end]; color=colors[name], label=name)
    end
end
save(joinpath(output, "periodic_adjustment.pdf"), fig)
save(joinpath(preview, "periodic_adjustment.png"), fig)

#####
##### Decaying turbulence
#####

decaying = cases(load_experiment("decaying"))
fig = Figure(size=(1150, 1150), fontsize=15)
for (row, snapshot_day) in enumerate((30, 180))
    for (column, (name, output)) in enumerate(decaying)
        s = output.snapshots[snapshot_day + 1]
        hm = field_panel!(fig[row, column], s.ζ[:, :, 1] ./ f₀, "$name: ζ/f, day $snapshot_day"; colorrange=(-0.5, 0.5))
        column == 3 && Colorbar(fig[row, 4], hm; label="ζ / f")
    end
end

ax = Axis(fig[3, 1]; title="(g) Kinetic energy spectrum", ylabel="Energy density [m² s⁻²]", spectrum_axis...)
for (name, output) in decaying, (snapshot_day, linestyle) in ((30, :solid), (180, :dash))
    s = output.snapshots[snapshot_day + 1]
    K, spectrum = isotropic_spectrum(s.u[:, :, 1], s.v[:, :, 1])
    lines!(ax, K[2:end], spectrum[2:end]; color=colors[name], linestyle, label=snapshot_day == 30 ? name : nothing)
end
axislegend(ax; position=:lb)
for (column, (title, quantity)) in enumerate((("(h) Kinetic energy K/K₀", kinetic_energy), ("(i) Enstrophy Z/Z₀", enstrophy)))
    panel = Axis(fig[3, column + 1]; title, xlabel="Time [days]", yscale=log10)
    for (name, output) in decaying
        q = quantity.(output.snapshots)
        lines!(panel, [s.t for s in output.snapshots] ./ day, q ./ q[1]; color=colors[name])
    end
end
save(joinpath(output, "periodic_decaying.pdf"), fig)
save(joinpath(preview, "periodic_decaying.png"), fig)

#####
##### Forced turbulence with drag
#####

forced = cases(load_experiment("forced"))
window = (300day, 360day)
fig = Figure(size=(1150, 420), fontsize=15)
for (column, (title, variable, unit)) in enumerate((("(a) Time-mean v along x", :v, "Power [m² s⁻²]"),
                                                    ("(b) Time-mean u along y", :u, "Power [m² s⁻²]"),
                                                    ("(c) Time-mean η along x", :η, "Power [m²]")))
    panel = Axis(fig[1, column]; title, ylabel=unit, spectrum_axis...)
    for (name, output) in forced
        field = time_mean(output.snapshots, variable, window)[:, :, 1]
        power = directional_spectrum(field, variable == :u ? 2 : 1)
        lines!(panel, index_wavenumbers(size(field, 1))[2:end], power[2:end]; color=colors[name], label=name)
    end
    column == 1 && axislegend(panel; position=:lb)
end
save(joinpath(output, "periodic_forced.pdf"), fig)
save(joinpath(preview, "periodic_forced.png"), fig)

#####
##### Budgets
#####

trapezoid(t, rate) = sum((rate[n+1] + rate[n]) / 2 * (t[n+1] - t[n]) for n in 1:length(t)-1)

function cumulative_terms(history)
    t = history.time
    integral(key, rates) = trapezoid(t, rates[key])
    result = Dict{Symbol, NTuple{2, Float64}}()
    for (term, keys) in ((:coriolis, (:coriolis,)), (:pressure, (:barotropic_pressure, :baroclinic_pressure)),
                         (:advection, (:advection,)), (:forcing, (:remainder, :boundary_fluxes)))
        result[term] = (sum(integral(k, history.energy_rates) for k in keys), sum(integral(k, history.enstrophy_rates) for k in keys))
    end
    ΔK = history.kinetic_energy[end] - history.kinetic_energy[1]
    ΔZ = history.enstrophy[end] - history.enstrophy[1]
    result[:residual] = (ΔK - sum(first(v) for v in values(result)), ΔZ - sum(last(v) for v in values(result)))
    return result
end

term_labels = (coriolis = "Coriolis", pressure = "Pressure", advection = "Advection", residual = "Residual")
fig = Figure(size=(1150, 750), fontsize=15)
for (row, (experiment, outputs)) in enumerate((("Adjustment", adjustment), ("Decaying turbulence", decaying)))
    for (column, (quantity, index)) in enumerate((("kinetic energy / K₀", 1), ("enstrophy / Z₀", 2)))
        panel = Axis(fig[row, column]; title="$experiment: change of $quantity by term",
                  xticks=(1:4, collect(values(term_labels))), ylabel="Cumulative contribution")
        positions, heights, groups = Int[], Float64[], Int[]
        for (g, (name, output)) in enumerate(outputs)
            terms = cumulative_terms(output.budget)
            scale = index == 1 ? output.budget.kinetic_energy[1] : output.budget.enstrophy[1]
            for (p, key) in enumerate(keys(term_labels))
                push!(positions, p); push!(heights, terms[key][index] / scale); push!(groups, g)
            end
        end
        barplot!(panel, positions, heights; dodge=groups, color=[colors[outputs[g][1]] for g in groups])
        hlines!(panel, 0; color=:black, linewidth=0.5)
    end
end
Legend(fig[3, 1:2], [PolyElement(color=colors[name]) for (name, _) in adjustment], [name for (name, _) in adjustment]; orientation=:horizontal)
save(joinpath(output, "periodic_budgets.pdf"), fig)
save(joinpath(preview, "periodic_budgets.png"), fig)

for (experiment, outputs) in (("adjustment", adjustment), ("decaying", decaying))
    for (name, output) in outputs
        terms = cumulative_terms(output.budget)
        K₀, Z₀ = output.budget.kinetic_energy[1], output.budget.enstrophy[1]
        @info @sprintf("%-10s %-10s ", experiment, name) * join([@sprintf("%s K %+.2e Z %+.2e", term_labels[k], terms[k][1] / K₀, terms[k][2] / Z₀) for k in keys(term_labels)], "  ")
    end
end
