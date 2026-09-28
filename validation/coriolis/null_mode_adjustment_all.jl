# Figure 3 (fate of the null mode u = (-1)ʲ) with every Coriolis option: four-point, C-D, and the oriented stencils
# (current, smaller chirality, fourth-order four-corner, and the two optimized 4 × 4 stencils).

using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: EnergyConserving
using Oceananigans.Coriolis: CDScheme
using Statistics
using Printf
using CairoMakie

include("/Users/simonesilvestri/development/temp/Oceananigans.jl/prs/cd-coriolis-scheme/validation/coriolis/oriented_coriolis.jl")

const f = 1e-4
const N = 16
const Δ = 50kilometers
const H = 1000
const U₀ = 0.1

fourth_order_weights = Dict((0.5, -0.5) => 1/16, (-0.5, 0.5) => -3/16, (0.5, 0.5) => 0.0, (-0.5, -0.5) => 0.0)
wide_weights = Dict((-1.5, -1.5) => +0.0176, (-1.5, -0.5) => +0.0186, (-1.5, 0.5) => +0.0266, (-1.5, 1.5) => +0.0140,
                    (-0.5, -1.5) => +0.0070, (-0.5, -0.5) => -0.0356, (-0.5, 0.5) => -0.1499, (-0.5, 1.5) => +0.0266,
                    (0.5, -1.5) => +0.0176, (0.5, -0.5) => +0.0191, (0.5, 0.5) => -0.0356, (0.5, 1.5) => +0.0186,
                    (1.5, -1.5) => +0.0134, (1.5, -0.5) => +0.0176, (1.5, 0.5) => +0.0070, (1.5, 1.5) => +0.0176)
wide_monopole_weights = Dict((-1.5, -1.5) => +0.0113, (-1.5, -0.5) => +0.0028, (-1.5, 0.5) => +0.0176, (-1.5, 1.5) => +0.0057,
                             (-0.5, -1.5) => -0.0220, (-0.5, -0.5) => -0.0517, (-0.5, 0.5) => -0.1961, (-0.5, 1.5) => +0.0176,
                             (0.5, -1.5) => -0.0056, (0.5, -0.5) => +0.0230, (0.5, 0.5) => -0.0517, (0.5, 1.5) => +0.0028,
                             (1.5, -1.5) => +0.0015, (1.5, -0.5) => -0.0056, (1.5, 0.5) => -0.0220, (1.5, 1.5) => +0.0113)

oriented = ("Oriented ε=1/8" => OrientedCoriolis(),
            "Oriented ε=1/32" => OrientedCoriolis(; ε=1/32, η=1/32),
            "Fourth-order oriented" => OrientedCoriolis(; weights=fourth_order_weights),
            "Wide" => OrientedCoriolis(; weights=wide_weights),
            "Wide + monopole" => OrientedCoriolis(; weights=wide_monopole_weights))

schemes = ("Four-point" => grid -> EnergyConserving(), "C-D" => grid -> CDScheme(grid),
           [name => (grid -> scheme) for (name, scheme) in oriented]...)
colors = Dict("Four-point" => :royalblue, "C-D" => :darkorange, "Oriented ε=1/8" => :crimson, "Oriented ε=1/32" => :lightsalmon,
              "Fourth-order oriented" => :forestgreen, "Wide" => :purple, "Wide + monopole" => :deeppink)

# m = cos(K/2) cos(L/2) - σ² ν*, ν = Σ w exp(i((a - 1/2) K + (b - 1/2) L)); the pattern sits at (K, L) = (0, π)
function symbol(scheme::OrientedCoriolis, K, L)
    ν = sum(w * exp(im * ((a - 1/2) * K + (b - 1/2) * L)) for (w, (a, b)) in zip(scheme.weights, corner_offsets))
    return cos(K / 2) * cos(L / 2) - (4 * sin(K / 2)^2 + 4 * sin(L / 2)^2) * conj(ν)
end

pattern_coupling = Dict(name => abs2(symbol(scheme, 0, π)) for (name, scheme) in oriented)
balanced_amplitude(m², deformation_radius) = 4 / (4 + m² * (Δ / deformation_radius)^2)
continuum_amplitude(deformation_radius) = π^2 / (π^2 + (Δ / deformation_radius)^2)
continuum_frequency(deformation_radius) = sqrt(1 + π^2 * (deformation_radius / Δ)^2)
discrete_frequency(m², deformation_radius) = sqrt(m² + 4 * (deformation_radius / Δ)^2)

function continuum_solution(t, deformation_radius)
    a = continuum_amplitude(deformation_radius)
    return a + (1 - a) * cos(continuum_frequency(deformation_radius) * f * t)
end

function continuum_energies(t, deformation_radius)
    a = continuum_amplitude(deformation_radius)
    ωt = continuum_frequency(deformation_radius) * f * t
    return continuum_solution(t, deformation_radius)^2 + (1 - a) * sin(ωt)^2, a * (1 - a) * (1 - cos(ωt))^2
end

function null_mode_run(scheme_constructor; deformation_radius, stop_time=10days, Δt=5minutes, save_interval=1hour)
    grid = RectilinearGrid(size=(N, N, 1), x=(0, N * Δ), y=(0, N * Δ), z=(-H, 0), topology=(Periodic, Periodic, Bounded))
    g = (f * deformation_radius)^2 / H
    model = HydrostaticFreeSurfaceModel(grid; coriolis=FPlane(; f, scheme=scheme_constructor(grid)), momentum_advection=nothing,
                                        free_surface=ExplicitFreeSurface(gravitational_acceleration=g), buoyancy=nothing,
                                        tracers=(), closure=nothing, timestepper=:SplitRungeKutta3)
    pattern = [(-1)^j for i in 1:N, j in 1:N, k in 1:1]
    set!(model, u=U₀ .* pattern)
    u, v, η = model.velocities.u, model.velocities.v, model.free_surface.displacement
    E₀ = H * U₀^2 / 2
    d_grid_energy(model) = model.coriolis.scheme isa CDScheme ?
        H * (mean(abs2, interior(model.coriolis.scheme.uᴰ)) + mean(abs2, interior(model.coriolis.scheme.vᴰ))) / 2 : 0.0
    history = (t = Float64[], a = Float64[], K = Float64[], P = Float64[], KD = Float64[])
    snapshots = []
    steps_per_save = round(Int, save_interval / Δt)
    for n in 0:round(Int, stop_time / Δt)
        n > 0 && time_step!(model, Δt)
        push!(history.t, model.clock.time)
        push!(history.a, mean(interior(u) .* pattern) / U₀)
        push!(history.K, H * (mean(abs2, interior(u)) + mean(abs2, interior(v))) / 2 / E₀)
        push!(history.P, g * mean(abs2, interior(η)) / 2 / E₀)
        push!(history.KD, d_grid_energy(model) / E₀)
        n % steps_per_save == 0 && push!(snapshots, (t = model.clock.time, u = Array(interior(u, :, :, 1)), η = Array(interior(η, :, :, 1))))
    end
    return (; history, snapshots)
end

regimes = (Δ / 4, 4Δ)
runs = Dict((name, R) => null_mode_run(scheme; deformation_radius=R) for (name, scheme) in schemes, R in regimes)

oscillation_frequency(name, R) = haskey(pattern_coupling, name) ? discrete_frequency(pattern_coupling[name], R) * f : f
averaging_window(name, R) = (10days - 10 * 2π / oscillation_frequency(name, R), 10days)
time_mean_amplitude(h, name, R) = mean(h.a[averaging_window(name, R)[1] .≤ h.t .≤ averaging_window(name, R)[2]])
predicted(name, R) = name == "Four-point" ? 1.0 : name == "C-D" ? 0.0 : name == "Continuum" ? continuum_amplitude(R) :
                     balanced_amplitude(pattern_coupling[name], R)

for R in regimes
    @info @sprintf("R_d = %.2f Δ: continuum balanced amplitude %.3f", R / Δ, continuum_amplitude(R))
    for (name, _) in schemes
        h = runs[(name, R)].history
        m = haskey(pattern_coupling, name) ? @sprintf("|m(0,π)| %.3f", sqrt(pattern_coupling[name])) : ""
        @info @sprintf("   %-22s time-mean %+.3f predicted %+.3f  energy change %+.1e  %s", name, time_mean_amplitude(h, name, R),
                       predicted(name, R), h.K[end] + h.P[end] + h.KD[end] - 1, m)
    end
end

fig = Figure(size=(1700, 1750), fontsize=15)
amplitude_axis = Ref{Any}(nothing)
for (column, R) in enumerate(regimes)
    ax = Axis(fig[1, 2column-1:2column]; title=@sprintf("Amplitude of u = (−1)ʲ, R_d = %s", R < Δ ? "Δ/4" : "4Δ"),
              xlabel="Time [inertial periods]", ylabel="a(t)", limits=(0, 4, -1.1, 1.15))
    column == 1 && (amplitude_axis[] = ax)
    t = range(0, 4 * 2π / f, length=8000)
    lines!(ax, t .* f ./ 2π, continuum_solution.(t, R); color=:black, linewidth=2.5, label="Continuum")
    for (name, _) in schemes
        h = runs[(name, R)].history
        lines!(ax, h.t .* f ./ 2π, h.a; color=colors[name], label=name)
    end
    hlines!(ax, [continuum_amplitude(R)]; color=:black, linestyle=:dash, label="Continuum balance")
    if R > Δ
        inset = Axis(fig[1, 2column-1:2column]; width=Relative(0.52), height=Relative(0.4), halign=0.93, valign=0.1,
                     backgroundcolor=:white, limits=(0, 1, 0.955, 1.005), title="First inertial period", titlesize=12,
                     xticklabelsize=10, yticklabelsize=10)
        translate!(inset.blockscene, 0, 0, 150)
        lines!(inset, t .* f ./ 2π, continuum_solution.(t, R); color=:black, linewidth=2.5)
        for (name, _) in schemes
            name == "C-D" && continue
            h = runs[(name, R)].history
            lines!(inset, h.t .* f ./ 2π, h.a; color=colors[name])
        end
        hlines!(inset, [continuum_amplitude(R)]; color=:black, linestyle=:dash)
    end
end
Legend(fig[2, 1:4], amplitude_axis[]; orientation=:horizontal, framevisible=false, nbanks=2)

R = Δ / 4
energy_panels = ("Continuum", first.(schemes)...)
for (n, name) in enumerate(energy_panels)
    row, column = 3 + (n - 1) ÷ 4, mod1(n, 4)
    ax = Axis(fig[row, column]; title="$name, R_d = Δ/4", xlabel=row == 4 ? "Time [inertial periods]" : "",
              ylabel=column == 1 ? "E / E₀" : "", limits=(0, 4, -0.05, 1.1))
    if name == "Continuum"
        t = range(0, 4 * 2π / f, length=4000)
        energies = continuum_energies.(t, R)
        K, P, KD, times = first.(energies), last.(energies), zeros(length(t)), t
    else
        h = runs[(name, R)].history
        K, P, KD, times = h.K, h.P, h.KD, h.t
    end
    lines!(ax, times .* f ./ 2π, K; color=:steelblue)
    lines!(ax, times .* f ./ 2π, P; color=:goldenrod)
    name == "C-D" && lines!(ax, times .* f ./ 2π, KD; color=:mediumpurple)
    lines!(ax, times .* f ./ 2π, K .+ P .+ KD; color=:black)
end
elements = [LineElement(color=c) for c in (:steelblue, :mediumpurple, :goldenrod, :black)]
Legend(fig[5, 1:4], elements, ["Kinetic (C grid)", "Kinetic (D grid)", "Potential", "Total"]; orientation=:horizontal, framevisible=false)

ax = Axis(fig[6, 1:4]; title="Time-mean amplitude over ten oscillation periods", xticks=(1:2, ["R_d = Δ/4", "R_d = 4Δ"]), ylabel="ā")
bars = (first.(schemes)..., "Continuum")
width = 0.8 / length(bars)
for (g, name) in enumerate(bars), (p, R) in enumerate(regimes)
    x = p + (g - (length(bars) + 1) / 2) * width
    value = name == "Continuum" ? continuum_amplitude(R) : time_mean_amplitude(runs[(name, R)].history, name, R)
    barplot!(ax, [x], [value]; width=0.9width, color=name == "Continuum" ? :gray40 : colors[name])
    scatter!(ax, [x], [predicted(name, R)]; color=:black, marker=:hline, markersize=18)
end
legend_elements = vcat([PolyElement(color=name == "Continuum" ? :gray40 : colors[name]) for name in bars],
                       [MarkerElement(marker=:hline, color=:black, markersize=18)])
Legend(fig[7, 1:4], legend_elements, [collect(bars); "Prediction"]; orientation=:horizontal, framevisible=false, nbanks=2)
rowsize!(fig.layout, 1, Relative(0.27))
rowsize!(fig.layout, 3, Relative(0.17))
rowsize!(fig.layout, 4, Relative(0.17))
rowsize!(fig.layout, 6, Relative(0.18))

output = "/private/tmp/claude-501/-Users-simonesilvestri-development-temp-Oceananigans-jl/5769e5fd-9a00-4281-94a2-2324549baf58/scratchpad/figures_preview"
save(joinpath(output, "null_mode_adjustment_all.pdf"), fig)
save(joinpath(output, "null_mode_adjustment_all.png"), fig)

movie = Figure(size=(2000, 650), fontsize=14)
frame = Observable(1)
reference = runs[("Four-point", R)].snapshots
Label(movie[0, 1:length(schemes)], lift(n -> @sprintf("u = 0.1 (−1)ʲ at t = 0, R_d = Δ/4, t = %.1f inertial periods", reference[n].t * f / 2π), frame); fontsize=18)
ηmax = maximum(maximum(abs, x.η) for x in runs[("Oriented ε=1/8", R)].snapshots)
for (column, (name, _)) in enumerate(schemes)
    s = runs[(name, R)].snapshots
    ax = Axis(movie[1, column]; title="$name: u", aspect=DataAspect(), titlesize=12); hidedecorations!(ax)
    heatmap!(ax, lift(n -> s[n].u, frame); colormap=:balance, colorrange=(-U₀, U₀))
    ax = Axis(movie[2, column]; title="$name: η", aspect=DataAspect(), titlesize=12); hidedecorations!(ax)
    heatmap!(ax, lift(n -> s[n].η, frame); colormap=:balance, colorrange=(-ηmax, ηmax))
end
record(movie, joinpath(output, "null_mode_adjustment_all.mp4"), eachindex(reference); framerate=12) do n
    frame[] = n
end
