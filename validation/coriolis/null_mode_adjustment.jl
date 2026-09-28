# Fate of the null mode u = (-1)ʲ of the four-point average on a doubly periodic f-plane, with linear dynamics and no
# dissipation. The mode carries grid-scale vorticity, hence potential vorticity, which no conserving scheme can remove:
# potential-vorticity conservation fixes the balanced part of the final state,
#
#   a = σ² / (σ² + |m|² Δ² / R_d²),   σ² = 4 at (K, L) = (0, π),
#
# which is 1 for the four-point average (|m| = 0) and 4 / (4 + Δ²/R_d²) for the oriented scheme (|m| = 1). The rest
# oscillates as an inertia–gravity wave with zero group velocity. The continuum reference is the exact solution of the
# linear shallow-water equations for u = U₀ cos(πy/Δ), the field that the pattern samples:
#
#   u = U₀ [a + (1 - a) cos(ωt)] cos(πy/Δ),   a = π² / (π² + Δ²/R_d²),   ω² = f² + g H π²/Δ².
#
# In the C-D scheme u exchanges energy with the D-grid velocity at the inertial frequency and has no balanced part.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: EnergyConserving
using Oceananigans.Coriolis: CDScheme
using Statistics
using Printf
using CairoMakie

include("oriented_coriolis.jl")

const f = 1e-4
const N = 16
const Δ = 50kilometers
const H = 1000
const U₀ = 0.1

schemes = ("Four-point" => grid -> EnergyConserving(), "C-D" => grid -> CDScheme(grid), "Oriented" => grid -> OrientedCoriolis())
colors = Dict("Four-point" => :royalblue, "C-D" => :darkorange, "Oriented" => :crimson)

balanced_amplitude(m², deformation_radius) = 4 / (4 + m² * (Δ / deformation_radius)^2)
continuum_amplitude(deformation_radius) = π^2 / (π^2 + (Δ / deformation_radius)^2)
continuum_frequency(deformation_radius) = sqrt(1 + π^2 * (deformation_radius / Δ)^2)

function continuum_solution(t, deformation_radius)
    a = continuum_amplitude(deformation_radius)
    return a + (1 - a) * cos(continuum_frequency(deformation_radius) * f * t)
end

# Kinetic and potential energy of the exact solution, normalized by the initial energy
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
    amplitude(model) = mean(interior(u) .* pattern) / U₀
    kinetic_energy(model) = H * (mean(abs2, interior(u)) + mean(abs2, interior(v))) / 2
    potential_energy(model) = g * mean(abs2, interior(η)) / 2
    E₀ = H * U₀^2 / 2

    d_grid_energy(model) = model.coriolis.scheme isa CDScheme ?
        H * (mean(abs2, interior(model.coriolis.scheme.uᴰ)) + mean(abs2, interior(model.coriolis.scheme.vᴰ))) / 2 : 0.0

    history = (t = Float64[], a = Float64[], K = Float64[], P = Float64[], KD = Float64[])
    snapshots = []
    steps_per_save = round(Int, save_interval / Δt)
    for n in 0:round(Int, stop_time / Δt)
        n > 0 && time_step!(model, Δt)
        push!(history.t, model.clock.time); push!(history.a, amplitude(model))
        push!(history.K, kinetic_energy(model) / E₀); push!(history.P, potential_energy(model) / E₀)
        push!(history.KD, d_grid_energy(model) / E₀)
        n % steps_per_save == 0 && push!(snapshots, (t = model.clock.time, u = Array(interior(u, :, :, 1)), η = Array(interior(η, :, :, 1))))
    end
    return (; history, snapshots)
end

regimes = (Δ / 4, 4Δ)
runs = Dict((name, R) => null_mode_run(scheme; deformation_radius=R) for (name, scheme) in schemes, R in regimes)

oriented_frequency(R) = sqrt(1 + 4 * (R / Δ)^2)

# Each scheme is averaged over the last ten periods of its own oscillation: the inertia–gravity wave of the oriented
# scheme and the inertial exchange with the D-grid velocity of the C-D scheme
oscillation_frequency(name, R) = name == "Oriented" ? oriented_frequency(R) * f : f
averaging_window(name, R) = (10days - 10 * 2π / oscillation_frequency(name, R), 10days)
time_mean_amplitude(h, name, R) = mean(h.a[averaging_window(name, R)[1] .≤ h.t .≤ averaging_window(name, R)[2]])
for R in regimes
    @info @sprintf("R_d = %.2f Δ: predicted balanced amplitude four-point 1, oriented %.3f, continuum %.3f; ω/f oriented %.3f, continuum %.3f",
                   R / Δ, balanced_amplitude(1, R), continuum_amplitude(R), oriented_frequency(R), continuum_frequency(R))
    for (name, _) in schemes
        h = runs[(name, R)].history
        late = h.t .≥ averaging_window(name, R)[1]
        @info @sprintf("   %-10s time-mean amplitude over ten oscillation periods %+.3f, range [%+.3f, %+.3f]",
                       name, time_mean_amplitude(h, name, R), minimum(h.a[late]), maximum(h.a[late]))
        name == "C-D" || @info @sprintf("   %-10s relative change of the energy after 10 days %+.1e", name, h.K[end] + h.P[end] - 1)
    end
end

fig = Figure(size=(1500, 1400), fontsize=15)
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
    hlines!(ax, [balanced_amplitude(1, R)]; color=colors["Oriented"], linestyle=:dash, label="Oriented balance")
    hlines!(ax, [continuum_amplitude(R)]; color=:black, linestyle=:dash, label="Continuum balance")
    if R > Δ
        inset = Axis(fig[1, 2column-1:2column]; width=Relative(0.52), height=Relative(0.4), halign=0.93, valign=0.1,
                     backgroundcolor=:white, limits=(0, 1, 0.965, 1.005), title="First inertial period", titlesize=12,
                     xticklabelsize=10, yticklabelsize=10)
        translate!(inset.blockscene, 0, 0, 150)
        lines!(inset, t .* f ./ 2π, continuum_solution.(t, R); color=:black, linewidth=2.5)
        for (name, _) in schemes
            name == "C-D" && continue
            h = runs[(name, R)].history
            lines!(inset, h.t .* f ./ 2π, h.a; color=colors[name])
        end
        hlines!(inset, [balanced_amplitude(1, R)]; color=colors["Oriented"], linestyle=:dash)
        hlines!(inset, [continuum_amplitude(R)]; color=:black, linestyle=:dash)
    end
end
Legend(fig[2, 1:4], amplitude_axis[]; orientation=:horizontal, framevisible=false)

R = Δ / 4
energy_panels = ("Continuum", "Four-point", "Oriented", "C-D")
for (column, name) in enumerate(energy_panels)
    ax = Axis(fig[3, column]; title="$name, R_d = Δ/4", xlabel="Time [inertial periods]", ylabel=column == 1 ? "E / E₀" : "",
              limits=(0, 4, -0.05, 1.1))
    if name == "Continuum"
        t = range(0, 4 * 2π / f, length=4000)
        energies = continuum_energies.(t, R)
        K, P, KD, times = first.(energies), last.(energies), zeros(length(t)), t
    else
        h = runs[(name, R)].history
        K, P, KD, times = h.K, h.P, h.KD, h.t
    end
    lines!(ax, times .* f ./ 2π, K; color=:steelblue, label="Kinetic (C grid)")
    lines!(ax, times .* f ./ 2π, P; color=:goldenrod, label="Potential")
    name == "C-D" && lines!(ax, times .* f ./ 2π, KD; color=:mediumpurple, label="Kinetic (D grid)")
    lines!(ax, times .* f ./ 2π, K .+ P .+ KD; color=:black, label="Total")
end
elements = [LineElement(color=c) for c in (:steelblue, :mediumpurple, :goldenrod, :black)]
Legend(fig[4, 1:4], elements, ["Kinetic (C grid)", "Kinetic (D grid)", "Potential", "Total"]; orientation=:horizontal, framevisible=false)

ax = Axis(fig[5, 1:4]; title="Time-mean amplitude over ten oscillation periods", xticks=(1:2, ["R_d = Δ/4", "R_d = 4Δ"]), ylabel="ā")
predicted = Dict("Four-point" => R -> 1.0, "C-D" => R -> 0.0, "Oriented" => R -> balanced_amplitude(1, R), "Continuum" => continuum_amplitude)
bars = (schemes..., "Continuum" => nothing)
for (g, (name, _)) in enumerate(bars), (p, R) in enumerate(regimes)
    x = p + (g - 2.5) * 0.2
    value = name == "Continuum" ? continuum_amplitude(R) : time_mean_amplitude(runs[(name, R)].history, name, R)
    barplot!(ax, [x], [value]; width=0.18, color=name == "Continuum" ? :gray40 : colors[name])
    scatter!(ax, [x], [predicted[name](R)]; color=:black, marker=:hline, markersize=22)
end
legend_elements = vcat([PolyElement(color=name == "Continuum" ? :gray40 : colors[name]) for (name, _) in bars],
                       [MarkerElement(marker=:hline, color=:black, markersize=22)])
axislegend(ax, legend_elements, [[name for (name, _) in bars]; "Prediction"]; position=(0.5, 0.5))
rowsize!(fig.layout, 1, Relative(0.33))
rowsize!(fig.layout, 3, Relative(0.27))
rowsize!(fig.layout, 5, Relative(0.22))

output = get(ENV, "FIGURE_DIRECTORY", ".")
save(joinpath(output, "null_mode_adjustment.pdf"), fig)
save(joinpath(get(ENV, "PREVIEW_DIRECTORY", output), "null_mode_adjustment.png"), fig)

movie = Figure(size=(1200, 700), fontsize=15)
frame = Observable(1)
R = Δ / 4
reference = runs[("Four-point", R)].snapshots
Label(movie[0, 1:3], @lift(@sprintf("u = 0.1 (−1)ʲ at t = 0, R_d = Δ/4, t = %.1f inertial periods", reference[$frame].t * f / 2π)); fontsize=18)
for (column, (name, _)) in enumerate(schemes)
    s = runs[(name, R)].snapshots
    ax = Axis(movie[1, column]; title="$name: u", aspect=DataAspect()); hidedecorations!(ax)
    heatmap!(ax, @lift(s[$frame].u); colormap=:balance, colorrange=(-U₀, U₀))
    ax = Axis(movie[2, column]; title="$name: η", aspect=DataAspect()); hidedecorations!(ax)
    ηmax = maximum(maximum(abs, x.η) for x in runs[("Oriented", R)].snapshots)
    heatmap!(ax, @lift(s[$frame].η); colormap=:balance, colorrange=(-ηmax, ηmax))
end
record(movie, joinpath(get(ENV, "MOVIE_DIRECTORY", output), "null_mode_adjustment.mp4"), eachindex(reference); framerate=12) do n
    frame[] = n
end
