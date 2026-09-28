# Nonlinear scores of C-grid Coriolis schemes on a doubly periodic, flat f-plane with a deformation radius of a
# quarter cell, the regime of the 1° gyres example. Three experiments, each with a movie:
#
#   1. adjustment: linear dynamics from white-noise velocity; the time-mean flow after adjustment is the
#      geostrophic part, which a null mode keeps entirely at the grid scale.
#   2. decaying:   freely decaying turbulence with WENO vector-invariant advection and no viscosity.
#   3. forced:     a steady body force that is white noise in space, balanced by linear drag, with WENO
#                  advection; the null-mode part of the force accumulates until the drag stops it (eq. 21).
#
# Usage: julia --project validation/coriolis/oriented_coriolis_periodic_experiments.jl [adjustment|decaying|forced]

using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: EnstrophyConserving, EnergyConserving
using Oceananigans.Operators: ζ₃ᶠᶠᶜ
using Random
using Statistics
using Printf
using CairoMakie

include("oriented_coriolis.jl")
include("momentum_budgets.jl")

using Oceananigans.Coriolis: CDScheme
using JLD2

const f₀ = 1e-4
const N = 96
const Δ = 50kilometers
const H = 1000
const deformation_radius = Δ / 4
const reduced_gravity = (f₀ * deformation_radius)^2 / H

schemes = ("EnstrophyConserving" => grid -> EnstrophyConserving(),
           "EnergyConserving"    => grid -> EnergyConserving(),
           "CDScheme"            => grid -> CDScheme(grid),
           "OrientedCoriolis"      => grid -> OrientedCoriolis())

@inline drag_and_body_force_u(i, j, k, grid, clock, fields, p) = @inbounds - p.r * fields.u[i, j, k] + p.Fu[i, j]
@inline drag_and_body_force_v(i, j, k, grid, clock, fields, p) = @inbounds - p.r * fields.v[i, j, k] + p.Fv[i, j]

function build_model(scheme_constructor; advection, forced=false)
    grid = RectilinearGrid(size=(N, N, 1), x=(0, N*Δ), y=(0, N*Δ), z=(-H, 0),
                           topology=(Periodic, Periodic, Bounded), halo=(5, 5, 5))

    forcing = if forced
        Random.seed!(4)
        F = 0.1 / 30days
        parameters = (r = 1 / 30days, Fu = F .* randn(N, N), Fv = F .* randn(N, N))
        (u = Forcing(drag_and_body_force_u; discrete_form=true, parameters),
         v = Forcing(drag_and_body_force_v; discrete_form=true, parameters))
    else
        NamedTuple()
    end

    return HydrostaticFreeSurfaceModel(grid; coriolis=FPlane(f=f₀, scheme=scheme_constructor(grid)), timestepper=:SplitRungeKutta3,
                                       momentum_advection=advection, forcing,
                                       free_surface=ExplicitFreeSurface(gravitational_acceleration=reduced_gravity),
                                       buoyancy=nothing, tracers=nothing, closure=nothing)
end

function random_streamfunction_velocities(; seed, wavenumbers, amplitude)
    Random.seed!(seed)
    ψ = zeros(N, N)
    for n in 1:60
        kx, ky, phase = rand(wavenumbers), rand(wavenumbers), 2π * rand()
        ψ .+= [sin(2π * (kx * i + ky * j) / N + phase) for i in 1:N, j in 1:N] ./ (kx^2 + ky^2)
    end
    ψ .*= amplitude * Δ / maximum(abs, diff(ψ, dims=1))
    u = - (circshift(ψ, (0, -1)) .- ψ) ./ Δ
    v =   (circshift(ψ, (-1, 0)) .- ψ) ./ Δ
    return reshape(u, N, N, 1), reshape(v, N, N, 1)
end

grid_scale_fraction(a, shift) = sqrt(mean(abs2, (circshift(a, shift) .- 2a .+ circshift(a, .-shift)) ./ 4) / mean(abs2, a))
γx(a) = grid_scale_fraction(a, (1, 0))
γy(a) = grid_scale_fraction(a, (0, 1))

function snapshot(model, ζ)
    compute!(ζ)
    return (u = Array(interior(model.velocities.u, :, :, 1)),
            v = Array(interior(model.velocities.v, :, :, 1)),
            η = Array(interior(model.free_surface.displacement, :, :, 1)),
            ζ = Array(interior(ζ, :, :, 1)),
            t = model.clock.time)
end

kinetic_energy(s) = (mean(abs2, s.u) + mean(abs2, s.v)) / 2
potential_energy(s) = reduced_gravity * mean(abs2, s.η) / 2H
total_energy(s) = kinetic_energy(s) + potential_energy(s)
enstrophy(s) = mean(abs2, s.ζ)

function simulate(model; stop_time, Δt, save_interval)
    ζ = Field(KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, model.grid, model.velocities.u, model.velocities.v))
    snapshots = [snapshot(model, ζ)]
    budget = MomentumBudget(model)
    steps_per_save = round(Int, save_interval / Δt)
    for n in 1:round(Int, stop_time / Δt)
        time_step!(model, Δt)
        record_budget!(budget)
        n % steps_per_save == 0 && push!(snapshots, snapshot(model, ζ))
    end
    return (; snapshots, budget = budget.history)
end

function adjustment(scheme)
    model = build_model(scheme; advection=nothing)
    Random.seed!(3)
    set!(model, u=0.1 .* randn(N, N, 1), v=0.1 .* randn(N, N, 1))
    return simulate(model; stop_time=20days, Δt=10minutes, save_interval=1hour)
end

function decaying(scheme)
    model = build_model(scheme; advection=WENOVectorInvariant(order=5))
    u, v = random_streamfunction_velocities(seed=11, wavenumbers=4:16, amplitude=1)
    set!(model; u, v)
    return simulate(model; stop_time=180days, Δt=20minutes, save_interval=1day)
end

function forced(scheme)
    model = build_model(scheme; advection=WENOVectorInvariant(order=5), forced=true)
    return simulate(model; stop_time=360days, Δt=20minutes, save_interval=1day)
end

#####
##### Scores and movie
#####

function report(experiment, results)
    names = first.(schemes)
    @info "Experiment $experiment (N = $N, Δ = $(Δ/1e3) km, R_d = Δ/4)"
    for name in names
        s = results[name]
        first_snapshot, last_snapshot = s[1], s[end]
        E₀ = experiment == "forced" ? one(total_energy(first_snapshot)) : total_energy(first_snapshot)
        line = @sprintf("%-20s E/E₀ = %.3f  KE/E₀ = %.3f  γx(u) %.2f γy(u) %.2f γx(v) %.2f γy(v) %.2f γ(η) %.2f",
                        name, total_energy(last_snapshot) / E₀, kinetic_energy(last_snapshot) / E₀,
                        γx(last_snapshot.u), γy(last_snapshot.u), γx(last_snapshot.v), γy(last_snapshot.v),
                        (γx(last_snapshot.η) + γy(last_snapshot.η)) / 2)
        if experiment in ("adjustment", "forced")
            late = experiment == "adjustment" ? filter(x -> x.t ≥ 10days, s) : filter(x -> x.t ≥ 300days, s)
            ū, v̄ = mean(x.u for x in late), mean(x.v for x in late)
            mean_kinetic_energy = (mean(abs2, ū) + mean(abs2, v̄)) / 2
            line *= experiment == "adjustment" ?
                @sprintf("  retained geostrophic KE %.3f", mean_kinetic_energy / kinetic_energy(first_snapshot)) :
                @sprintf("  time-mean KE %.2e m² s⁻²", mean_kinetic_energy)
            line *= @sprintf("  γx(v̄) %.2f γy(ū) %.2f", γx(v̄), γy(ū))
        elseif experiment == "decaying"
            ν = [-(kinetic_energy(s[n+1]) - kinetic_energy(s[n])) / (s[n+1].t - s[n].t) /
                  ((enstrophy(s[n+1]) + enstrophy(s[n])) / 2) for n in 1:length(s)-1]
            line *= @sprintf("  ν_eff days 0-30 %.1f, 30-180 %.1f m² s⁻¹", mean(ν[1:30]), mean(ν[31:end]))
        end
        @info line
    end
    return nothing
end

# One-day running mean for the adjustment experiment, which filters the inertia–gravity waves
function displayed_snapshots(experiment, snapshots)
    experiment == "adjustment" || return snapshots
    window = 24
    return [(u = mean(x.u for x in snapshots[max(1, n-window+1):n]),
             v = mean(x.v for x in snapshots[max(1, n-window+1):n]),
             η = mean(x.η for x in snapshots[max(1, n-window+1):n]),
             ζ = mean(x.ζ for x in snapshots[max(1, n-window+1):n]),
             t = snapshots[n].t) for n in eachindex(snapshots)]
end

function movie(experiment, results)
    names = first.(schemes)
    shown = Dict(name => displayed_snapshots(experiment, results[name]) for name in names)
    reference = shown[names[1]]
    frames = experiment == "adjustment" ? (1:4:length(reference)) : eachindex(reference)
    n = Observable(first(frames))
    scale_snapshot = experiment == "decaying" ? reference[1] : reference[end]
    ζmax = maximum(abs, scale_snapshot.ζ) / f₀ / 2
    vmax = maximum(abs, scale_snapshot.v) / 2
    time_unit, time_label = experiment == "adjustment" ? (hour, "hours") : (day, "days")

    fig = Figure(size=(500 * length(names), 1150))
    Label(fig[0, 1:length(names)], @lift(@sprintf("%s, R_d = Δ/4, t = %d %s", experiment,
                                      round(Int, reference[$n].t / time_unit), time_label)); fontsize=22, tellwidth=false)
    for (column, name) in enumerate(names)
        averaging = experiment == "adjustment" ? " (1-day mean)" : ""
        ax = Axis(fig[1, column]; title=name * ", ζ / f" * averaging, aspect=DataAspect())
        hidedecorations!(ax)
        heatmap!(ax, @lift(shown[name][$n].ζ ./ f₀); colormap=:balance, colorrange=(-ζmax, ζmax))
        ax = Axis(fig[2, column]; title=name * ", v [m s⁻¹]" * averaging, aspect=DataAspect())
        hidedecorations!(ax)
        heatmap!(ax, @lift(shown[name][$n].v); colormap=:balance, colorrange=(-vmax, vmax))
    end
    energy_label = experiment == "forced" ? "E(t) [m² s⁻²]" : "E(t) / E(0)"
    ax1 = Axis(fig[3, 1:2]; xlabel="t [$time_label]", ylabel=energy_label, height=180)
    ax2 = Axis(fig[3, 3:length(names)]; xlabel="t [$time_label]", ylabel="γx(v)", height=180)
    for name in names
        s = results[name]
        t = [x.t / time_unit for x in s]
        E₀ = experiment == "forced" ? 1 : total_energy(s[1])
        lines!(ax1, t, [total_energy(x) / E₀ for x in s]; label=name)
        lines!(ax2, t, [γx(x.v) for x in shown[name]]; label=name)
    end
    vlines!(ax1, @lift(reference[$n].t / time_unit); color=:gray)
    vlines!(ax2, @lift(reference[$n].t / time_unit); color=:gray)
    axislegend(ax1; position=:lb)

    record(fig, "oriented_coriolis_$experiment.mp4", frames; framerate=12) do frame
        n[] = frame
    end
    return nothing
end

experiments = (adjustment = adjustment, decaying = decaying, forced = forced)
selected = isempty(ARGS) ? keys(experiments) : Symbol.(ARGS)

function report_budgets(experiment, outputs)
    @info "Budgets for $experiment: cumulative terms at the final time, relative to the initial K and Z"
    for (name, _) in schemes
        history = outputs[name].budget
        integrated = integrated_budget(history)
        K₀, Z₀ = history.kinetic_energy[1], history.enstrophy[1]
        terms = sort(collect(keys(integrated.energy)), by=string)
        line = @sprintf("%-20s", name)
        for term in terms
            line *= @sprintf("  %s K %+.2e Z %+.2e", term, integrated.energy[term][end] / K₀, integrated.enstrophy[term][end] / Z₀)
        end
        line *= @sprintf("  residual K %+.2e Z %+.2e", integrated.energy_residual[end] / K₀, integrated.enstrophy_residual[end] / Z₀)
        @info line
    end
    return nothing
end

for experiment in selected
    outputs = Dict(name => experiments[experiment](scheme) for (name, scheme) in schemes)
    results = Dict(name => output.snapshots for (name, output) in outputs)
    jldsave("oriented_coriolis_$(experiment).jld2"; outputs)
    report(string(experiment), results)
    report_budgets(string(experiment), outputs)
    movie(string(experiment), results)
end
