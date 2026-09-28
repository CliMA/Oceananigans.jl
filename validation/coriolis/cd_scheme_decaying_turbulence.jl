# Freely decaying turbulence on an f-plane with WENO vector-invariant advection and no explicit viscosity.
# The effective viscosity ν = -(dE/dt) / Z, with E the kinetic energy and Z the enstrophy, measures all
# implicit dissipation. The difference between the C-D scheme and EnstrophyConserving is the dissipation
# implied by the C-D scheme.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: EnstrophyConserving
using Oceananigans.Coriolis: CDScheme
using Oceananigans.Operators: ζ₃ᶠᶠᶜ
using Random
using Statistics
using Printf
using CairoMakie

function decaying_turbulence(scheme; N=96, Δ=50kilometers, stop_time=60days, Δt=20minutes, seed=11)
    grid = RectilinearGrid(size=(N, N, 1), x=(0, N*Δ), y=(0, N*Δ), z=(-1000, 0), topology=(Periodic, Periodic, Bounded), halo=(5, 5, 5))

    model = HydrostaticFreeSurfaceModel(grid; coriolis=FPlane(f=1e-4, scheme=scheme(grid)), timestepper=:SplitRungeKutta3,
                                        momentum_advection=WENOVectorInvariant(order=5),
                                        free_surface=SplitExplicitFreeSurface(grid; substeps=30),
                                        buoyancy=nothing, tracers=nothing, closure=nothing)

    # Nondivergent initial flow from a random streamfunction with wavelengths of 10 to 30 cells
    Random.seed!(seed)
    ψ = zeros(N, N)
    for n in 1:60
        kx, ky, phase = rand(3:10), rand(3:10), 2π * rand()
        ψ .+= [sin(2π * (kx * i + ky * j) / N + phase) for i in 1:N, j in 1:N] ./ (kx^2 + ky^2)
    end
    ψ .*= 0.5 * Δ / maximum(abs, diff(ψ, dims=1))
    u = - (circshift(ψ, (0, -1)) .- ψ) ./ Δ
    v =   (circshift(ψ, (-1, 0)) .- ψ) ./ Δ
    set!(model, u=reshape(u, N, N, 1), v=reshape(v, N, N, 1))

    ζ = Field(KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, model.velocities.u, model.velocities.v))
    energy() = (mean(interior(model.velocities.u) .^ 2) + mean(interior(model.velocities.v) .^ 2)) / 2
    enstrophy() = (compute!(ζ); mean(interior(ζ) .^ 2))

    times, energies, enstrophies = [0.0], [energy()], [enstrophy()]
    vorticities = [Array(interior(ζ, :, :, 1))]
    steps_per_day = round(Int, 1day / Δt)
    for n in 1:round(Int, stop_time / Δt)
        time_step!(model, Δt)
        if n % steps_per_day == 0
            push!(times, model.clock.time)
            push!(energies, energy())
            push!(enstrophies, enstrophy())
            push!(vorticities, Array(interior(ζ, :, :, 1)))
        end
    end

    return (; times, energies, enstrophies, vorticities)
end

effective_viscosity(r, n) = - (r.energies[n+1] - r.energies[n]) / (r.times[n+1] - r.times[n]) / ((r.enstrophies[n+1] + r.enstrophies[n]) / 2)

schemes = ("EnstrophyConserving" => grid -> EnstrophyConserving(),
           "CDScheme τ=∞"        => grid -> CDScheme(grid; relaxation_time=Inf),
           "CDScheme τ=10d"      => grid -> CDScheme(grid; relaxation_time=10days))

results = Dict(name => decaying_turbulence(scheme) for (name, scheme) in schemes)

@info "Kinetic energy E(t) / E(0) and effective viscosity -(dE/dt)/Z [m² s⁻¹] averaged over each window"
for (name, _) in schemes
    r = results[name]
    windows = ((1, 10), (11, 30), (31, 60))
    ν = [mean(effective_viscosity(r, n) for n in first:last) for (first, last) in windows]
    E = [r.energies[n] / r.energies[1] for n in (11, 31, 61)]
    @info @sprintf("%-20s E(10d) %.3f  E(30d) %.3f  E(60d) %.3f   ν days 0-10: %6.1f  10-30: %6.1f  30-60: %6.1f",
                   name, E..., ν...)
end

f = 1e-4
names = first.(schemes)
ζmax = 0.8 * maximum(abs, results[names[1]].vorticities[1]) / f
n = Observable(1)

fig = Figure(size=(1500, 700))
Label(fig[0, 1:3], @lift(@sprintf("Day %d", round(Int, results[names[1]].times[$n] / 1day))); fontsize=22, tellwidth=false)
for (column, name) in enumerate(names)
    ax = Axis(fig[1, column]; title=name * ", ζ / f", aspect=DataAspect())
    hidedecorations!(ax)
    heatmap!(ax, @lift(results[name].vorticities[$n] ./ f); colormap=:balance, colorrange=(-ζmax, ζmax))
end
ax = Axis(fig[2, 1:3]; xlabel="Time [days]", ylabel="E(t) / E(0)", height=180)
for name in names
    r = results[name]
    lines!(ax, r.times ./ 1day, r.energies ./ r.energies[1]; label=name)
end
vlines!(ax, @lift(results[names[1]].times[$n] / 1day); color=:gray)
axislegend(ax; position=:lb)

record(fig, "cd_scheme_decaying_turbulence.mp4", eachindex(results[names[1]].times); framerate=8) do frame
    n[] = frame
end
