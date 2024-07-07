using Oceananigans
using Statistics

N = 32
grid = RectilinearGrid(size=(N, N, N), extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))


function run_3d_turbulence(closure; grid = grid)
    model = NonhydrostaticModel(; grid, timestepper = :RungeKutta3, advection = UpwindBiasedFifthOrder(),
                                closure = closure)

    u, v, w = model.velocities
    uᵢ = rand(size(u)...); vᵢ = rand(size(v)...)
    uᵢ .-= mean(uᵢ); vᵢ .-= mean(vᵢ)
    set!(model, u=uᵢ, v=vᵢ)

    simulation = Simulation(model, Δt=0.2, stop_time=80)

    wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=0.5)
    add_callback!(simulation, wizard, IterationInterval(10))

    ω = ∂x(v) - ∂y(u)
    S² = KernelFunctionOperation{Center, Center, Center}(Oceananigans.TurbulenceClosures.ΣᵢⱼΣᵢⱼᶜᶜᶜ, model.grid, u, v, w)

    if closure isa ScaleInvariantSmagorinsky
        c²ₛ = model.diffusivity_fields.LM_avg / model.diffusivity_fields.MM_avg
        outputs = (; ω, S², c²ₛ)
    else
        outputs = (; ω, S²)
    end

    filename = "3d_turbulence_" * string(nameof(typeof(closure)))
    simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                          schedule = TimeInterval(0.6),
                                                          filename = filename * ".jld2",
                                                          overwrite_existing = true)
    run!(simulation)
end

closures = [SmagorinskyLilly(), ScaleInvariantSmagorinsky()]
for closure in closures
    @info "Running" closure
    run_3d_turbulence(closure)
end


#+++ Plotting
c²ₛ_timeseries = FieldTimeSeries(filename * ".jld2", "c²ₛ")
ω_timeseries = FieldTimeSeries(filename * ".jld2", "ω")
S_timeseries = FieldTimeSeries(filename * ".jld2", "S²")

times = ω_timeseries.times

xω, yω, zω = nodes(ω_timeseries)
xc, yc, zc = nodes(S_timeseries)

using CairoMakie
set_theme!(Theme(fontsize = 18))

fig = Figure(size = (800, 800))

axis_kwargs = (xlabel = "x", ylabel = "y", limits = ((0, 2π), (0, 2π)), aspect = AxisAspect(1))

ax_1 = Axis(fig[2, 1]; title = "Vorticity", axis_kwargs...)
ax_2 = Axis(fig[2, 2]; title = "Strain rate squared", axis_kwargs...)


n = Observable(1)

ω = @lift interior(ω_timeseries[$n], :, :, 1)
S² = @lift interior(S_timeseries[$n], :, :, 1)

heatmap!(ax_1, xω, yω, ω; colormap = :balance, colorrange = (-2, 2))
heatmap!(ax_2, xc, yc, S²; colormap = :speed, colorrange = (0, 3))

c²ₛ = interior(c²ₛ_timeseries, 1, 1, 1, :)
cₛ = sqrt.(max.(c²ₛ, 0))
title = @lift "t = " * string(round(times[$n], digits=2)) * ", cₛ = " * string(round(cₛ[$n], digits=4)) 
Label(fig[1, 1:2], title, fontsize=24, tellwidth=false)

frames = 1:length(times)
@info "Making a neat animation of vorticity and speed..."
record(fig, filename * ".mp4", frames, framerate=24) do i
    n[] = i
end
