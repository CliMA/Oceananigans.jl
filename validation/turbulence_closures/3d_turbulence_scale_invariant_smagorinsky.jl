using Oceananigans
using Oceananigans.Fields: interpolate!
using Statistics

N = 32
grid = RectilinearGrid(size=(N, N, N), extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))
coarse_grid = RectilinearGrid(size=(N÷4, N÷4, N÷4), extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))


function run_3d_turbulence(closure; grid = grid, coarse_grid = coarse_grid)
    model = NonhydrostaticModel(; grid, timestepper = :RungeKutta3, advection = UpwindBiasedFifthOrder(),
                                closure = closure)

    random_c = CenterField(coarse_grid) # Technically this shouldn't be a CenterField, but oh well
    random_c .= rand(size(random_c)...)

    u, v, w = model.velocities
    interpolate!(u, random_c)
    interpolate!(v, random_c)
    interpolate!(w, random_c)
    u .-= mean(u)
    v .-= mean(v)
    w .-= mean(w)
    @show u

    simulation = Simulation(model, Δt=0.2, stop_time=80)

    wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=0.5)
    add_callback!(simulation, wizard, IterationInterval(10))

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


@info "Start plotting"
using CairoMakie
set_theme!(Theme(fontsize = 18))
fig = Figure(size = (800, 800))
axis_kwargs = (xlabel = "x", ylabel = "y", limits = ((0, 2π), (0, 2π)), aspect = AxisAspect(1))
n = Observable(1)

for (i, closure) in enumerate(closures)
    closure_name = string(nameof(typeof(closure)))
    local filename = "3d_turbulence_" * closure_name
    @info "Plotting from " * filename
    local S_timeseries = FieldTimeSeries(filename * ".jld2", "S²")
    local S² = @lift interior(S_timeseries[$n], :, :, 1)

    local ax = Axis(fig[2, i]; title = "ΣᵢⱼΣᵢⱼ; $closure_name", axis_kwargs...)
    local xc, yc, zc = nodes(S_timeseries)
    heatmap!(ax, xc, yc, S²; colormap = :speed, colorrange = (0, 2))

    times = ω_timeseries.times
    if closure isa ScaleInvariantSmagorinsky
        c²ₛ_timeseries = FieldTimeSeries(filename * ".jld2", "c²ₛ")
        c²ₛ = interior(c²ₛ_timeseries, 1, 1, 1, :)
        global cₛ = sqrt.(max.(c²ₛ, 0))
        local ax_cₛ = Axis(fig[3, 1:length(closures)]; title = "Smagorinsky coefficient", xlabel = "Time", limits = ((0, nothing), (0, 0.2)))
        lines!(ax_cₛ, times, cₛ, color=:black, label="Scale Invariant Smagorinsky")
        hlines!(ax_cₛ, [0.16], linestyle=:dash, color=:blue)
    end
end

title = @lift "t = " * string(round(times[$n], digits=2)) * ", cₛ = " * string(round(cₛ[$n], digits=4)) 
Label(fig[1, 1:2], title, fontsize=24, tellwidth=false)
frames = 1:length(times)
@info "Making a neat animation of vorticity and speed..."
record(fig, filename * ".mp4", frames, framerate=24) do i
    n[] = i
end
