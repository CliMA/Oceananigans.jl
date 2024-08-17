using Oceananigans
using Oceananigans.Units
using Printf
simname = "wall_flow"

const κ = 0.4
H = 1
L = 2π*H
z₀ = 1e-4*H
N = 32
u★ = 1

grid = RectilinearGrid(CPU(), size=(N, N, N÷2), topology=(Periodic, Periodic, Bounded),
                       x=(0, L), y=(0, L), z=(0, H))
@show grid

function run_wall_flow(closure; grid=grid, H=1, L=2π*H, N=32, u★=1)
    z₁ = first(znodes(grid, Center()))
    cᴰ = (κ / log(z₁ / z₀))^2

    @inline drag_u(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * u
    @inline drag_v(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * v

    u_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(drag_u, field_dependencies = (:u, :v), parameters = (; cᴰ)))
    v_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(drag_v, field_dependencies = (:u, :v), parameters = (; cᴰ)))

    @inline x_pressure_gradient(x, y, z, t, p) = p.u★^2 / p.H
    u_forcing = Forcing(x_pressure_gradient, parameters=(; u★, H))

    model = NonhydrostaticModel(; grid, timestepper = :RungeKutta3,
                                advection = CenteredFourthOrder(),
                                boundary_conditions = (; u=u_bcs, v=v_bcs),
                                forcing = (; u = u_forcing),
                                closure = closure)
    @show model

    noise(x, y, z) = 1e0 * u★ * randn()
    u₀(x, y, z) = (u★/κ) * log(z/z₀)
    set!(model, u=u₀, v=noise, w=noise)

    Δt₀ = 1e-4 * (H / u★) / N
    simulation = Simulation(model, Δt = Δt₀, stop_time = 20)

    wizard = TimeStepWizard(max_change=1.1, cfl=0.9)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))

    start_time = time_ns() # so we can print the total elapsed wall time
    progress_message(sim) = @printf("Iteration: %04d,  time: %s,  Δt: %s,  max|u|: %.1e m/s,  wall time: %s\n",
                                    iteration(sim), prettytime(time(sim)), prettytime(sim.Δt), maximum(abs, sim.model.velocities.u), prettytime((time_ns() - start_time) * 1e-9))
    add_callback!(simulation, Callback(progress_message, IterationInterval(100)))

    closure_name = string(nameof(typeof(closure)))

    u, v, w = model.velocities

    U = Average(u, dims=(1,2))
    V = Average(v, dims=(1,2))
    w² = w*w
    uw = u*w
    vw = v*w
    τ = √(uw^2 + vw^2)

    outputs = (; u, v, w, U, V, w², τ)

    simulation.output_writers[:fields] = NetCDFOutputWriter(model, outputs;
                                                            filename = joinpath(@__DIR__, simname *"_"* closure_name *".nc"),
                                                            schedule = TimeInterval(1),
                                                            overwrite_existing = true)
    run!(simulation)

end




closures = [SmagorinskyLilly(),]
for closure in closures
    @info "Running" closure
    run_free_convection(closure)
end



@info "Start plotting"
using CairoMakie
set_theme!(Theme(fontsize = 18))
fig = Figure(size = (800, 800))
axis_kwargs = (xlabel = "x", ylabel = "y", limits = ((0, 2π), (0, 2π)), aspect = AxisAspect(1))
n = Observable(1)

for (i, closure) in enumerate(closures)
    closure_name = string(nameof(typeof(closure)))
    local filename = simname * "_" * closure_name
    @info "Plotting from " * filename
    local w_timeseries = FieldTimeSeries(filename * ".jld2", "w")
    local w = @lift interior(w_timeseries[$n], :, :, 1)

    local ax = Axis(fig[2, i]; title = "ΣᵢⱼΣᵢⱼ; $closure_name", axis_kwargs...)
    local xc, yc, zc = nodes(w_timeseries)
    heatmap!(ax, xc, yc, w, colormap = :speed, colorrange = (0, 2))

    global times = w_timeseries.times
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
record(fig, simname * ".mp4", frames, framerate=24) do i
    n[] = i
end
