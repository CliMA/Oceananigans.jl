using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode
using Oceananigans.TurbulenceClosures: DirectionallyAveragedCoefficient
using NCDatasets
using Printf: @printf
simname = "wall_flow"

const κ = 0.4

function run_wall_flow(closure; arch=CPU(), H=1, L=2π*H, N=32, u★=1, z₀ = 1e-4*H, stop_time=30)
    grid = RectilinearGrid(arch, size=(N, N, N÷2), topology=(Periodic, Periodic, Bounded),
                           x=(0, L), y=(0, L), z=(0, H))
    @show grid

    z₁ = first(znodes(grid, Center()))
    cᴰ = (κ / log(z₁ / z₀))^2

    @inline drag_u(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * u
    @inline drag_v(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + v^2) * v

    u_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(drag_u, field_dependencies = (:u, :v), parameters = (; cᴰ)))
    v_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(drag_v, field_dependencies = (:u, :v), parameters = (; cᴰ)))

    @inline x_pressure_gradient(x, y, z, t, p) = p.u★^2 / p.H
    u_forcing = Forcing(x_pressure_gradient, parameters=(; u★, H))

    model = NonhydrostaticModel(; grid, timestepper = :RungeKutta3,
                                advection = WENO(grid, order=5),
                                boundary_conditions = (; u=u_bcs, v=v_bcs),
                                forcing = (; u = u_forcing),
                                closure = closure)
    @show model

    noise(x, y, z) = u★ * randn() * exp(-(z/0.1)^2)
    u₀(x, y, z) = (u★/κ) * log(z/z₀)
    set!(model, u=u₀, v=noise, w=noise)

    Δt₀ = 1e-4 * (H / u★) / N
    simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

    wizard = TimeStepWizard(max_change=1.1, cfl=0.9)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))

    start_time = time_ns() # so we can print the total elapsed wall time
    progress_message(sim) = @printf("Iteration: %04d,  time: %s,  Δt: %s,  max|u|: %.2e m/s,  wall time: %s\n",
                                    iteration(sim), prettytime(time(sim)), prettytime(sim.Δt), maximum(abs, sim.model.velocities.u),
                                    prettytime((time_ns() - start_time) * 1e-9))
    add_callback!(simulation, Callback(progress_message, IterationInterval(100)))

    closure_name = string(nameof(typeof(closure.coefficient)))

    u, v, w = model.velocities

    U = Average(u, dims=(1,2))
    z = @show KernelFunctionOperation{Nothing, Nothing, Face}(znode, model.grid, Center(), Center(), Face())
    ϕ = @show @at (Nothing, Nothing, Center) κ * z / u★ * ∂z(Field(U))

    if closure.coefficient isa DirectionallyAveragedCoefficient
        cₛ² = model.diffusivity_fields.LM_avg / model.diffusivity_fields.MM_avg
    else
        cₛ² = Field{Nothing, Nothing, Center}(grid)
        cₛ² .= model.closure.coefficient^2
    end
    outputs = (; u, w, U, ϕ, cₛ²)

    simulation.output_writers[:fields] = NetCDFWriter(model, outputs;
                                                      filename = joinpath(@__DIR__, simname *"_"* closure_name *".nc"),
                                                      schedule = TimeInterval(1),
                                                      global_attributes = (; u★, z₀, H, L),
                                                      overwrite_existing = true)
    run!(simulation)

end

closures = [SmagorinskyLilly(coefficient=0.16), SmagorinskyLilly(coefficient=DirectionallyAveragedCoefficient(1,2))]
for closure in closures
    @info "Running" closure
    run_wall_flow(closure, N=64, stop_time=60, arch=GPU())
end



@info "Start plotting"
using CairoMakie
using NCDatasets
using Statistics: mean
set_theme!(Theme(fontsize = 18))
fig = Figure(size = (800, 500))
ax1 = Axis(fig[2, 1]; xlabel = "cₛ", ylabel = "z", limits = ((0, 0.3), (0, 0.8)))
ax2 = Axis(fig[2, 2]; xlabel = "z", ylabel = "U", limits = ((1e-3, 1), (10, 30)), xscale = log10)
ax3 = Axis(fig[2:3, 3]; xlabel = "ϕ = κ z ∂z(U) / u★", ylabel = "z", limits = ((0.0, 2.0), (0, 0.8)))
ax4 = Axis(fig[3, 1]; xlabel = "x (SmagLilly)", ylabel = "y")
ax5 = Axis(fig[3, 2]; xlabel = "x (ScaleInv)", ylabel = "y")
n = Observable(1)

colors = [:red, :blue]
Δt = 10
for (i, closure) in enumerate(closures)
    closure_name = string(nameof(typeof(closure.coefficient)))
    local filename = simname * "_" * closure_name
    @info "Plotting from " * filename
    ds = NCDataset(filename * ".nc", "r")

    xc, yc, zc = ds["xC"], ds["yC"], ds["zC"]

    #cₛ² = @lift sqrt.(max.(ds["cₛ²"], 0))[:, $n]
    cₛ² = mean(sqrt.(max.(ds["cₛ²"], 0))[:, end-Δt:end], dims=2)[:, 1]
    scatterlines!(ax1, cₛ², zc, color=colors[i], markercolor=colors[i], label=closure_name)

    #U = @lift ds["U"][:, $n]
    U = mean(ds["U"][:, end-Δt:end], dims=2)[:, 1]
    scatterlines!(ax2, zc, U, color=colors[i], markercolor=colors[i])

    #ϕ = @lift ds["ϕ"][:, $n]
    ϕ = mean(ds["ϕ"][:, end-Δt:end], dims=2)[:, 1]
    scatterlines!(ax3, ϕ, zc, color=colors[i], markercolor=colors[i])

    if i == 1
        û = (ds.attrib["u★"] / κ) * log.(zc / ds.attrib["z₀"])
        lines!(ax2, zc, û, color=:black)
        lines!(ax3, ones(length(zc)), zc, color=:black, linestyle=:dash)
    end

    u = @lift ds["u"][:, :, 4, $n]
    heatmap!(fig[3, i], xc, yc, u,)

    global times = ds["time"]
end

axislegend(ax1, labelsize=10)
title = @lift "t = " * string(round(times[$n], digits=2))
Label(fig[1, 1:2], title, fontsize=24, tellwidth=false)
frames = 1:length(times)
record(fig, simname * ".mp4", frames, framerate=8) do i
    n[] = i
end
