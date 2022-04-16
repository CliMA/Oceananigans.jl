using Printf
using Statistics
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, mask_immersed_field!

@inline bottom_drag_func(x, y, t, u, Cᴰ) = - Cᴰ * u^2

function hilly_simulation(;
                          Nx = 64,
                          Nz = Nx,
                          ϵ = 0.1,
                          Re = 1e4,
                          N² = 1e-2,
                          boundary_condition = :no_slip,
                          stop_time = 1,
                          save_interval = 0.1,
                          architecture = CPU(),
                          filename = "flow_over_hills")

    underlying_grid = RectilinearGrid(architecture,
                                      size = (Nx, Nz),
                                      halo = (3, 3),
                                      x = (0, 2π),
                                      z = (0, 2π),
                                      topology = (Periodic, Flat, Bounded))

    if ϵ > 0
        x, y, z = nodes((Center, Center, Center), underlying_grid, reshape=true)
        hills = @. ϵ * (1 + sin(x)) / 2
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(hills))
    else # no hills
        grid = underlying_grid
    end

    closure = isfinite(Re) ? ScalarDiffusivity(ν=1/Re, κ=1/Re) : nothing

    if boundary_condition == :no_slip
        no_slip = ValueBoundaryCondition(0)
        u_bcs = FieldBoundaryConditions(top=no_slip, bottom=no_slip, immersed=no_slip)
        boundary_conditions = (; u = u_bcs)
    elseif boundary_condition == :bottom_drag
        Δz = 2π / Nz
        z₀ = 1e-4
        κ = 0.4
        Cᴰ = (κ / log(Δz / 2z₀))^2
        u_bottom_bc = FluxBoundaryCondition(bottom_drag_func, field_dependencies=:u, parameters=Cᴰ)
        u_bcs = FieldBoundaryConditions(bottom=u_bottom_bc)
        boundary_conditions = (; u = u_bcs)
        @info string("Using a bottom drag with coefficient ", Cᴰ)
    else
        boundary_conditions = NamedTuple()
    end

    model = NonhydrostaticModel(; grid, closure, boundary_conditions,
                                advection = WENO5(),
                                timestepper = :RungeKutta3,
                                tracers = :b,
                                buoyancy = BuoyancyTracer())

    bᵢ(x, y, z) = N² * z + 1e-9 * rand()
    set!(model, b=bᵢ, u=1)

    Δx = 2π / Nx
    Δt = 0.1 * Δx
    simulation = Simulation(model; Δt, stop_time)

    u, v, w = model.velocities
    Uᵢ = mean(u)

    wall_clock = Ref(time_ns())

    function progress(sim)
        δU = mean(u) / Uᵢ
        elapsed = 1e-9 * (time_ns() - wall_clock[])
        @info @sprintf("Iter: %d, time: %.2e, δU: %.2e, wall time: %s",
                       iteration(sim), time(sim), δU, prettytime(elapsed))
        wall_clock[] = time_ns()
        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    U = Average(u, dims=(1, 2, 3))
    ξ = ∂z(u) - ∂x(w)

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, merge(model.velocities, model.tracers, (; ξ, U));
                         schedule = TimeInterval(save_interval),
                         filename,
                         overwrite_existing = true)

    @info "Made a simulation of"
    @show model

    @info "The x-velocity is"
    @show model.velocities.u

    return simulation
end

function momentum_time_series(filename)
    U = FieldTimeSeries(filename * ".jld2", "U")
    t = U.times
    δU = [U[1, 1, 1, n] / U[1, 1, 1, 1] for n=1:length(t)]
    return δU, t
end

Nx = 64
stop_time = 10.0

reference_name = "hills_reference"
reference_sim = hilly_simulation(; stop_time, Nx, ϵ=0.0, filename=reference_name, boundary_condition=:no_slip)
run!(reference_sim)
δU_reference, t_reference = momentum_time_series(reference_name)

no_slip_name = "hills_no_slip"
no_slip_sim = hilly_simulation(; stop_time, Nx, ϵ=0.1, filename=no_slip_name, boundary_condition=:no_slip)
run!(no_slip_sim)
δU_no_slip, t_no_slip = momentum_time_series(no_slip_name)

free_slip_name = "hills_free_slip"
free_slip_sim = hilly_simulation(; stop_time, Nx, ϵ=0.1, filename=free_slip_name, boundary_condition=:free_slip)
run!(free_slip_sim)
δU_free_slip, t_free_slip = momentum_time_series(free_slip_name)

#=
experiments = []
for ϵ = [0.02, 0.05, 0.1]
    name = string("flow_over_hills_height_", ϵ)
    push!(experiments, (; ϵ, name))
    simulation = hilly_simulation(; stop_time, Nx, ϵ, name)
    run!(simulation)
end

using GLMakie

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, t_ref, δU_ref, label="Reference with bottom drag")

for experiment in experiments
    name = experiment.name
    ϵ = experiment.ϵ
    δU, t = momentum_time_series(name * ".jld2")
    lines!(ax, t, δU, label=string("ϵ = ", ϵ))
end

axislegend(ax)

display(fig)

# Animate vorticity if you like
filepath = experiments[end].name * ".jld2"
ξ = FieldTimeSeries(filepath, "ξ")
Nt = length(ξ.times)

fig = Figure()
slider = Slider(fig[2, 1], range=1:Nt, startvalue=1)
n = slider.value

title = @lift @sprintf("Vorticity in flow over hills at t = %.2e", ξ.times[$n])
ax = Axis(fig[1, 1]; title)
ξn = @lift interior(ξ[$n], :, 1, :)

masked_ξn = @lift begin
    ξn = ξ[$n]
    mask_immersed_field!(ξn, NaN)
    interior(ξn, :, 1, :)
end

heatmap!(ax, ξn)

display(fig)
=#

