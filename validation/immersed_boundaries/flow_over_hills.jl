using Printf
using Statistics
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, mask_immersed_field!

"""
    Hills(ϵ)

Return a representation of sinusoidal hills with period 2π
and amplitude ϵ, such that

```math
h(x) = ϵ * (1 + sin(x) / 2)
```

where ``h`` is the elevation of the bottom.
"""
struct Hills{T}
    ϵ :: T
end

@inline (h::Hills)(x, y) = h.ϵ * (1 + sin(x)) / 2

function hilly_simulation(;
                          Nx = 64,
                          Nz = Nx,
                          ϵ = 0.1,
                          Re = 1e4,
                          N² = 1e-2,
                          bottom_drag = false,
                          stop_time = 1,
                          save_interval = 0.1,
                          architecture = CPU(),
                          name = "flow_over_hills.jld2")

    underlying_grid = RectilinearGrid(architecture,
                                      size = (Nx, Nz),
                                      halo = (3, 3),
                                      x = (0, 2π),
                                      z = (0, 2π),
                                      topology = (Periodic, Flat, Bounded))

    if ϵ > 0
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(Hills(ϵ)))
    else # no hills
        grid = underlying_grid
    end

    closure = isfinite(Re) ? ScalarDiffusivity(ν=1/Re, κ=1/Re) : nothing

    if bottom_drag
        Δz = 2π / Nz
        z₀ = 1e-4
        κ = 0.4
        Cᴰ = (κ / log(Δz / 2z₀))^2
        @inline bottom_drag_func(x, y, t, u, Cᴰ) = - Cᴰ * u^2
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

    function progress(sim)
        δU = mean(u) / Uᵢ
        @info @sprintf("Iter: %d, time: %.2e, δU: %.2e", iteration(sim), time(sim), δU)
        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    U = Average(u, dims=(1, 2, 3))
    ξ = ∂z(u) - ∂x(w)

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, merge(model.velocities, model.tracers, (; ξ, U)),
                         schedule = TimeInterval(save_interval),
                         prefix = name,
                         force = true)

    @info "Make a simulation:"
    @show simulation

    return simulation
end

function momentum_time_series(filepath)
    U = FieldTimeSeries(filepath, "U")
    t = U.times
    δU = [U[1, 1, 1, n] / U[1, 1, 1, 1] for n=1:length(t)]
    return δU, t
end

name = "bottom_drag_reference"
reference_sim = hilly_simulation(; Nx=64, Nz=64, ϵ=0, name, bottom_drag=true)
run!(reference_sim)
δU_ref, t_ref = momentum_time_series(name * ".jld2")

experiments = []
for ϵ = [0.02, 0.05, 0.1]
    name = string("flow_over_hills_height_", ϵ)
    push!(experiments, (; ϵ, name))
    simulation = hilly_simulation(; ϵ, name)
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

#=
# Animate vorticity if you like
ξ = FieldTimeSeries(filepath, "ξ")
Nt = length(ξ.times)


fig = Figure()
slider = Slider(fig[2, 1], range=1:Nt, startvalue=1)
n = slider.value

title = @lift @sprintf("Vorticity in flow over hills at t = %.2e", ξ.times[$n])
ax = Axis(fig[1, 1]; title)
ξn = @lift interior(ξ[$n], :, 1, :)

#=
masked_ξn = @lift begin
    ξn = ξ[$n]
    mask_immersed_field!(ξn, NaN)
    interior(ξn, :, 1, :)
end
=#

heatmap!(ax, ξn)

display(fig)
=#
