using Oceananigans
using Printf
using Statistics
using GLMakie

using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Utils: prettysummary

# Monin-Obukhov drag coefficient
z₀ = 1e-4 # Charnock roughness
κ = 0.4 # Von Karman constant
Cᴰ(Δz) = (κ / log(Δz / 2z₀))^2

@inline bottom_drag_u(x, y, t, u, w, Cᴰ) = - Cᴰ * u * sqrt(u^2 + w^2)
@inline bottom_drag_w(x, y, t, u, w, Cᴰ) = - Cᴰ * w * sqrt(u^2 + w^2)
@inline bottom_drag_u(x, y, z, t, u, w, Cᴰ) = - Cᴰ * u * sqrt(u^2 + w^2)
@inline bottom_drag_w(x, y, z, t, u, w, Cᴰ) = - Cᴰ * w * sqrt(u^2 + w^2)

function hilly_simulation(; Nx = 64,
                            Nz = Nx,
                            h = 0.1,
                            Re = 1e4,
                            N² = 1e-2,
                            boundary_condition = :no_slip,
                            stop_time = 1,
                            save_interval = 0.1,
                            architecture = CPU(),
                            filename = "flow_over_hills")

    underlying_grid = RectilinearGrid(architecture, size = (Nx, Nz), halo = (3, 3),
                                      x = (0, 2π), z = (0, 1),
                                      topology = (Periodic, Flat, Bounded))

    if h > 0
        hills(x, y) = h * (1 + sin(x)) / 2
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(hills))
    else # no hills
        grid = underlying_grid
    end

    closure = isfinite(Re) ? ScalarDiffusivity(ν=1/Re, κ=1/Re) : nothing

    if boundary_condition == :no_slip
        no_slip = ValueBoundaryCondition(0)
        u_bcs = FieldBoundaryConditions(bottom=no_slip, immersed=no_slip)
        w_bcs = FieldBoundaryConditions(immersed=no_slip)
        boundary_conditions = (; u = u_bcs, w = w_bcs)
    elseif boundary_condition == :bottom_drag
        Δz = 1 / Nz
        Δx = 2π / Nz
        u_drag_bc = FluxBoundaryCondition(bottom_drag_u, field_dependencies=(:u, :w), parameters=Cᴰ(Δz))
        w_drag_bc = FluxBoundaryCondition(bottom_drag_w, field_dependencies=(:u, :w), parameters=Cᴰ(Δx))
        u_bcs = FieldBoundaryConditions(bottom=u_drag_bc, immersed=u_drag_bc)
        w_bcs = FieldBoundaryConditions(immersed=w_drag_bc)
        boundary_conditions = (; u = u_bcs, w = w_bcs)
        @info string("Using a bottom drag with coefficient ", Cᴰ(Δz))
    else
        boundary_conditions = NamedTuple()
    end

    model = NonhydrostaticModel(; grid, closure, boundary_conditions,
                                advection = WENO5(),
                                timestepper = :RungeKutta3,
                                tracers = :b,
                                buoyancy = BuoyancyTracer())

    # Steady flow + perturbations
    δh = 0.1
    ∂z_ψᵋ(x, z) = 4π * sin(4x) * cos(4π * z) * exp(-(z - h)^2 / 2δh^2)
    ∂x_ψᵋ(x, z) = 4  * cos(4x) * sin(4π * z) * exp(-(z - h)^2 / 2δh^2)
    bᵢ(x, y, z) = N² * z + 1e-9 * rand()
    uᵢ(x, y, z) = 1.0 + ∂z_ψᵋ(x, z)
    wᵢ(x, y, z) = - ∂x_ψᵋ(x, z)
    set!(model, b=bᵢ, u=uᵢ, w=wᵢ)

    Δx = 2π / Nx
    Δt = 0.1 * Δx
    simulation = Simulation(model; Δt, stop_time)

    u, v, w = model.velocities
    Uᵢ = mean(u)

    wall_clock = Ref(time_ns())

    function progress(sim)
        δU = mean(u) / Uᵢ
        elapsed = 1e-9 * (time_ns() - wall_clock[])
        @info @sprintf("Iter: %d, time: %.2e, δU: %.2e, max|w|: %.2e, wall time: %s",
                       iteration(sim), time(sim), δU, maximum(abs, w), prettytime(elapsed))
        wall_clock[] = time_ns()
        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    wizard = TimeStepWizard(cfl=0.5)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    U = Average(u, dims=(1, 2, 3))
    ξ = ∂z(u) - ∂x(w)

    ke = @at (Center, Center, Center) (u^2 + v^2 + w^2) / 2
    KE = Average(ke, dims=(1, 2, 3))

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, merge(model.velocities, model.tracers, (; ξ, U, KE));
                         schedule = TimeInterval(save_interval),
                         with_halos = true,
                         filename,
                         overwrite_existing = true)

    @info "Made a simulation of"
    @show model

    @info "The grid is"
    @show model.grid

    @info "The x-velocity immersed boundary condition is"
    @show model.velocities.u.boundary_conditions.immersed

    return simulation
end

#####
##### Run them!
#####

Nx = 32
stop_time = 20.0

experiments = ["reference", "no_slip", "free_slip", "bottom_drag"]
Nexp = length(experiments)

for exp in experiments
    filename = "hills_$(exp)_$Nx"
    h = exp == "reference" ? 0.0 : 0.2
    reference_sim = hilly_simulation(; stop_time, Nx, filename, h, boundary_condition=Symbol(exp))
    run!(reference_sim)
end

#####
##### Plot results
#####

ξ  = Dict(exp => FieldTimeSeries("hills_$(exp)_$Nx.jld2", "ξ")  for exp in experiments)
U  = Dict(exp => FieldTimeSeries("hills_$(exp)_$Nx.jld2", "U")  for exp in experiments)
KE = Dict(exp => FieldTimeSeries("hills_$(exp)_$Nx.jld2", "KE") for exp in experiments)

t = ξ["reference"].times
Nt = length(t)
t = t[1:Nt]
δU_series(U) = [(U[1, 1, 1, n] - U[1, 1, 1, 1]) / U[1, 1, 1, 1] for n = 1:Nt]
δK_series(K) = [(K[1, 1, 1, n] - K[1, 1, 1, 1]) / K[1, 1, 1, 1] for n = 1:Nt]
δU = Dict(exp => δU_series(u) for (exp, u) in U)
δKE = Dict(exp => δK_series(k) for (exp, k) in KE)

fig = Figure(resolution=(1800, 1200))
ax = Dict(exp => Axis(fig[i+1, 2:4], aspect=2π, xlabel="x", ylabel="z", title=exp)
          for (i, exp) in enumerate(experiments))

axu = Axis(fig[2:3, 5], xlabel="t", ylabel="Total momentum")
axe = Axis(fig[4:5, 5], xlabel="t", ylabel="Total kinetic energy")

slider = Slider(fig[Nexp+2, 2:4], range=1:Nt, startvalue=1)
n = slider.value

# Title
title = @lift string("Flow over hills at t = ", prettysummary(t[$n]))
Label(fig[1, 1:5], title)

# Vorticity heatmaps
ξi(ξ) = @lift begin
    ξn = ξ[$n]
    mask_immersed_field!(ξn, NaN)
    interior(ξn, :, 1, :)
end

ξⁱ = Dict(exp => ξi(ξ[exp]) for exp in experiments)
x, y, z = nodes(ξ["reference"])
ξmax = maximum(abs, ξ["reference"])
ξlim = ξmax / 50

hm = Dict(exp => heatmap!(ax[exp], x, z, ξⁱ[exp], colorrange=(-ξlim, ξlim), colormap=:redblue)
          for exp in experiments)

cb = Colorbar(fig[2:5, 1], hm["reference"], vertical=true, flipaxis=true, label="Vorticity, ∂z(u) - ∂x(w)")

# Momentum and energy plots
for exp in experiments
    lines!(axu, t, δU[exp], label=exp)
    lines!(axe, t, δKE[exp], label=exp)
end

axislegend(axu, position=:lb)
axislegend(axe, position=:rt)

tn = @lift t[$n]
min_δU = minimum(minimum(δ) for δ in values(δU))
min_δK = minimum(minimum(δ) for δ in values(δKE))
vlines!(axu, tn, ymin=min_δU, ymax=1.0)
vlines!(axe, tn, ymin=min_δK, ymax=1.0)

display(fig)

record(fig, "flow_over_hills.mp4", 1:Nt, framerate=24) do nn
    n[] = nn
end

