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
        boundary_conditions = (; u = u_bcs)
    elseif boundary_condition == :bottom_drag
        Δz = 2π / Nz
        u_drag_bc = FluxBoundaryCondition(bottom_drag_u, field_dependencies=(:u, :w), parameters=Cᴰ)
        w_drag_bc = FluxBoundaryCondition(bottom_drag_w, field_dependencies=(:u, :w), parameters=Cᴰ)
        u_bcs = FieldBoundaryConditions(bottom=u_drag_bc, immersed=u_drag_bc)
        w_bcs = FieldBoundaryConditions(immersed=w_drag_bc)
        boundary_conditions = (; u = u_bcs, w = w_bcs)
        @info string("Using a bottom drag with coefficient ", Cᴰ)
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

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, merge(model.velocities, model.tracers, (; ξ, U));
                         schedule = TimeInterval(save_interval),
                         with_halos = true,
                         filename,
                         overwrite_existing = true)

    @info "Made a simulation of"
    @show model

    @info "The x-velocity is"
    @show model.velocities.u

    @info "The grid is"
    @show model.grid

    return simulation
end

function momentum_time_series(filename)
    U = FieldTimeSeries(filename * ".jld2", "U")
    t = U.times
    δU = [U[1, 1, 1, n] / U[1, 1, 1, 1] for n=1:length(t)]
    return δU, t
end

Nx = 64
stop_time = 100.0

#=
reference_name = "hills_reference_$Nx"
reference_sim = hilly_simulation(; stop_time, Nx, h=0.0, filename=reference_name, boundary_condition=:no_slip)
run!(reference_sim)
δU_reference, t_reference = momentum_time_series(reference_name)

free_slip_name = "hills_free_slip_$Nx"
free_slip_sim = hilly_simulation(; stop_time, Nx, h=0.2, filename=free_slip_name, boundary_condition=:free_slip)
run!(free_slip_sim)
δU_free_slip, t_free_slip = momentum_time_series(free_slip_name)

no_slip_name = "hills_no_slip_$Nx"
no_slip_sim = hilly_simulation(; stop_time, Nx, h=0.2, filename=no_slip_name, boundary_condition=:no_slip)
run!(no_slip_sim)
δU_no_slip, t_no_slip = momentum_time_series(no_slip_name)
=#

bottom_drag_name = "hills_bottom_drag_$Nx"
bottom_drag_sim = hilly_simulation(; stop_time, Nx, h=0.2, filename=bottom_drag_name, boundary_condition=:bottom_drag)
run!(bottom_drag_sim)
δU_bottom_drag, t_bottom_drag = momentum_time_series(bottom_drag_name)

ξr = FieldTimeSeries("hills_reference_$Nx.jld2", "ξ")
ξn = FieldTimeSeries("hills_no_slip_$Nx.jld2", "ξ")
ξf = FieldTimeSeries("hills_free_slip_$Nx.jld2", "ξ")
t = ξf.times
Nt = length(t)

Ur = FieldTimeSeries("hills_reference_$Nx.jld2", "U")
Un = FieldTimeSeries("hills_no_slip_$Nx.jld2", "U")
Uf = FieldTimeSeries("hills_free_slip_$Nx.jld2", "U")

U₀ = Ur[1, 1, 1, 1]
δUr = [(Ur[1, 1, 1, n] - U₀) / U₀ for n = 1:Nt]
δUn = [(Un[1, 1, 1, n] - U₀) / U₀ for n = 1:Nt]
δUf = [(Uf[1, 1, 1, n] - U₀) / U₀ for n = 1:Nt]

x, y, z = nodes(ξr)

ξmax = maximum(abs, ξf)
ξlim = ξmax / 50

fig = Figure(resolution=(2400, 1200))
ax1 = Axis(fig[2, 2:4], aspect=2π, xlabel="x", ylabel="z", title="Reference")
ax2 = Axis(fig[3, 2:4], aspect=2π, xlabel="x", ylabel="z", title="No-slip")
ax3 = Axis(fig[4, 2:4], aspect=2π, xlabel="x", ylabel="z", title="Free-slip")
slider = Slider(fig[5, 2:4], range=1:Nt, startvalue=1)
n = slider.value

axu = Axis(fig[2:4, 5], xlabel="t", ylabel="Total momentum")
lines!(axu, t, δUr, label="Reference")
lines!(axu, t, δUn, label="No-slip")
lines!(axu, t, δUf, label="Free-slip")
axislegend(axu, position=:lb)

tn = @lift t[$n]
min_δU = min(minimum(δUr), minimum(δUn), minimum(δUf))
vlines!(axu, tn, ymin=min_δU, ymax=1.0)

title = @lift string("Flow over hills at t = ", prettysummary(t[$n]))
Label(fig[1, 1:4], title)

ξrⁿ = @lift interior(ξr[$n], :, 1, :)

ξnⁿ = @lift begin
    ξnn = ξn[$n]
    mask_immersed_field!(ξnn, NaN)
    interior(ξnn, :, 1, :)
end

ξfⁿ = @lift begin
    ξfn = ξf[$n]
    mask_immersed_field!(ξfn, NaN)
    interior(ξfn, :, 1, :)
end

heatmap!(ax1, x, z, ξrⁿ, colorrange=(-ξlim, ξlim), colormap=:redblue) 
heatmap!(ax2, x, z, ξnⁿ, colorrange=(-ξlim, ξlim), colormap=:redblue) 
hmξ = heatmap!(ax3, x, z, ξfⁿ, colorrange=(-ξlim, ξlim), colormap=:redblue) 

cb = Colorbar(fig[2:4, 1], vertical=true, flipaxis=true, label="Vorticity, ∂z(u) - ∂x(w)")

display(fig)

record(fig, "flow_over_hills.mp4", 1:Nt, framerate=24) do nn
    n[] = nn
end

