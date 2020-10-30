# # Barotropic instability in a channel on a β-plane

using Printf
using Plots
using Statistics

using Oceananigans
using Oceananigans.Advection
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.AbstractOperations
using Oceananigans.Utils

# ## Model setup

# We instantiate the model with a hyperdiffusivity. We use a grid with 128² points,
# a fifth-order advection scheme, third-order Runge-Kutta time-stepping,
# and a small isotropic viscosity.

bickley_jet(x, y, z, t, p) = p.U * sech(y / p.L)^2

U = BackgroundField(bickley_jet, parameters=(U=2.0, L=1e4))

model = IncompressibleModel(timestepper = :RungeKutta3, 
                              advection = UpwindBiasedFifthOrder(),

                                   grid = RegularCartesianGrid(size=(64, 64, 1),
                                                               x=(-1e5, 1e5), y=(-1e5, 1e5), z=(0, 1),
                                                               topology=(Periodic, Bounded, Bounded)),

                               coriolis = BetaPlane(latitude=45),
                      background_fields = (u=U,),
                               buoyancy = nothing,
                                tracers = nothing,
                                closure = AnisotropicBiharmonicDiffusivity(νh=1e8)
                           )

# ## Random initial conditions
#
# Our initial condition randomizes `model.velocities.u` and `model.velocities.v`.
# We ensure that both have zero mean for aesthetic reasons.

Ξ(x, y, z) = 1e-2 * model.background_fields.velocities.u.parameters.U * randn()

set!(model, u=Ξ, v=Ξ)

# ## Power method
#
# u(x, t₁) = u₁(x, t) = u₀(x, 0) exp(σ t₁ - i ω t₁)
#
# uᵢ = u₀ exp(σ t) cos(ω t₁)
#
# => u₁ / u₀ = exp(σ t₁ - i ω t₁)
#
# => u₁ / u₀ * exp(- σ t₁) = exp(- i ω t₁)
#
# => log(u₁ / u₀) - σ t₁ = - i ω t₁
#
# => σ t₁ - log(u₁ / u₀) = i ω t₁

u, v, w = model.velocities

ω = ComputedField(∂x(v) - ∂y(u))
E = mean(1/2 * (u^2 + v^2 + w^2), dims=(1, 2, 3))

mutable struct PowerProgress{U}
    starting_energy :: Float64
    starting_time :: Float64
    starting_u :: U
end

function(pp::PowerProgress)(sim)
    compute!(ω)
    compute!(E)

    ΔE = E[1, 1, 1] / pp.starting_energy

    Δt = sim.model.clock.time - pp.starting_time

    growth_rate = log(ΔE) / 2Δt
    growth_time_scale = 1 / growth_rate

    # => σ t₁ - log(u₁ / u₀) = i ω t₁
    #
    # => - i ( log(u₁ / u₀) / t₁ - σ) = ω
    
    u_ratio = interior(sim.model.velocities.u)[32, 32, 1] ./ interior(pp.starting_u)[32, 32, 1]

    mean_u_ratio = u_ratio
    
    instability_frequency = try
        acos(mean_u_ratio * exp(-growth_rate * Δt)) / Δt
    catch
        NaN
    end

    @info @sprintf("σ: %.3e, ω: %.3e", growth_rate, instability_frequency)

    return nothing
end

compute!(E)
progress = PowerProgress(E[1, 1, 1], 0.0, XFaceField(CPU(), model.grid))

function vorticity_threshold(sim; ω_threshold = 1e-5)
    compute!(ω)
    return maximum(abs, interior(ω)) > ω_threshold
end

simulation = Simulation(model, Δt=hour/4, iteration_interval=10, progress=progress)

push!(simulation.stop_criteria, vorticity_threshold)

function rescale!(model; scale=1e-3)
    model.velocities.u.data.parent .*= scale
    model.velocities.v.data.parent .*= scale
    model.velocities.w.data.parent .*= scale
    return nothing
end

x, y, z = nodes(ω)

function visualize!(ω, model, power_iteration)

    Ro = interior(ω)[:, :, 1] / model.coriolis.f₀

    Ro_max = maximum(abs, Ro)
    Ro_lim = 0.8 * Ro_max

    Ro_levels = range(-Ro_lim, stop=Ro_lim, length=21)
    Ro_lim < Ro_max && (Ro_levels = vcat([-Ro_max], Ro_levels, [Ro_max]))

    kwargs = (xlabel="x", ylabel="y", aspectratio=1, linewidth=0, colorbar=true,
              xlims=(-model.grid.Lx/2, model.grid.Lx/2), ylims=(-model.grid.Ly/2, model.grid.Ly/2))

    Ro_plot = contourf(x, y, Ro';
                       color = :balance,
                      levels = Ro_levels,
                       clims = (-Ro_lim, Ro_lim),
                       title = "Rossby number, iteration $power_iteration",
                      kwargs...)

    display(Ro_plot)
end

power_iteration = 1

while true
    global power_iteration

    # Initialize progress
    compute!(E)
    simulation.progress.starting_energy = E[1, 1, 1]
    simulation.progress.starting_time = model.clock.time
    simulation.progress.starting_u.data.parent .= model.velocities.u.data.parent

    run!(simulation)

    @info "Iteration $power_iteration"

    visualize!(ω, model, power_iteration)
    rescale!(simulation.model)

    power_iteration += 1
end
