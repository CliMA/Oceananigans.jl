# # Barotropic instability in a channel on a β-plane
#
# This example calculates the growth rate of the "barotropic" (that is
# vertically-uniform) instability of a boundary-hugging jet on a ``β``-plane.
#
# # The "power method" for calculating growth rates and eigenmodes
#
# An "instability" is a small-amplitude solution that develops due to the presence
# of an unstable "basic state" or background flow. Because these solutions are
# small amplitude, we express any of the prognostic variables as
#
# ```math
# u = û exp(σ t)
# ```
#
# where ``σ`` is the "growth rate" of the instability. Our object is to
# compute ``σ`` and also to get a feel for ``û``, which represents the 
# "eigenmode", or the spatial structure of the instability.
#
# The power method iteratively simulates the growth of the instability, using
# a "rescaling" method to successively isolate the instability from other motions
# that develop during a simulation of the fully nonlinear equations.
#
# For this we design a criterion for stopping a simulation based on the amplitude
# of the perturbation ``u``-velocity field:

# 
# # The domain 

using Oceananigans

grid = RegularCartesianGrid(size=(64, 64, 1), x=(-2π, 2π), y=(-2, 5), z=(0, 1),
                                  topology=(Periodic, Bounded, Bounded))

# # The background flow
#
# Our background flow is a boundary-hugging "Bickley jet":

using Oceananigans.Fields

bickley_jet(x, y, z, t) = sech(y)^2

U = BackgroundField(bickley_jet)

# which looks like

using Oceananigans.Grids
using Plots

y = ynodes(Cell, grid)

background_flow = plot(bickley_jet.(0, y, 0, 0), y, label=nothing,
                       linewidth=2, xlabel="U(y)", ylabel="y",
                       title="Boundary-hugging Bickley jet")

display(background_flow)

# The Bickley jet is well studied, ie...
# But we usually don't put it right on a boundary.
#
# # The ``β``-plane
#
# To season our problem with Oceananographic flavor, we investigate the instability
# on the ``β``-plane, where the background rotation rate varies in ``y``:

coriolis = BetaPlane(f₀=1, β=0.01)

# Note that instability is suppressed if ``β > U''(y) \approx 1.2`` [cite Rayleigh-Kuo].

# # The model

using Oceananigans.Advection

model = IncompressibleModel(timestepper = :RungeKutta3, 
                              advection = UpwindBiasedFifthOrder(),
                                   grid = grid,
                               coriolis = coriolis,
                      background_fields = (u=U,),
                                closure = IsotropicDiffusivity(ν=1e-6),
                               buoyancy = nothing,
                                tracers = nothing)

# # A _Power_ful algorithm
#
# We set up an algorithm that rescales the velocity field whenever
# the ``u``-component of the velocity field exeeds some threshold.

velocity_exceeds_threshold(sim; threshold=1e-2) = maximum(abs, interior(sim.model.velocities.u)) > threshold

simulation = Simulation(model, Δt=0.1, iteration_interval=100,
                        progress=sim -> @info("Model iteration: $(sim.model.clock.iteration)"))

push!(simulation.stop_criteria, velocity_exceeds_threshold)

# Next, we define a function that simulates instability growth and returns an
# estimated growth rate.

"""
    grow_instability!(simulation, e)

Grow an instability by running `simulation`.

Estimates the growth rate ``σ`` of the instability
using the fractional change in volume-mean kinetic energy,
over the course of the `simulation`

``
e(t₁) / e(t₀) ≈ exp(2 σ (t₁ - t₀))
``

where ``t₀`` and ``t₁`` are the starting and ending times of the
simulation. We thus find that

``
σ ≈ log(e(t₁) / e(t₀)) / (2 * (t₁ - t₀)) .
``
"""
function grow_instability!(simulation, e)
    simulation.model.clock.iteration = 0
    t₀ = simulation.model.clock.time = 0
    e₀ = e[1, 1, 1]

    run!(simulation)

    compute!(e)
    e₁ = e[1, 1, 1]
    Δt = simulation.model.clock.time - t₀

    σ = growth_rate = log(e₁ / e₀) / 2Δt

    return growth_rate    
end

# Finally, we write a function that rescales the velocity field
# at thewrite 

"""
    rescale!(velocities; factor=1e-1)

"""
function rescale!(velocities; factor=1e-1)
    velocities.u.data.parent .*= factor
    velocities.v.data.parent .*= factor
    velocities.w.data.parent .*= factor
    return nothing
end

using Printf

relative_change(σⁿ, σⁿ⁻¹) = isfinite(σⁿ) ? abs((σⁿ - σⁿ⁻¹) / σⁿ) : Inf

"""
    estimate_growth_rate!(simulation, e, ω; convergence_criterion=1e-2)

Computes 
"""
function estimate_growth_rate!(simulation, e, ω; convergence_criterion=1e-1)
    σ = [0.0, grow_instability!(simulation, e)]

    while relative_change(σ[end], σ[end-1]) > convergence_criterion

        push!(σ, grow_instability!(simulation, e))

        @info @sprintf("*** Power iteration %d, σⁿ: %.2e, relative Δσ: %.2e",
                       length(σ), σ[end], relative_change(σ[end], σ[end-1]))

        compute!(ω)
        display(eigenplot!(interior(ω)[:, :, 1], σ, nothing))

        rescale!(simulation.model.velocities)
    end

    return σ
end

# # Eigenplotting
#
# A good algorithm wouldn't be complete without a good visualization tool,

using Oceananigans.AbstractOperations

u, v, w = model.velocities

vorticity = ComputedField(∂x(v) - ∂y(u))

x, y, z = nodes(vorticity)

eigentitle(σ, t) = "Iteration $(length(σ)): most unstable eigenmode"
eigentitle(::Nothing, t) = @sprintf("Vorticity at t = %.2f", t)

function eigenplot!(ω, σ, t; ω_lim=maximum(abs, ω)+1e-16)
    
    background_flow = plot(bickley_jet.(0, y, 0, 0), y, label=nothing,
                           ylims=(grid.yF[1], grid.yF[grid.Ny]), xlabel="U(y)", ylabel="y")

    ω_contours = contourf(x, y, clamp.(ω, -ω_lim, ω_lim)';
                          color = :balance, aspectratio = 1,
                          levels = range(-ω_lim, stop=ω_lim, length=21),
                          xlims = (grid.xF[1], grid.xF[grid.Nx]),
                          ylims = (grid.yF[1], grid.yF[grid.Ny]),
                          clims = (-ω_lim, ω_lim), linewidth = 0)
                          
    eigenplot = plot(background_flow, ω_contours,
                     layout = Plots.grid(1, 2, widths=[0.2, 0.8]),
                     link = :y, size = (600, 200),
                     title = ["Background flow" eigentitle(σ, t)])

    return eigenplot
end

# # Rev your engines...
#
# We initialize the power iteration with random noise.
# The amplitude of the initial condition is arbitrary since our algorithm
# will rescale the velocity field iteratively until the simulation's stop_criteria
# is no longer met.

using Random, Statistics, Oceananigans.AbstractOperations

mean_perturbation_energy = mean(1/2 * (u^2 + v^2), dims=(1, 2, 3))

noise(x, y, z) = randn()

set!(model, u=noise, v=noise)

growth_rates = estimate_growth_rate!(simulation, mean_perturbation_energy, vorticity)

@info "\n Power iterations converged! Estimated growth rate: $(growth_rates[end]) \n"

# # Plot the result

scatter(filter(σ -> isfinite(σ) && σ > 0, growth_rates),
        xlabel = "Power iteration",
        ylabel = "Growth rate",
        yscale = :log10,
         label = nothing)

# # The fun part
#
# Now for the fun part: simulating the transition to turbulence.

estimated_growth_rate = growth_rates[end]

using Oceananigans.OutputWriters

simulation.output_writers[:vorticity] = JLD2OutputWriter(model, (ω=vorticity,),
                                                         schedule = IterationInterval(20),
                                                         prefix = "barotropic_instability",
                                                         force = true)

pop!(simulation.stop_criteria) # Remove the vorticity_exceeds_threshold stop_criteria

model.clock.iteration = 0
model.clock.time = 0
simulation.stop_iteration = 2000 #stop_time = 10 / growth_rates[end]

@info "*** Running a simulation of barotropic instability..."

rescale!(model.velocities, factor=10)
run!(simulation)

# ## Visualizing the results
#
# We load the output and make a movie.

using JLD2

file = jldopen(simulation.output_writers[:vorticity].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

@info "Making a neat movie of barotropic instability..."

anim = @animate for (i, iteration) in enumerate(iterations)

    @info "Plotting frame $i from iteration $iteration..."
    
    t = file["timeseries/t/$iteration"]
    ω_snapshot = file["timeseries/ω/$iteration"][:, :, 1]

    eigenplot = eigenplot!(ω_snapshot, nothing, t, ω_lim=1)
end

gif(anim, "barotropic_instability.gif", fps = 8) # hide
