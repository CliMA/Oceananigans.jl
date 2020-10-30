# # Barotropic instability in a channel on a β-plane
#
# This example simulates the instability of a boundary-hugging
# barotropic jet (that is, vertically-uniform) on a beta plane.
# 
# The domain 

using Oceananigans

grid = RegularCartesianGrid(size=(64, 64, 1), x=(-2π, 2π), y=(0, 6), z=(0, 1),
                                  topology=(Periodic, Bounded, Bounded))

# For the jet in question, we use "half-Bickley" form

using Oceananigans.Fields

half_bickley(x, y, z, t) = sech(y)^2

U = BackgroundField(half_bickley)

# which looks like

using Oceananigans.Grids
using Plots

y = ynodes(Cell, grid)

background_flow = plot(half_bickley.(0, y, 0, 0), y, label=nothing, xlabel="U(y)", ylabel="y", title="A half-Bickley")
display(background_flow)

# The half-Bickley is well studied, ie...
#
# To lend our problem an Oceananographic flavor, we investigate the instability on the β-plane,
# where the background rotatin rate varies in ``y``:

coriolis = BetaPlane(f₀=1, β=0.1)

# The model

model = IncompressibleModel(timestepper = :RungeKutta3, 
                                   grid = grid,
                               coriolis = coriolis,
                      background_fields = (u=U,),
                                closure = IsotropicDiffusivity(ν=1e-6),
                               buoyancy = nothing,
                                tracers = nothing)

# The "Power method" for diagnosing instability growth-rates and eigenmodes
#
# An "instability" is a small-amplitude solution that develops due to the presence
# of an unstable "basic state" or background flow. Because these solutions are
# small amplitude, we can write them in the form
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

using Random

noise(x, y, z) = randn()

set!(model, u=noise, v=noise)

progress(sim) = @info "i: $(sim.model.clock.iteration), t: $(sim.model.clock.time), max(u): $(maximum(abs, interior(sim.model.velocities.u)))"
simulation = Simulation(model, Δt=0.1, progress=progress, iteration_interval=100)

velocity_exceeds_threshold(sim; threshold=1e-1) = maximum(abs, interior(sim.model.velocities.u)) > threshold

push!(simulation.stop_criteria, velocity_exceeds_threshold)

function grow_instability!(simulation, e)
    e₀ = e[1, 1, 1]
    t₀ = model.clock.time

    run!(simulation)

    compute!(e)
    Δe = e[1, 1, 1] / e₀
    Δt = simulation.model.clock.time - t₀

    growth_rate = log(Δe) / 2Δt

    return growth_rate    
end

function rescale!(velocities; scale=1e-1)
    velocities.u.data.parent .*= scale
    velocities.v.data.parent .*= scale
    velocities.w.data.parent .*= scale
    return nothing
end


function eigenmode!(ω, iteration)
    x, y, z = nodes(ω)
    ω_max = maximum(abs, interior(ω)) + 1e-9
    ω_levels = range(-ω_max, stop=ω_max, length=21)
    ω_plot = contourf(x, y, interior(ω)[:, :, 1]'; color = :balance, levels = ω_levels, clims = (-ω_max, ω_max),
                      title = "Iteration $iteration: most unstable eigenmode") # of the boundary-hugging half-Bickley jet")
    display(ω_plot)
end

using Printf

function compute_growth_rate!(simulation, e, ω)
    σⁿ⁻¹ = 0.0
    σⁿ = grow_instability!(simulation, e)
    iteration = 0

    while true #abs((σⁿ⁻¹ - σⁿ) / σⁿ⁻¹) > 0.1
        σⁿ⁻¹ = σⁿ
        σⁿ = grow_instability!(simulation, e)

        iteration += 1
        @info @sprintf("Power iteration %d, estimated σ: %.2e", iteration, σⁿ)

        eigenmode!(ω, iteration)
        rescale!(simulation.model.velocities)
        simulation.model.clock.time = 0
    end

    return σⁿ
end

using Statistics, Oceananigans.AbstractOperations

u, v, w = model.velocities

# Vorticity...
vorticity = ComputedField(∂x(v) - ∂y(u))
  
# Mean perturbation energy.
mean_perturbation_energy = mean(1/2 * (u^2 + v^2), dims=(1, 2, 3))

compute_growth_rate!(simulation, mean_perturbation_energy, vorticity)
