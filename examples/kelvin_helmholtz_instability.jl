
# # Stratified Kelvin-Helmholtz instability
#
# We simulate Kelvin-Helmholtz instability in two-dimensions in ``x, z``,

using Oceananigans

grid = RegularCartesianGrid(size=(64, 1, 65), x=(-5, 5), y=(0, 1), z=(-5, 5),
                                  topology=(Periodic, Periodic, Bounded))

# # The basic state
#
# We're simulating the instability of a sheared and stably-stratified basic state
# ``U(z)`` and ``B(z)``. Two parameters define our basic state: the Richardson number
# defined as,
#
# ```math
# Ri = \frac{∂_z B}{∂_z U^2}
# ```
#
# and the width of the stratification layer, ``h``.

using Oceananigans.Fields

shear_flow(x, y, z, t) = tanh(z)

stratification(x, y, z, t, p) = p.h * p.Ri * tanh(z / p.h)

U = BackgroundField(shear_flow)
B = BackgroundField(stratification, parameters=(Ri=0.1, h=1/4))

# Our basic state thus has a thin layer of stratification in the center of
# the channel, embedded within a thicker shear layer surrounded by unstratified fluid.

using Plots, Oceananigans.Grids

z = znodes(Cell, grid)

Ri, h = B.parameters

kwargs = (ylabel="z", linewidth=2, label=nothing)

 U_plot = plot(shear_flow.(0, 0, z, 0), z; xlabel="U(z)", kwargs...)
 B_plot = plot([stratification(0, 0, z, 0, (Ri=Ri, h=h)) for z in z], z; xlabel="B(z)", color=:red, kwargs...)
Ri_plot = plot(Ri * sech.(z / h).^2 ./ sech.(z).^2, z; xlabel="Ri(z)", color=:black, kwargs...)

plot(U_plot, B_plot, Ri_plot, layout=(1, 3), size=(800, 400))

# # The model

using Oceananigans.Advection

model = IncompressibleModel(timestepper = :RungeKutta3, 
                              advection = UpwindBiasedFifthOrder(),
                                   grid = grid,
                               coriolis = nothing,
                      background_fields = (u=U, b=B),
                                closure = IsotropicDiffusivity(ν=5e-5, κ=5e-5),
                               buoyancy = BuoyancyTracer(),
                                tracers = :b)

# # A _Power_ful algorithm
#
# We will find the most unstable mode of the Kelvin-Helmholtz instability using the "power method". 
# Linear instabilities, such as the Kelvin-Helmholtz instability, are described by linearized equations about a base state that take the form
# ```math
# \partial_t \Phi = L \Phi \, .
# ```
# In this example, ``\Phi = (u, v, w, b)`` is a vector of the velocities ``u, v, w``, and 
# buoyancy ``b`` and ``L`` is a linear operator that depends on the base state (the `background_fields`) and other problem parameters.
#
# The linear operator ``L`` has eigenmodes ``u_j`` and corresponding and corresponding 
# eigenvalues ``\lambda_j``,
# ```math
# \mathcal{L} \, \phi_j = \lambda \, \phi_j \quad j=1,2,\dots \, .
# ```
# We use the convention that eigenvalues are ordered according to their real part, ``\real(\lambda_1) \ge \real(\lambda_2) \dotsb``.
# 
# Successive application of ``\mathcal{L}`` to a random initial state will render it parallel 
# with eigenmode ``\phi_1``:
# ```math
# \lim_{n \to \infty} \mathcal{L}^n \Phi \propto \phi_1 \, .
# ```
# Of course, if ``\phi_1`` is an unstable mode, i.e., has ``\sigma_1 = \real(\lambda_1) > 0``, then successive application 
# of ``L`` will lead to exponential amplification. (Or, if ``\sigma_1 < 0``, it will
# lead to exponential decay of ``\Phi`` down to machine precision.) Therefore, after applying
# the linear operator ``L`` to our state ``\Phi``, we should rescale ``\Phi`` back to
# a pre-selected amplitude. Oceananigans.jl, does not include a "linear" version of the equations,
# but we can ensure we remain in the "linear" regime if we pick the pre-selected amplitude to 
# be small enough, i.e., ensuring that terms quadratic in perturbations are much smaller than 
# all other terms.
# 
# So, we initialize a model with random initial conditions with amplitude much less than those of
# the base state have amplitude (``O(1)``). For each iteration of the power method includes:
# - compute the perturbation energy, ``E_0``
# - evolve the system for ``\Delta t``
# - compute the perturbation energy, ``E_1``
# - determine the exponential growth of the most unstable mode during interval ``\Delta t`` as
#  ``\sigma = \log(E_1 / E_0) / (2 \Delta t)``.
# - repeat the above until ``\sigma`` converges.

simulation = Simulation(model, Δt=0.1, iteration_interval=20, stop_iteration=100)
                        
"""
    grow_instability!(simulation, e)

Grow an instability by running `simulation`.

Estimates the growth rate ``σ`` of the instability
using the fractional change in volume-mean kinetic energy,
over the course of the `simulation`

``
energy(t₁) / energy(t₀) ≈ exp(2 σ (t₁ - t₀))
``

where ``t₀`` and ``t₁`` are the starting and ending times of the
simulation. We thus find that the growth rate is measured by

``
σ = log(energy(t₁) / energy(t₀)) / (2 * (t₁ - t₀)) .
``
"""
function grow_instability!(simulation, energy)
    ## Initialize
    simulation.model.clock.iteration = 0
    t₀ = simulation.model.clock.time = 0
    compute!(energy)
    energy₀ = energy[1, 1, 1]

    ## Grow
    run!(simulation)

    ## Analyze
    compute!(energy)
    energy₁ = energy[1, 1, 1]
    Δt = simulation.model.clock.time - t₀

    ## (u² + v²) / 2 ~ exp(2 σ Δt)
    σ = growth_rate = log(energy₁ / energy₀) / 2Δt
    return growth_rate    
end
nothing # hide

# Finally, we write a function that rescales the velocity field
# at the write

"""
    rescale!(model, energy; target_kinetic_energy=1e-3)

Rescales all model fields so that `energy = target_kinetic_energy`.
"""
function rescale!(model, energy; target_kinetic_energy=1e-6)
    compute!(energy)

    rescale_factor = √(target_kinetic_energy / energy[1, 1, 1])

    model.velocities.u.data.parent .*= rescale_factor
    model.velocities.v.data.parent .*= rescale_factor
    model.velocities.w.data.parent .*= rescale_factor
    model.tracers.b.data.parent .*= rescale_factor

    return nothing
end

using Printf

""" Compute the relative difference between ``σⁿ`` and ``σⁿ⁻¹``, avoiding `NaN`s. """
relative_difference(σ) = length(σ) > 1 ? abs((σ[end] - σ[end-1]) / σ[end-1]) : Inf

""" Check if the growth rate has converged. If the array `σ` has at least 2 elements then return the relative difference between ``σ[end]`` and ``σ[end-1]``. """
convergence(σ) = length(σ) > 1 ? abs((σ[end] - σ[end-1]) / σ[end]) : 9.1e18 # pretty big (not Inf tho)

"""
    estimate_growth_rate!(simulation, energy, ω; convergence_criterion=1e-2)

Estimates the growth rate iteratively until the relative change
in the estimated growth rate ``σ`` falls below `convergence_criterion`.

Returns ``σ``.
"""
function estimate_growth_rate!(simulation, energy, ω, b; convergence_criterion=1e-3)
    σ = []
    
    plotseries = Array{Any, 1}(undef, 20)  # expect no more than 20 iterations
    iteration = 1
    while convergence(σ) > convergence_criterion

        push!(σ, grow_instability!(simulation, energy))

        compute!(energy)

        @info @sprintf("Power method iteration %d, kinetic energy: %.2e, σⁿ: %.2e, relative Δσ: %.2e",
                       length(σ), energy[1, 1, 1], σ[end], relative_difference(σ))

        compute!(ω)
        plot_powermethoditeration = powermethodplot(interior(ω)[:, 1, :], interior(b)[:, 1, :], σ, nothing)

        rescale!(simulation.model, energy)
        compute!(energy)
        
        plotseries[iteration] = plot_powermethoditeration
        iteration += 1
        
        @info @sprintf("Kinetic energy after rescaling: %.2e", energy[1, 1, 1])
    end

    return σ, plotseries
end
nothing # hide

# # Eigenplotting
#
# A good algorithm wouldn't be complete without a good visualization,

using Oceananigans.AbstractOperations

u, v, w = model.velocities
b = model.tracers.b

perturbation_vorticity = ComputedField(∂z(u) - ∂x(w))

x, y, z = nodes(perturbation_vorticity)
xb, yb, zb = nodes(b)

eigentitle(σ, t) = @sprintf("Iteration #%i; growth rate %.2e", length(σ), σ[end])
eigentitle(::Nothing, t) = @sprintf("Vorticity at t = %.2f", t)

function eigenplot(ω, b, σ, t; ω_lim=maximum(abs, ω)+1e-16, b_lim=maximum(abs, b)+1e-16)
    kwargs = (xlabel="x", ylabel="z", linewidth=0, label=nothing, color = :balance, aspectratio = 1,)
    plot_ω = contourf(x, z, clamp.(ω, -ω_lim, ω_lim)';
                      levels = range(-ω_lim, stop=ω_lim, length=20),
                      xlims = (grid.xF[1], grid.xF[grid.Nx]),
                      ylims = (grid.zF[1], grid.zF[grid.Nz]),
                      clims = (-ω_lim, ω_lim),
                      title = "vorticity", kwargs...)
    plot_b = contourf(xb, zb, clamp.(b, -b_lim, b_lim)';
                    levels = range(-b_lim, stop=b_lim, length=20),
                    xlims = (grid.xC[1], grid.xC[grid.Nx]),
                    ylims = (grid.zC[1], grid.zC[grid.Nz]),
                    clims = (-b_lim, b_lim),
                    title = "buoyancy", kwargs...)
    return plot(plot_ω, plot_b, layout=(1, 2), size=(800, 380))
end

function powermethodplot(ω, b, σ, t)
    plot_growthrates = scatter(σ,
                                xlabel = "Power iteration",
                                ylabel = "Growth rate",
                                 title = title = eigentitle(σ, t),
                                 label = nothing)
    plot_eigenmode = eigenplot(ω, b, σ, t)
    
    return plot(plot_growthrates, plot_eigenmode, layout=@layout([A{0.25h}; B]), size=(800, 600))
end
nothing # hide

# # Rev your engines...
#
# We initialize the power iteration with random noise.
# The amplitude of the initial condition is arbitrary since our algorithm
# will rescale the velocity field iteratively until the simulation's stop_criteria
# is no longer met.

using Random, Statistics, Oceananigans.AbstractOperations

mean_perturbation_kinetic_energy = mean(1/2 * (u^2 + w^2), dims=(1, 2, 3))

noise(x, y, z) = 1e-6 * randn()

set!(model, u=noise, w=noise, b=noise)

growth_rates, powermethod_plotseries = estimate_growth_rate!(simulation, mean_perturbation_kinetic_energy, perturbation_vorticity, b)

@info "Power iterations converged! Estimated growth rate: $(growth_rates[end])"

# # Powerful convergence
#
# We can animate the power method steps. A scatter plot illustrates how the growth rate converges
# as the power method iterates,

anim_powermethod = @animate for iteration in 1:length(growth_rates)
    plot(powermethod_plotseries[iteration])
end

gif(anim_powermethod, "powermethod.gif", fps = 1) # hide

# # Now for the fun part
#
# Now we simulate the nonlinear evolution of the perfect eigenmode
# we've isolated for a few e-folding times ``1/\sigma``,

## Reset the clock
model.clock.iteration = 0
model.clock.time = 0

estimated_growth_rate = growth_rates[end]

simulation.stop_time = 5 / estimated_growth_rate
simulation.stop_iteration = 9.1e18 # pretty big (not Inf tho)

## Rescale the eigenmode
rescale!(simulation.model, mean_perturbation_kinetic_energy, target_kinetic_energy=1e-4)

# Let's save and plot the perturbation vorticity and the
# total vorticity (perturbation + basic state):

using Oceananigans.OutputWriters

total_u = ComputedField(u + model.background_fields.velocities.u)
total_vorticity = ComputedField(∂z(total_u) - ∂x(w))
total_b = ComputedField(b + model.background_fields.tracers.b)

simulation.output_writers[:vorticity] =
    JLD2OutputWriter(model, (ω=perturbation_vorticity, Ω=total_vorticity, b=b, B=total_b),
                     schedule = TimeInterval(0.10 / estimated_growth_rate),
                     prefix = "kelvin_helmholtz_instability",
                     force = true)

# And now we

@info "*** Running a simulation of Kelvin-Helmholtz instability..."
run!(simulation)

# ## Pretty things
#
# Load it; plot it.

using JLD2

file = jldopen(simulation.output_writers[:vorticity].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

@info "Making a neat movie of stratified shear flow..."

anim = @animate for (i, iteration) in enumerate(iterations)

    @info "Plotting frame $i from iteration $iteration..."
    
    t = file["timeseries/t/$iteration"]
    ω_snapshot = file["timeseries/ω/$iteration"][:, 1, :]
    b_snapshot = file["timeseries/b/$iteration"][:, 1, :]
    
    eigenplot(ω_snapshot, b_snapshot, nothing, t; ω_lim=1, b_lim=0.05)
end

gif(anim, "kelvin_helmholtz_instability.gif", fps = 8) # hide
