
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
# Ri = \frac{∂_z B}{(∂_z U)^2}
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

Ri_plot = plot(@. Ri * sech(z / h)^2 / sech(z)^2, z; xlabel="Ri(z)", color=:black, kwargs...) # Ri(z)= ∂_z B / (∂_z U)²; derivatives computed by hand

plot(U_plot, B_plot, Ri_plot, layout=(1, 3), size=(800, 400))

# # Linear Instabilities
#
# The base state ``U(z)``, ``B(z)`` is a solution of the inviscid equations of motion. Whether the base state is 
# stable or not is determined by whether small perturbations about this base state grow or decay. To formalize this, 
# we study the linearized dynamics satisfied by perturbations about the base state:
# ```math
# \partial_t \Phi = L \Phi \, .
# ```
# where ``\Phi = (u, v, w, b)`` is a vector of the perturbation velocities ``u, v, w`` and perturbation buoyancy ``b`` 
# and ``L`` a linear operator that depends on the base state, ``L = L(U(z), B(z))`` (the `background_fields`).
# Eigenanalysis of the linear operator ``L`` determines the stability of the base state, such as the Kelvin-Helmholtz 
# instability. That is, by using the ansantz
# ```math
# \Phi(x, y, z, t) = \phi(x, y, z) \, \exp(\lambda t) \, ,
# ```
# then ``\lambda`` and ``\phi`` are respectivealy eigenvalues and eigenmodes of ``L``,
# ```math
# L \, \phi_j = \lambda \, \phi_j \quad j=1,2,\dots \, .
# ```
# Here we use the convention that the eigenvalues are ordered according to their real part, ``\real(\lambda_1) \ge \real(\lambda_2) \ge \dotsb``.
#
# Two remarks:
# 
# - Oceananigans.jl, does not include a linearized version of the equations, but we can ensure we remain in the linear regime if keep our perturbation fields small thus ensuring that terms quadratic in perturbations are much smaller than all other terms. This way we get approximately: ``\partial_t \Phi \approx L \Phi``.
# - Even with the small-amplitude trick, although Oceananigans.jl will be able to solve the approximately linear equations of motion, it won't allow us to get the linear operator ``L`` and perform eigendecomposition. 
# 
# This last "caveat" can be alleviated using the following algorithm.
#
# # A _Power_ful algorithm
#
# Successive application of ``L`` to a random initial state will render it parallel 
# with eigenmode ``\phi_1``:
# ```math
# \lim_{n \to \infty} L^n \Phi \propto \phi_1 \, .
# ```
# Of course, if ``\phi_1`` is an unstable mode (i.e., ``\sigma_1 = \real(\lambda_1) > 0``), then successive application 
# of ``L`` will lead to exponential amplification. (Or, if ``\sigma_1 < 0``, it will
# lead to exponential decay of ``\Phi`` down to machine precision.) Therefore, after each 
# application of the linear operator ``L`` to our state ``\Phi``, we should rescale the output
# ``L \Phi `` back to a pre-selected amplitude before applying ``L`` again.
# 
# So, we initialize a `simulation` with random initial conditions with amplitude much less than those of
# the base state (which are ``O(1)``). Instead of "applying" ``L`` on our initial state, we evolve the
# (approximately) linear dynamics for interval ``\Delta \tau``. We measure how much the energy has grown 
# during that interval, rescale the perturbations back to original energy amplitude and repeat.
# After some iterations the state will converge to the most unstable eigenmode.
# 
# In summary, each iteration of the power method includes:
# - compute the perturbation energy, ``E_0``,
# - evolve the system for a time-interval ``\Delta \tau``,
# - compute the perturbation energy, ``E_1``,
# - determine the exponential growth of the most unstable mode during the interval ``\Delta \tau`` as  ``\log(E_1 / E_0) / (2 \Delta \tau)``,
# - repeat the above until growth rate converges.
# 
# By fiddling a bit with ``\Delta t`` we can get convergence after only a few iterations.
# 
# Let's apply all these to our example.

# # The model

using Oceananigans.Advection

model = IncompressibleModel(timestepper = :RungeKutta3, 
                              advection = UpwindBiasedFifthOrder(),
                                   grid = grid,
                               coriolis = nothing,
                      background_fields = (u=U, b=B),
                                closure = IsotropicDiffusivity(ν=1e-4, κ=1e-4),
                               buoyancy = BuoyancyTracer(),
                                tracers = :b)

# We have included a "pinch" of viscosity and diffusivity in anticipation of what will follow furtherdown: 
# viscosity and diffusivity will ensure numerical stability when we evolve the unstable mode to the point it becomes nonlinear.

# For this example, we take ``\Delta \tau = 15``.

simulation = Simulation(model, Δt=0.1, stop_iteration=150)
              
              
# Now some helper functions that will be used during for the power method algorithm.
# 
# First a function that evolves the state for ``\Delta \tau`` and measure the energy growth
# over that period.

"""
    grow_instability!(simulation, e)

Grow an instability by running `simulation`.

Estimates the growth rate ``σ`` of the instability
using the fractional change in volume-mean kinetic energy,
over the course of the `simulation`

``
energy(t₀ + Δτ) / energy(t₀) ≈ exp(2 σ Δτ)
``

where ``t₀`` is the starting time of the simulation and ``t₀ + Δτ`` 
the ending time of the simulation. We thus find that the growth rate 
is measured by

``
σ = log(energy(t₀ + Δτ) / energy(t₀)) / (2 * Δτ) .
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
    Δτ = simulation.model.clock.time - t₀

    ## ½(u² + v²) ~ exp(2 σ Δτ)
    σ = growth_rate = log(energy₁ / energy₀) / 2Δτ
    return growth_rate    
end
nothing # hide

# Finally, we write a function that rescales the state. The rescaling is done via measureing the
# kinetic energy and then rescaling back to a prescribed energy value.
# 
# (Measuring the growth via the kinetic energy works fine _unless_ an unstable mode has _only_
# buoynancy structure. In that case, the total perturbation energy will be adequate.)

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

# Some more helper function for the power method,

""" Compute the relative difference between ``σⁿ`` and ``σⁿ⁻¹``, avoiding `NaN`s. """
relative_difference(σ) = length(σ) > 1 ? abs((σ[end] - σ[end-1]) / σ[end-1]) : Inf

""" Check if the growth rate has converged. If the array `σ` has at least 2 elements then return the relative difference between ``σ[end]`` and ``σ[end-1]``. """
convergence(σ) = length(σ) > 1 ? abs((σ[end] - σ[end-1]) / σ[end]) : 9.1e18 # pretty big (not Inf tho)
nothing # hide

# and the main function that performs the power method iteration.
# (Note that `estimate_growth_rate!()` also return the plot of the perturbation field after each iteration.
# This is not a requirement and can be disabled for speeding up the algorigithm but we'll keep it
# here for illustration purposes.)
"""
    estimate_growth_rate!(simulation, energy, ω; convergence_criterion=1e-3)

Estimates the growth rate iteratively until the relative change
in the estimated growth rate ``σ`` falls below `convergence_criterion`.

Returns ``σ``.
"""
function estimate_growth_rate!(simulation, energy, ω, b; convergence_criterion=1e-3)
    σ = []
    movie_frames = []
    
    compute!(ω)
    
    frame = power_method_plot(interior(ω)[:, 1, :], interior(b)[:, 1, :], σ, nothing)
    
    push!(movie_frames, frame)
    
    while convergence(σ) > convergence_criterion
        compute!(energy)

        @info @sprintf("About to start power method iteration %d; kinetic energy: %.2e", length(σ)+1, energy[1, 1, 1])

        push!(σ, grow_instability!(simulation, energy))

        compute!(energy)

        @info @sprintf("Power method iteration %d, kinetic energy: %.2e, σⁿ: %.2e, relative Δσ: %.2e",
                       length(σ), energy[1, 1, 1], σ[end], relative_difference(σ))

        compute!(ω)
        
        frame = power_method_plot(interior(ω)[:, 1, :], interior(b)[:, 1, :], σ, nothing)
        
        push!(movie_frames, frame)
        
        rescale!(simulation.model, energy)
    end

    return σ, movie_frames
end
nothing # hide

# # Eigenplotting
#
# A good algorithm wouldn't be complete without a good visualization,

using Oceananigans.AbstractOperations

u, v, w = model.velocities
b = model.tracers.b

perturbation_vorticity = ComputedField(∂z(u) - ∂x(w))

xF, yF, zF = nodes(perturbation_vorticity)

xC, yC, zC = nodes(b)

eigentitle(σ, t) = length(σ) > 0 ? @sprintf("Iteration #%i; growth rate %.2e", length(σ), σ[end]) : @sprintf("Initial perturbation fields")

function eigenplot(ω, b, σ, t; ω_lim=maximum(abs, ω)+1e-16, b_lim=maximum(abs, b)+1e-16)
    
    kwargs = (xlabel="x", ylabel="z", linewidth=0, label=nothing, color = :balance, aspectratio = 1,)
    
    ω_title(t) = t == nothing ? @sprintf("vorticity") : @sprintf("vorticity at t = %.2f", t)
    
    plot_ω = contourf(xF, zF, clamp.(ω, -ω_lim, ω_lim)';
                      levels = range(-ω_lim, stop=ω_lim, length=20),
                      xlims = (xF[1], xF[grid.Nx]),
                      ylims = (zF[1], zF[grid.Nz]),
                      clims = (-ω_lim, ω_lim),
                      title = ω_title(t), kwargs...)
                      
    b_title(t) = t == nothing ? @sprintf("buoyancy") : @sprintf("buoyancy at t = %.2f", t)
    
    plot_b = contourf(xC, zC, clamp.(b, -b_lim, b_lim)';
                    levels = range(-b_lim, stop=b_lim, length=20),
                    xlims = (xC[1], xC[grid.Nx]),
                    ylims = (zC[1], zC[grid.Nz]),
                    clims = (-b_lim, b_lim),
                    title = b_title(t), kwargs...)
                    
    return plot(plot_ω, plot_b, layout=(1, 2), size=(800, 380))
end

function power_method_plot(ω, b, σ, t)
    plot_growthrates = scatter(σ,
                                xlabel = "Power iteration",
                                ylabel = "Growth rate",
                                 title = eigentitle(σ, nothing),
                                 label = nothing)
                                 
    plot_eigenmode = eigenplot(ω, b, σ, nothing)
    
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

noise(x, y, z) = randn()

set!(model, u=noise, w=noise, b=noise)

rescale!(simulation.model, mean_perturbation_kinetic_energy, target_kinetic_energy=1e-6)

growth_rates, power_method_frames = estimate_growth_rate!(simulation, mean_perturbation_kinetic_energy, perturbation_vorticity, b)

@info "Power iterations converged! Estimated growth rate: $(growth_rates[end])"

# # Powerful convergence
#
# We can animate the power method steps. A scatter plot illustrates how the growth rate converges
# as the power method iterates,

anim_powermethod = @animate for iteration in 1:length(growth_rates)
    plot(power_method_frames[iteration])
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

total_vorticity = ComputedField(∂z(u) + ∂z(model.background_fields.velocities.u) - ∂x(w))
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
# Load it; plot it. First the nonlinear equilibration of the perturbation fields.

using JLD2

file = jldopen(simulation.output_writers[:vorticity].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

@info "Making a neat movie of stratified shear flow..."

anim_perturbations = @animate for (i, iteration) in enumerate(iterations)

    @info "Plotting frame $i from iteration $iteration..."
    
    t = file["timeseries/t/$iteration"]
    ω_snapshot = file["timeseries/ω/$iteration"][:, 1, :]
    b_snapshot = file["timeseries/b/$iteration"][:, 1, :]
    
    eigenplot(ω_snapshot, b_snapshot, nothing, t; ω_lim=1, b_lim=0.1)
end

gif(anim_perturbations, "kelvin_helmholtz_instability_perturbations.gif", fps = 8) # hide

# And then the total vorticity & buoyancy of the fluid.

anim_total = @animate for (i, iteration) in enumerate(iterations)

    @info "Plotting frame $i from iteration $iteration..."
    
    t = file["timeseries/t/$iteration"]
    ω_snapshot = file["timeseries/Ω/$iteration"][:, 1, :]
    b_snapshot = file["timeseries/B/$iteration"][:, 1, :]
    
    eigenplot(ω_snapshot, b_snapshot, nothing, t; ω_lim=1, b_lim=0.05)
end

gif(anim_total, "kelvin_helmholtz_instability_total.gif", fps = 8) # hide
