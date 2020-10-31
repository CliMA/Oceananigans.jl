# # Stratified Kelvin-Helmholtz instability
#
# # The domain 

using Oceananigans

using Plots

grid = RegularCartesianGrid(size=(64, 1, 64), x=(-5, 5), y=(0, 1), z=(-3, 3),
                                  topology=(Periodic, Periodic, Bounded))

# # The background flow
#
# Our background flow constists of linear stratification and constant shear

using Oceananigans.Fields, Oceananigans.Grids

shear_flow(x, y, z, t) = tanh(z)

# Ri = ∂z B / (∂z U)²
stratification(x, y, z, t, p) = p.h * p.Ri * tanh(z / p.h)

Ri = 0.10
h = 1/10

U = BackgroundField(shear_flow)
B = BackgroundField(stratification, parameters=(Ri=Ri, h=h))

z = znodes(Cell, grid)

background_b = @. h * Ri * tanh(z / h)

∂z_U = @. sech(z)^2
∂z_B = @. Ri * sech(z / h)^2

Ri_background = @. ∂z_B / ∂z_U^2

#=
kwargs = (ylabel = z, linewidth = 2)
U_plot = plot(shear_flow.(0, 0, z, 0), z, xlabel="U(z)", kwargs...)
B_plot = plot(background_buoyancy.(0, 0, z, 0), z, xlabel="B(z)", kwargs...)
Ri_plot = plot(background_Ri.(0, 0, z, 0), z, xlabel="B(z)", kwargs...)
base_state = plot([shear_flow.(0, 0, z, 0) background_b Ri_background], z,
                  linewidth=2, xlabel="Ri", ylabel="y",
                  size = (600, 400),
                  label = ["U(z)" "B(z)" "Ri(z)"])

display(base_state)
=#

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

simulation = Simulation(model, Δt=0.1, iteration_interval=20, stop_iteration=200)
                        
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
    ## Initialize
    simulation.model.clock.iteration = 0
    t₀ = simulation.model.clock.time = 0
    e₀ = e[1, 1, 1]

    ## Grow
    run!(simulation)

    ## Analyze
    compute!(e)
    e₁ = e[1, 1, 1]
    Δt = simulation.model.clock.time - t₀

    ## (u² + v²) / 2 ~ exp(2 σ Δt)
    σ = growth_rate = log(e₁ / e₀) / 2Δt

    return growth_rate    
end

# Finally, we write a function that rescales the velocity field
# at the write

"""
    rescale!(model, e; target_kinetic_energy=1e-3)

Rescales all model fields so that `e = target_kinetic_energy`.

The rescaling factor is calculated via

``
r = \\sqrt{e_\\mathrm{target} / e}
``
"""
function rescale!(model, e; target_kinetic_energy=1e-6)
    compute!(e)
    r = sqrt(target_kinetic_energy / e[1, 1, 1])

    model.velocities.u.data.parent .*= r
    model.velocities.v.data.parent .*= r
    model.velocities.w.data.parent .*= r
    model.tracers.b.data.parent .*= r
    return nothing
end

using Printf

relative_change(σⁿ, σⁿ⁻¹) = isfinite(σⁿ) ? abs((σⁿ - σⁿ⁻¹) / σⁿ) : Inf

"""
    estimate_growth_rate!(simulation, e, ω; convergence_criterion=1e-2)

Estimates the growth rate.
"""
function estimate_growth_rate!(simulation, e, ω; convergence_criterion=1e-3)
    σ = [0.0, grow_instability!(simulation, e)]

    while relative_change(σ[end], σ[end-1]) > convergence_criterion

        push!(σ, grow_instability!(simulation, e))

        compute!(e)

        @info @sprintf("*** Power iteration %d, e: %.2e, σⁿ: %.2e, relative Δσ: %.2e",
                       length(σ), e[1, 1, 1], σ[end], relative_change(σ[end], σ[end-1]))

        compute!(ω)
        display(eigenplot!(interior(ω)[:, 1, :], σ, nothing))

        rescale!(simulation.model, e)
        compute!(e)

        @info @sprintf("*** Kinetic energy after rescaling: %.2e", e[1, 1, 1])
                       
    end

    return σ
end

# # Eigenplotting
#
# A good algorithm wouldn't be complete without a good visualization tool,

using Oceananigans.AbstractOperations

u, v, w = model.velocities
b = model.tracers.b

perturbation_vorticity = ComputedField(∂z(u) - ∂x(w))

x, y, z = nodes(perturbation_vorticity)

eigentitle(σ, t) = "Iteration $(length(σ)): most unstable eigenmode"
eigentitle(::Nothing, t) = @sprintf("Vorticity at t = %.2f", t)

eigenplot!(ω, σ, t; ω_lim=maximum(abs, ω)+1e-16) =
    contourf(x, z, clamp.(ω, -ω_lim, ω_lim)';
             color = :balance, aspectratio = 1,
             levels = range(-ω_lim, stop=ω_lim, length=20),
             xlims = (grid.xF[1], grid.xF[grid.Nx]),
             ylims = (grid.zF[1], grid.zF[grid.Nz]),
             clims = (-ω_lim, ω_lim), linewidth = 0,
              size = (600, 200),
             title = eigentitle(σ, t))

# # Rev your engines...
#
# We initialize the power iteration with random noise.
# The amplitude of the initial condition is arbitrary since our algorithm
# will rescale the velocity field iteratively until the simulation's stop_criteria
# is no longer met.

using Random, Statistics, Oceananigans.AbstractOperations

mean_perturbation_energy = mean(1/2 * (u^2 + w^2), dims=(1, 2, 3))

noise(x, y, z) = 1e-4 * randn()

set!(model, u=noise, w=noise, b=noise)

growth_rates = estimate_growth_rate!(simulation, mean_perturbation_energy, perturbation_vorticity)

@info "\n Power iterations converged! Estimated growth rate: $(growth_rates[end]) \n"

# # Plot the result

scatter(filter(σ -> isfinite(σ), growth_rates),
        xlabel = "Power iteration",
        ylabel = "Growth rate",
         label = nothing)

# # The fun part
#
# Now for the fun part: simulating the Kelvin-Helmholtz instability growth and nonlinear equilibration.

estimated_growth_rate = growth_rates[end]

using Oceananigans.OutputWriters

simulation.output_writers[:vorticity] =
    JLD2OutputWriter(model, (ω=perturbation_vorticity,),
                     schedule = TimeInterval(0.10 / estimated_growth_rate),
                     prefix = "kelvin_helmholtz",
                     force = true)

## Prep a fresh run
#
# Initialize a simulation with the eigenmode and run for a few e-folding time ``1/\sigma``.

model.clock.iteration = 0
model.clock.time = 0
simulation.stop_time = 7 / estimated_growth_rate
simulation.stop_iteration = 9.1e18 # pretty big (not Inf tho)

rescale!(simulation.model, mean_perturbation_energy, target_kinetic_energy=1e-4)

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
    
    eigenplot = eigenplot!(ω_snapshot, nothing, t, ω_lim=1)
end

gif(anim, "kelvin_helmholtz.gif", fps = 8) # hide
