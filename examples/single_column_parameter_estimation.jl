# # [Single column parameter estimation with Reactant and Enzyme](@id single_column_parameter_estimation_example)
#
# This example shows how to estimate turbulence closure parameters by differentiating
# through an Oceananigans simulation. We set up a "twin experiment": first we run a
# "nature run" with known parameters, then we pretend we don't know those parameters
# and recover them by gradient descent, using gradients computed with
# [Enzyme](https://github.com/EnzymeAD/Enzyme.jl) automatic differentiation
# and compiled with [Reactant](https://github.com/EnzymeAD/Reactant.jl).
#
# This example demonstrates:
#
#   * How to run single column models with `TKEDissipationVerticalDiffusivity` (k-ϵ) on `ReactantState`.
#   * How to make closure parameters differentiable with Reactant.
#   * How to compute the gradient of a cost function with respect to closure parameters.
#   * How to recover the parameters of a nature run with gradient descent.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Enzyme, Reactant, CUDA, CairoMakie"
# ```
#
# Reactant needs CUDA.jl to be loaded to compile Oceananigans' kernels,
# even when it compiles for the CPU.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: VariableStabilityFunctions
using Enzyme
using Reactant
using CUDA
using CairoMakie
using Printf
using Statistics: mean

# ## Two single column models
#
# The k-ϵ model computes an eddy viscosity and an eddy diffusivity from the
# turbulent kinetic energy `e` and its dissipation rate `ϵ`,
#
# ```math
# κu = 𝕊u \frac{e²}{ϵ} \quad \text{and} \quad κc = 𝕊c \frac{e²}{ϵ} \, ,
# ```
#
# where the "stability functions" `𝕊u` and `𝕊c` depend on the local shear and
# stratification. We estimate the two constants that set the magnitude of `𝕊u` and `𝕊c`:
# `Cu₀`, which controls the eddy viscosity, and `Cc₀`, which controls the eddy diffusivity.
#
# Our "observations" come from two columns: one mixed by a wind stress, and one mixed
# by convection driven by surface cooling.
#
# To differentiate with respect to the closure parameters we store the closure's numbers as
# Reactant numbers with `Reactant.to_rarray(closure; track_numbers=Number)`. Otherwise,
# Reactant treats them as constants when it compiles the model. For the same reason, we
# represent the surface fluxes with `Field`s: that way the wind and convection columns have
# the same type and can share compiled code, even though their forcing differs.

grid = RectilinearGrid(ReactantState(), size=64, z=(-128, 0), topology=(Flat, Flat, Bounded))

function surface_flux(value)
    flux = Field{Center, Center, Nothing}(grid)
    set!(flux, value)
    return flux
end

# By default, `VariableStabilityFunctions` computes `𝕊u₀` from `Cu₀`. We hold it fixed instead,
# so that the parameters we differentiate with respect to are the only ones that change.

𝕊u₀ = VariableStabilityFunctions().𝕊u₀

function column_model(Cu₀, Cc₀; τˣ=0, Jᵇ=0)
    stability_functions = VariableStabilityFunctions(; Cu₀, Cc₀, 𝕊u₀)
    closure = TKEDissipationVerticalDiffusivity(; stability_functions)
    closure = Reactant.to_rarray(closure; track_numbers=Number)

    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(surface_flux(τˣ)))
    b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(surface_flux(Jᵇ)))

    return HydrostaticFreeSurfaceModel(grid; closure,
                                       coriolis = FPlane(f=1e-4),
                                       tracers = :b,
                                       buoyancy = BuoyancyTracer(),
                                       boundary_conditions = (u=u_bcs, b=b_bcs))
end

column_models(Cu₀, Cc₀) = (wind = column_model(Cu₀, Cc₀; τˣ=-1e-4),
                           convection = column_model(Cu₀, Cc₀; Jᵇ=5e-8))

# Both columns start at rest with constant stratification, and run for 12 hours.
# The k-ϵ equations need a fairly short time step: with `Δt = 10minutes`, for example, the
# solution becomes noisy and the cost function jagged.

N² = 1e-5
bᵢ = set!(CenterField(grid), z -> N² * z)

Δt = 1minute
Nt = 720

function run_column!(model, bᵢ, Δt, Nt)
    set!(model, u=0, v=0, b=bᵢ)
    @trace track_numbers=false for n = 1:Nt
        time_step!(model, Δt)
    end
    return nothing
end

# ## The nature run
#
# The nature run uses the default parameters, `Cu₀ = 0.1067` and `Cc₀ = 0.1120`.
# We compile `run_column!` once and use it for both columns.

θ★ = (0.1067, 0.1120)
nature = column_models(θ★...)

compiled_run_column! = @compile raise=true raise_first=true sync=true run_column!(nature.wind, bᵢ, Δt, Nt)

for model in nature
    compiled_run_column!(model, bᵢ, Δt, Nt)
end

# We copy the final state of each column of the nature run into "observations",

function observe(model)
    u★ = XFaceField(grid)
    v★ = YFaceField(grid)
    b★ = CenterField(grid)
    interior(u★) .= interior(model.velocities.u)
    interior(v★) .= interior(model.velocities.v)
    interior(b★) .= interior(model.tracers.b)
    return (; u★, v★, b★)
end

observations = map(observe, nature)

# ## Cost function and gradient
#
# The cost of each column is the normalized mean square difference between
# its final state and the observations. The total cost is the sum over both columns.

function column_cost(model, bᵢ, observations, Δt, Nt)
    run_column!(model, bᵢ, Δt, Nt)

    u, v = model.velocities
    b = model.tracers.b
    u★, v★, b★ = observations

    U² = 1e-2
    B² = (N² * 10)^2

    Ju = mean((interior(u) .- interior(u★)).^2) / U²
    Jv = mean((interior(v) .- interior(v★)).^2) / U²
    Jb = mean((interior(b) .- interior(b★)).^2) / B²

    return Ju + Jv + Jb
end

# To compute the gradient we differentiate `column_cost` in reverse mode.
# Enzyme accumulates the derivatives of the cost with respect to every differentiable
# number in the model, including the closure parameters, into a "shadow" model.

function column_cost_and_shadow!(model, shadow, bᵢ, observations, Δt, Nt)
    mode = Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal)
    _, J = Enzyme.autodiff(mode, column_cost, Enzyme.Active,
                           Enzyme.Duplicated(model, shadow),
                           Enzyme.Const(bᵢ),
                           Enzyme.Const(observations),
                           Enzyme.Const(Δt),
                           Enzyme.Const(Nt))
    return J
end

# We compile the cost and its gradient using a model with an initial guess for the parameters.
# Compiling the gradient takes a few minutes.

θ₀ = (0.07, 0.16)
guess = column_models(θ₀...)
shadow = Enzyme.make_zero(guess.wind)

compiled_column_cost = @compile raise=true raise_first=true sync=true column_cost(guess.wind, bᵢ, observations.wind, Δt, Nt)
compiled_column_cost_and_shadow! = @compile raise=true raise_first=true sync=true column_cost_and_shadow!(guess.wind, shadow, bᵢ, observations.wind, Δt, Nt)

# Next we define functions that return the cost and its gradient for parameters `θ = (Cu₀, Cc₀)`.
# The two columns share parameters, so we sum their contributions to the gradient.

function J(θ)
    models = column_models(θ...)
    column_costs = map(models, observations) do model, obs
        Float64(compiled_column_cost(model, bᵢ, obs, Δt, Nt))
    end
    return sum(column_costs)
end

function cost_and_gradient(θ)
    models = column_models(θ...)
    Jθ = 0.0
    ∇J = zeros(2)

    for (model, obs) in zip(models, observations)
        shadow = Enzyme.make_zero(model)
        Jθ += Float64(compiled_column_cost_and_shadow!(model, shadow, bᵢ, obs, Δt, Nt))
        ∇J[1] += Float64(shadow.closure.stability_functions.Cu₀)
        ∇J[2] += Float64(shadow.closure.stability_functions.Cc₀)
    end

    return Jθ, ∇J
end

# Let's compare the gradient with a centered finite difference,

J₀, ∇J₀ = cost_and_gradient(θ₀)

δ = 1e-4
∂J∂Cu₀ = (J((θ₀[1] + δ, θ₀[2])) - J((θ₀[1] - δ, θ₀[2]))) / 2δ
∂J∂Cc₀ = (J((θ₀[1], θ₀[2] + δ)) - J((θ₀[1], θ₀[2] - δ))) / 2δ

@info @sprintf("Enzyme:            ∂J/∂Cu₀ = %.4f, ∂J/∂Cc₀ = %.4f", ∇J₀...)
@info @sprintf("Finite difference: ∂J/∂Cu₀ = %.4f, ∂J/∂Cc₀ = %.4f", ∂J∂Cu₀, ∂J∂Cc₀)

# ## Gradient descent
#
# We minimize the cost with gradient descent. Each iteration steps along `-∇J` with a
# step size chosen by a backtracking line search: we halve the step until the
# cost decreases sufficiently, then try a longer step at the next iteration.

function gradient_descent(θ₀; iterations=12, α=0.5)
    θ = collect(θ₀)
    history = [(θ=copy(θ), J=J(θ))]

    for n = 1:iterations
        Jⁿ, ∇J = cost_and_gradient(θ)

        θ′ = θ .- α .* ∇J
        J′ = J(θ′)
        while J′ > Jⁿ - 1e-4 * α * sum(abs2, ∇J)
            α /= 2
            θ′ = θ .- α .* ∇J
            J′ = J(θ′)
        end

        θ .= θ′
        α *= 2
        push!(history, (θ=copy(θ), J=J′))
        @info @sprintf("iteration %2d: J = %.3e, Cu₀ = %.4f, Cc₀ = %.4f", n, J′, θ...)
    end

    return history
end

history = gradient_descent(θ₀)
θ̂ = history[end].θ

@info @sprintf("Estimated: Cu₀ = %.4f, Cc₀ = %.4f (nature run: Cu₀ = %.4f, Cc₀ = %.4f)", θ̂..., θ★...)

# ## Visualizing the parameter estimation
#
# To see the problem that gradient descent solves, we map the cost function over
# the parameter space,

Cu₀s = range(0.05, 0.17, length=9)
Cc₀s = range(0.05, 0.17, length=9)
Js = [J((Cu₀, Cc₀)) for Cu₀ in Cu₀s, Cc₀ in Cc₀s]

# Finally, we compare the nature run with the model at the initial guess
# and at the estimated parameters.

function final_state(θ)
    models = column_models(θ...)
    for model in models
        compiled_run_column!(model, bᵢ, Δt, Nt)
    end
    u = Array(interior(models.wind.velocities.u))[:]
    v = Array(interior(models.wind.velocities.v))[:]
    bʷ = Array(interior(models.wind.tracers.b))[:]
    bᶜ = Array(interior(models.convection.tracers.b))[:]
    return (; u, v, bʷ, bᶜ)
end

nature_state = final_state(θ★)
guess_state = final_state(θ₀)
estimated_state = final_state(θ̂)

z = Array(znodes(grid, Center()))

fig = Figure(size=(1200, 800))
top = fig[1, 1] = GridLayout()
bottom = fig[2, 1] = GridLayout()

ax = Axis(top[1, 1]; xlabel="Cu₀", ylabel="Cc₀", title="Cost function and gradient descent path")
hm = heatmap!(ax, Cu₀s, Cc₀s, log10.(Js); colormap=:deep)
Colorbar(top[1, 2], hm; label="log₁₀ J")
Cu₀_path = [h.θ[1] for h in history]
Cc₀_path = [h.θ[2] for h in history]
scatterlines!(ax, Cu₀_path, Cc₀_path; color=:orange, markersize=8, label="gradient descent")
scatter!(ax, [θ★[1]], [θ★[2]]; color=:red, marker=:star5, markersize=20, label="nature run")
axislegend(ax, position=:rt)

ax = Axis(top[1, 3]; xlabel="Iteration", ylabel="J", yscale=log10, title="Cost")
scatterlines!(ax, 0:length(history)-1, [h.J for h in history])

axu = Axis(bottom[1, 1]; xlabel="u (m s⁻¹)", ylabel="z (m)", title="Wind: u")
axv = Axis(bottom[1, 2]; xlabel="v (m s⁻¹)", title="Wind: v")
axbʷ = Axis(bottom[1, 3]; xlabel="b (m s⁻²)", title="Wind: b", xticks=LinearTicks(3))
axbᶜ = Axis(bottom[1, 4]; xlabel="b (m s⁻²)", title="Convection: b", xticks=LinearTicks(3))

for (state, label, style) in ((nature_state, "nature run", :solid),
                              (guess_state, "initial guess", :dot),
                              (estimated_state, "estimated", :dash))
    lines!(axu, state.u, z; label, linestyle=style, linewidth=3)
    lines!(axv, state.v, z; label, linestyle=style, linewidth=3)
    lines!(axbʷ, state.bʷ, z; label, linestyle=style, linewidth=3)
    lines!(axbᶜ, state.bᶜ, z; label, linestyle=style, linewidth=3)
end

for ax in (axu, axv, axbʷ, axbᶜ)
    ylims!(ax, -80, 0)
end

axislegend(axu, position=:lb)

current_figure() #hide
