# # [Single column parameter estimation with Reactant and Enzyme](@id single_column_parameter_estimation_example)
#
# This example shows how to estimate turbulence closure parameters and surface fluxes
# by differentiating through an Oceananigans simulation. We set up a "twin experiment":
# first we run a "nature run" with known parameters, then we pretend we don't know those
# parameters and recover them by minimizing the mismatch between the model and the nature run.
# We compute gradients with [Enzyme](https://github.com/EnzymeAD/Enzyme.jl) automatic
# differentiation, compiled by [Reactant](https://github.com/EnzymeAD/Reactant.jl).
#
# This example demonstrates:
#
#   * How to run a single column model with `TKEDissipationVerticalDiffusivity` (k-ϵ) on `ReactantState`.
#   * How to assign closure parameters and boundary fluxes to a model inside a differentiable function.
#   * How to compute gradients with respect to closure parameters and boundary fluxes.
#   * How to recover the parameters of a nature run with a gradient-based optimizer.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Enzyme, Reactant, CUDA, Optim, CairoMakie"
# ```
#
# Reactant needs CUDA.jl to be loaded to compile Oceananigans' kernels,
# even when it compiles for the CPU.
#
# Reactant can run with one of two runtimes, PJRT (the default) or IFRT. Oceananigans'
# Reactant tests use IFRT. To use IFRT, add the following to the `Project.toml` or
# `LocalPreferences.toml` of your project, and restart Julia:
#
# ```toml
# [preferences.Reactant]
# xla_runtime = "IFRT"
# ```

using Oceananigans
using Oceananigans.Units
using Enzyme
using Reactant
using CUDA
using CairoMakie
using Printf

# A single column is small, so we run Reactant on the CPU even when a GPU is available,

Reactant.set_default_backend("cpu")

# ## A single column model
#
# We simulate a column of ocean mixed by a surface wind stress `τˣ` and cooled by
# a surface buoyancy flux `Jᵇ`. The k-ϵ model computes an eddy viscosity and an eddy
# diffusivity from the turbulent kinetic energy `e` and its dissipation rate `ϵ`,
#
# ```math
# κu = 𝕊u \frac{e²}{ϵ} \quad \text{and} \quad κc = 𝕊c \frac{e²}{ϵ} \, ,
# ```
#
# where the "stability functions" `𝕊u` and `𝕊c` depend on the local shear and
# stratification. We estimate two constants that set the magnitude of the stability functions,
# `Cu₀` for the eddy viscosity and `Cc₀` for the eddy diffusivity, together with the surface
# fluxes `τˣ` and `Jᵇ`.
#
# To assign new parameters inside a compiled function, the closure's numbers must be
# Reactant numbers, which we obtain with `Reactant.to_rarray(closure; track_numbers=Number)`.
# Similarly, we represent the surface fluxes with `Field`s, whose values we can change.

using Oceananigans.Architectures: ReactantState

grid = RectilinearGrid(ReactantState(), size=64, z=(-128, 0), topology=(Flat, Flat, Bounded))
closure = Reactant.to_rarray(TKEDissipationVerticalDiffusivity(); track_numbers=Number)

τˣ = Field{Face, Center, Nothing}(grid)
Jᵇ = Field{Center, Center, Nothing}(grid)
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τˣ))
b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Jᵇ))

model = HydrostaticFreeSurfaceModel(grid; closure,
                                    coriolis = FPlane(f=1e-4),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (u=u_bcs, b=b_bcs))

# ## Assigning parameters
#
# We work with parameters normalized by typical values, so that every parameter is order one,

scales = (Cu₀ = 0.1, Cc₀ = 0.1, τˣ = 1e-4, Jᵇ = 1e-8)

# The function below assigns `normalized_parameters` to the model. It builds a new closure with
# the `VariableStabilityFunctions` constructor, which also computes another constant, `𝕊u₀`, from `Cu₀`.
# Because we call the constructor inside the function we differentiate, the gradient accounts
# for the dependence of `𝕊u₀` on `Cu₀`.

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: VariableStabilityFunctions
const vitd = VerticallyImplicitTimeDiscretization()

function assign_parameters!(model, normalized_parameters)
    Cu₀ = scales.Cu₀ * normalized_parameters.Cu₀
    Cc₀ = scales.Cc₀ * normalized_parameters.Cc₀
    FT = typeof(Cu₀)

    stability_functions = VariableStabilityFunctions(FT; Cu₀, Cc₀)
    model.closure = TKEDissipationVerticalDiffusivity(vitd, FT; stability_functions)

    τˣ = model.velocities.u.boundary_conditions.top.condition
    Jᵇ = model.tracers.b.boundary_conditions.top.condition
    parent(τˣ) .= scales.τˣ * normalized_parameters.τˣ
    parent(Jᵇ) .= scales.Jᵇ * normalized_parameters.Jᵇ

    return nothing
end

# We represent parameters with a `NamedTuple` of Reactant numbers,

reactant_parameters(values) = NamedTuple{keys(scales)}(Tuple(Reactant.ConcreteRNumber.(values)))

# ## Running the column
#
# Each run starts from rest with constant stratification and lasts 12 hours. Because we reuse the same
# model for every run, we first `reset!` the model, which zeros its fields (including `e` and `ϵ`),
# its tendencies, and the closure's state saved from previous time steps, so that every run starts
# from the same state.
# The k-ϵ equations need a fairly short time step: with `Δt = 10minutes`, for example, the solution
# becomes noisy.

using Oceananigans.Models: reset!

N² = 1e-5
bᵢ = set!(CenterField(grid), z -> N² * z)

Δt = 1minute
Nt = 720

function run_column!(model, normalized_parameters, bᵢ, Δt, Nt)
    assign_parameters!(model, normalized_parameters)

    reset!(model)
    set!(model, b=bᵢ)

    @trace track_numbers=false for n = 1:Nt
        time_step!(model, Δt)
    end

    return nothing
end

# ## The nature run
#
# The nature run uses the default closure parameters `Cu₀ = 0.1067` and `Cc₀ = 0.1120`,
# a wind stress `τˣ = -10⁻⁴ m² s⁻²`, and a cooling buoyancy flux `Jᵇ = 2 × 10⁻⁸ m² s⁻³`.
# We collect the normalized parameters in the vector `θ = (Cu₀, Cc₀, τˣ, Jᵇ) ./ scales`,
# and convert them to Reactant numbers with `reactant_parameters`.

θ★ = [1.067, 1.120, -1, 2]

compiled_run_column! = @compile raise=true raise_first=true sync=true run_column!(
    model, reactant_parameters(θ★), bᵢ, Δt, Nt)

compiled_run_column!(model, reactant_parameters(θ★), bᵢ, Δt, Nt)

# We save the final state of the nature run as our "observations",

u★ = XFaceField(grid)
v★ = YFaceField(grid)
b★ = CenterField(grid)
set!(u★, model.velocities.u)
set!(v★, model.velocities.v)
set!(b★, model.tracers.b)

obs = (; u★, v★, b★)

# ## Cost function and its gradient
#
# The cost function is the normalized mean square difference between
# the final state of the model and the observations,

using Statistics: mean

function cost(normalized_parameters, model, bᵢ, obs, Δt, Nt)
    run_column!(model, normalized_parameters, bᵢ, Δt, Nt)

    u, v = model.velocities
    b = model.tracers.b
    u★, v★, b★ = obs

    U² = 1e-2
    B² = (N² * 10)^2

    𝒥u = mean((u - u★)^2) / U²
    𝒥v = mean((v - v★)^2) / U²
    𝒥b = mean((b - b★)^2) / B²

    return 𝒥u + 𝒥v + 𝒥b
end

# To compute the gradient, we differentiate `cost` in reverse mode. Enzyme accumulates the
# gradient of the cost with respect to the parameters into the "shadow" parameters `cost_gradient`.
# The model is also mutated by `cost`, so we give it a shadow as well.

function cost_and_gradient!(cost_gradient, parameters, model, shadow, bᵢ, obs, Δt, Nt)
    mode = Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal)
    _, 𝒥 = Enzyme.autodiff(mode, cost, Enzyme.Active,
                           Enzyme.Duplicated(parameters, cost_gradient),
                           Enzyme.Duplicated(model, shadow),
                           Enzyme.Const(bᵢ),
                           Enzyme.Const(obs),
                           Enzyme.Const(Δt),
                           Enzyme.Const(Nt))
    return 𝒥
end

# We start from an initial guess that's quite different from the nature run,

θ₀ = [0.6, 1.8, -0.5, 0.5]

# and compile the cost and its gradient. Compiling the gradient takes a few minutes.

shadow = Enzyme.make_zero(model)
cost_gradient = reactant_parameters(zeros(4))

compiled_cost = @compile raise=true raise_first=true sync=true cost(
    reactant_parameters(θ₀), model, bᵢ, obs, Δt, Nt)

compiled_cost_and_gradient! = @compile raise=true raise_first=true sync=true cost_and_gradient!(
    cost_gradient, reactant_parameters(θ₀), model, shadow, bᵢ, obs, Δt, Nt)

𝒥(θ) = Float64(compiled_cost(reactant_parameters(θ), model, bᵢ, obs, Δt, Nt))

function cost_and_gradient(θ)
    shadow = Enzyme.make_zero(model)
    cost_gradient = reactant_parameters(zeros(4))
    𝒥θ = compiled_cost_and_gradient!(cost_gradient, reactant_parameters(θ), model, shadow, bᵢ, obs, Δt, Nt)
    return Float64(𝒥θ), Float64.(collect(cost_gradient))
end

# ## Minimizing the cost
#
# We minimize the cost with the [BFGS](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm)
# algorithm implemented by [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl). BFGS is a quasi-Newton
# flavor of gradient descent: rather than stepping along `-∇𝒥`, it steps along `-H ∇𝒥`, where `H` is an
# estimate of the inverse Hessian of `𝒥` that BFGS builds from the change in the gradient between iterations.
# Optim needs the cost `𝒥`, and a function that computes the gradient in place,

using Optim

function ∇𝒥!(G, θ)
    _, ∇𝒥θ = cost_and_gradient(θ)
    G .= ∇𝒥θ
    return G
end

options = Optim.Options(iterations=10, store_trace=true, extended_trace=true)
bfgs_result = optimize(𝒥, ∇𝒥!, θ₀, BFGS(), options)

# The trace records the parameters and cost at each iteration,

history = [(θ=θ, 𝒥=𝒥θ) for (θ, 𝒥θ) in zip(Optim.x_trace(bfgs_result), Optim.f_trace(bfgs_result))]

for (n, h) in enumerate(history)
    @info @sprintf("iteration %2d: 𝒥 = %.2e, θ / θ★ = %s", n - 1, h.𝒥, string(round.(h.θ ./ θ★, digits=3)))
end

# ## Visualizing the parameter estimation
#
# We compute the final state of the model at each iteration,

z = Array(znodes(grid, Center()))

function final_state(θ)
    compiled_run_column!(model, reactant_parameters(θ), bᵢ, Δt, Nt)
    u = Array(interior(model.velocities.u))[:]
    v = Array(interior(model.velocities.v))[:]
    b = Array(interior(model.tracers.b))[:] .- N² .* z
    return (; u, v, b)
end

nature_state = final_state(θ★)
iteration_states = [final_state(h.θ) for h in history]

# and plot the parameters, the cost, and the profiles of velocity and the buoyancy
# anomaly `b - N² z`. We plot the state of the optimization at iteration `n`, starting
# with the final iteration.

fig = Figure(size=(1200, 800))
top = fig[2, 1] = GridLayout()
bottom = fig[3, 1] = GridLayout()

n = Observable(length(history))

title = @lift @sprintf("Iteration %d, 𝒥 = %.1e", $n - 1, history[$n].𝒥)
Label(fig[1, 1], title, fontsize=20, tellwidth=false)

iterations = 0:length(history)-1
labels = ["Cu₀", "Cc₀", "τˣ", "Jᵇ"]

ax = Axis(top[1, 1]; xlabel="Iteration", ylabel="Parameter / nature run value", title="Parameters")
hlines!(ax, 1; color=:gray, linestyle=:dash)
for i in 1:4
    ratio = [h.θ[i] / θ★[i] for h in history]
    points = @lift Point2f.(iterations[1:$n], ratio[1:$n])
    scatterlines!(ax, points; label=labels[i])
end
xlims!(ax, -0.5, length(history) - 0.5)
ylims!(ax, 0, 2)
axislegend(ax, position=:rt)

costs = [h.𝒥 for h in history]
ax = Axis(top[1, 2]; xlabel="Iteration", ylabel="𝒥", yscale=log10, title="Cost")
cost_points = @lift Point2f.(iterations[1:$n], costs[1:$n])
scatterlines!(ax, cost_points)
xlims!(ax, -0.5, length(history) - 0.5)
ylims!(ax, minimum(costs) / 2, 2 * maximum(costs))

axu = Axis(bottom[1, 1]; xlabel="u (m s⁻¹)", ylabel="z (m)", title="u")
axv = Axis(bottom[1, 2]; xlabel="v (m s⁻¹)", title="v")
axb = Axis(bottom[1, 3]; xlabel="b - N² z (m s⁻²)", title="Buoyancy anomaly", xticks=LinearTicks(3))

for (ax, name) in zip((axu, axv, axb), (:u, :v, :b))
    lines!(ax, getproperty(nature_state, name), z; linewidth=4, label="nature run")
    lines!(ax, getproperty(first(iteration_states), name), z; linewidth=2, linestyle=:dot, color=(:gray, 0.6), label="initial guess")
    profile = @lift getproperty(iteration_states[$n], name)
    lines!(ax, profile, z; linewidth=3, linestyle=:dash, label="estimate")

    all_profiles = [getproperty(state, name) for state in (nature_state, iteration_states...)]
    xmin = minimum(minimum, all_profiles)
    xmax = maximum(maximum, all_profiles)
    δx = (xmax - xmin) / 20
    xlims!(ax, xmin - δx, xmax + δx)
    ylims!(ax, -80, 0)
end

axislegend(axu, position=:lb)

current_figure() #hide

# Finally, we animate the progress of the optimization,

CairoMakie.record(fig, "single_column_parameter_estimation.mp4", 1:length(history), framerate=2) do i
    n[] = i
end
nothing #hide

# ![](single_column_parameter_estimation.mp4)

# ## Comparing BFGS with gradient descent
#
# Optim makes it easy to try other optimizers with the same compiled cost and gradient.
# We repeat the optimization with plain gradient descent, starting from the same initial guess
# and taking the same number of iterations,

gradient_descent_result = optimize(𝒥, ∇𝒥!, θ₀, GradientDescent(), options)

# The optimizers' line searches evaluate the cost and gradient a different number of times per
# iteration, so we also count evaluations, and measure how long each evaluation takes,

forward_time = @elapsed 𝒥(θ₀)
gradient_time = @elapsed cost_and_gradient(θ₀)

@info @sprintf("Evaluating the cost takes %.2f s and evaluating its gradient takes %.2f s", forward_time, gradient_time)

for (name, result) in (("BFGS", bfgs_result), ("Gradient descent", gradient_descent_result))
    θ = Optim.minimizer(result)
    @info @sprintf("%s: 𝒥 = %.2e after %d iterations with %d cost and %d gradient evaluations; θ / θ★ = %s",
                   name, Optim.minimum(result), Optim.iterations(result),
                   Optim.f_calls(result), Optim.g_calls(result), string(round.(θ ./ θ★, digits=3)))
end

fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1]; xlabel="Iteration", ylabel="𝒥", yscale=log10, title="Cost")

for (label, result) in (("BFGS", bfgs_result), ("gradient descent", gradient_descent_result))
    costs = Optim.f_trace(result)
    scatterlines!(ax, 0:length(costs)-1, costs; label)
end

axislegend(ax)

current_figure() #hide
