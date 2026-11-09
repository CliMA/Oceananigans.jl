# # [Single column parameter estimation with Reactant and Enzyme](@id single_column_parameter_estimation_example)
#
# We estimate turbulence closure parameters and surface fluxes in a "twin experiment":
# generate observations with known parameters, then fit a model to those observations.
# [Enzyme](https://github.com/EnzymeAD/Enzyme.jl) computes gradients through the simulation,
# compiled by [Reactant](https://github.com/EnzymeAD/Reactant.jl).
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans Enzyme Reactant CUDA Optim CairoMakie"
# ```
#
# Reactant needs CUDA.jl to be loaded to compile Oceananigans' kernels,
# even when it compiles for the CPU.
#
# To use the IFRT runtime, as Oceananigans' Reactant tests do, add the following to
# your project's `LocalPreferences.toml` and restart Julia:
#
# ```toml
# [Reactant]
# xla_runtime = "IFRT"
# ```

using Oceananigans
using Oceananigans.Units
using Enzyme
using Reactant
using CUDA
using CairoMakie
using Printf

# A single column is small, so we run Reactant on the CPU even when a GPU is available.

Reactant.set_default_backend("cpu")

# ## A single column model
#
# Surface wind stress `τˣ` and cooling buoyancy flux `Jᵇ` mix an ocean column.
# The k-ϵ closure computes viscosity and diffusivity from turbulent kinetic energy `e`
# and its dissipation rate `ϵ`,
#
# ```math
# κu = 𝕊u \frac{e²}{ϵ} \quad \text{and} \quad κc = 𝕊c \frac{e²}{ϵ} \, ,
# ```
#
# The stability functions `𝕊u` and `𝕊c` depend on shear and stratification. We estimate
# their magnitude parameters `Cu₀` and `Cc₀`, along with the surface fluxes `τˣ` and `Jᵇ`.
#
# To change parameters inside a compiled function, we store closure parameters as
# Reactant numbers and surface fluxes as mutable `Field`s.

using Oceananigans.Architectures: ReactantState

grid = RectilinearGrid(ReactantState(), size=32, z=(-128, 0), topology=(Flat, Flat, Bounded))
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

# We rebuild the closure inside the differentiated function because `VariableStabilityFunctions`
# derives another constant, `𝕊u₀`, from `Cu₀`. This includes that dependence in the gradient.

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: VariableStabilityFunctions

function assign_parameters!(model, normalized_parameters)
    Cu₀ = scales.Cu₀ * normalized_parameters.Cu₀
    Cc₀ = scales.Cc₀ * normalized_parameters.Cc₀
    FT = typeof(Cu₀)

    stability_functions = VariableStabilityFunctions(FT; Cu₀, Cc₀)
    model.closure = TKEDissipationVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; stability_functions)

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
# Each run starts from rest with constant stratification and lasts 12 hours. We reset the
# fields, tendencies, closure state, and clock before each run. Reactant compiles `run!`
# and its time-stepping loop together. We use a one-minute step; ten-minute steps give
# noisy solutions with this closure.

using Oceananigans.Models: reset!

N² = 1e-5
bᵢ = set!(CenterField(grid), z -> N² * z)

simulation = Simulation(model; Δt=1minute, stop_time=12hours, verbose=false)

function run_column!(simulation, normalized_parameters, bᵢ)
    model = simulation.model
    assign_parameters!(model, normalized_parameters)

    reset!(model)
    reset!(model.clock)
    set!(model, b=bᵢ)
    run!(simulation)

    return nothing
end

# ## The nature run
#
# The nature run uses the default closure parameters `Cu₀ = 0.1067` and `Cc₀ = 0.1120`,
# a wind stress `τˣ = -10⁻⁴ m² s⁻²`, and a cooling buoyancy flux `Jᵇ = 2 × 10⁻⁸ m² s⁻³`.
# The vector `θ★` contains these values divided by `scales`, in the same order.

θ★ = [1.067, 1.120, -1, 2]

compiled_run_column! = @compile raise=true raise_first=true sync=true run_column!(
    simulation, reactant_parameters(θ★), bᵢ)

compiled_run_column!(simulation, reactant_parameters(θ★), bᵢ)

# We save the final state of the nature run as our "observations",

observations = deepcopy((u=model.velocities.u, v=model.velocities.v, b=model.tracers.b))

# ## Cost function and its gradient
#
# The cost function is the normalized mean square difference between
# the final state of the model and the observations,

using Statistics: mean

function cost(normalized_parameters, simulation, bᵢ, observations)
    run_column!(simulation, normalized_parameters, bᵢ)

    model = simulation.model
    u, v = model.velocities
    b = model.tracers.b
    u★, v★, b★ = observations

    U² = 1e-2
    B² = (N² * 10)^2

    𝒥u = mean((u - u★)^2) / U²
    𝒥v = mean((v - v★)^2) / U²
    𝒥b = mean((b - b★)^2) / B²

    return 𝒥u + 𝒥v + 𝒥b
end

# `Enzyme.gradient` manages derivative buffers for the parameters and the mutated simulation.
# `ReverseWithPrimal` returns the cost too. We keep the parameter gradient and treat the
# initial condition and observations as constants.

function cost_and_gradient(parameters, simulation, bᵢ, observations)
    mode = Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal)
    result = Enzyme.gradient(mode, cost, parameters, simulation, Enzyme.Const(bᵢ), Enzyme.Const(observations))
    return result.val, result.derivs[1]
end

# We start from a deliberately inaccurate guess,

θ₀ = [0.6, 1.8, -0.5, 0.5]

# and compile the cost and its gradient. Compiling the gradient takes a few minutes.

compiled_cost = @compile raise=true raise_first=true sync=true cost(
    reactant_parameters(θ₀), simulation, bᵢ, observations)

compiled_cost_and_gradient = @compile raise=true raise_first=true sync=true cost_and_gradient(
    reactant_parameters(θ₀), simulation, bᵢ, observations)

# ## Minimizing the cost
#
# We use [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)'s
# [BFGS](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm) optimizer.
# A combined cost-and-gradient function reuses the cost from differentiation;
# a cost-only request uses the forward calculation.

using Optim
using Optim.NLSolversBase: only_fg!

function cost_and_gradient!(F, G, θ)
    parameters = reactant_parameters(θ)
    if isnothing(G)
        return Float64(compiled_cost(parameters, simulation, bᵢ, observations))
    end
    value, gradient = compiled_cost_and_gradient(parameters, simulation, bᵢ, observations)
    G .= Float64.(collect(gradient))
    return isnothing(F) ? nothing : Float64(value)
end

options = Optim.Options(iterations=10, store_trace=true, extended_trace=true)
bfgs_result = optimize(only_fg!(cost_and_gradient!), θ₀, BFGS(), options)

# The trace records the parameters and cost at each iteration,

history = [(θ=θ, 𝒥=𝒥θ) for (θ, 𝒥θ) in zip(Optim.x_trace(bfgs_result), Optim.f_trace(bfgs_result))]

# ## Visualizing the parameter estimation
#
# We compute the final state of the model at each iteration,

using Oceananigans.Architectures: on_architecture

z = on_architecture(CPU(), znodes(grid, Center()))

function final_state(θ)
    compiled_run_column!(simulation, reactant_parameters(θ), bᵢ)
    u = vec(on_architecture(CPU(), interior(model.velocities.u)))
    v = vec(on_architecture(CPU(), interior(model.velocities.v)))
    b = vec(on_architecture(CPU(), interior(model.tracers.b))) .- N² .* z
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

ax = Axis(top[1, 1]; xlabel="Iteration", ylabel="Parameter / nature run value", title="Parameters")
hlines!(ax, 1; color=:gray, linestyle=:dash)
for (i, name) in enumerate(keys(scales))
    ratio = [h.θ[i] / θ★[i] for h in history]
    points = @lift Point2f.(iterations[1:$n], ratio[1:$n])
    scatterlines!(ax, points; label=string(name))
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
