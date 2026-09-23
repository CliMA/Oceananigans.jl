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
using LinearAlgebra: dot

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

# The function below assigns normalized parameters `x` to the model. It builds a new closure with
# the `VariableStabilityFunctions` constructor, which also computes another constant, `𝕊u₀`, from `Cu₀`.
# Because we call the constructor inside the function we differentiate, the gradient accounts
# for the dependence of `𝕊u₀` on `Cu₀`.

function assign_parameters!(model, x)
    Cu₀ = scales.Cu₀ * x.Cu₀
    Cc₀ = scales.Cc₀ * x.Cc₀
    FT = typeof(Cu₀)

    stability_functions = VariableStabilityFunctions(FT; Cu₀, Cc₀)
    model.closure = TKEDissipationVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; stability_functions)

    τˣ = model.velocities.u.boundary_conditions.top.condition
    Jᵇ = model.tracers.b.boundary_conditions.top.condition
    parent(τˣ) .= scales.τˣ * x.τˣ
    parent(Jᵇ) .= scales.Jᵇ * x.Jᵇ

    return nothing
end

# We represent parameters with a `NamedTuple` of Reactant numbers,

parameters(values) = NamedTuple{keys(scales)}(Tuple(Reactant.ConcreteRNumber.(values)))

# ## Running the column
#
# Each run starts from rest with constant stratification and lasts 12 hours. Because we reuse the same
# model for every run, we first zero its velocities, tracers (including `e` and `ϵ`), and the
# tendencies and velocities saved from previous time steps, so that every run starts from the same state.
# The k-ϵ equations need a fairly short time step: with `Δt = 10minutes`, for example, the solution
# becomes noisy.

N² = 1e-5
bᵢ = set!(CenterField(grid), z -> N² * z)

Δt = 1minute
Nt = 720

function reset_column!(model)
    u, v = model.velocities
    u⁻, v⁻ = model.closure_fields.previous_velocities
    fields = (u, v, u⁻, v⁻, model.tracers..., model.timestepper.Gⁿ..., model.timestepper.G⁻...)

    for field in fields
        fill!(field, 0)
    end

    return nothing
end

function run_column!(model, x, bᵢ, Δt, Nt)
    assign_parameters!(model, x)
    reset_column!(model)
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

x★ = [1.067, 1.120, -1, 2]

compiled_run_column! = @compile raise=true raise_first=true sync=true run_column!(model, parameters(x★), bᵢ, Δt, Nt)
compiled_run_column!(model, parameters(x★), bᵢ, Δt, Nt)

# We save the final state of the nature run as our "observations",

function observe(model)
    u★ = XFaceField(grid)
    v★ = YFaceField(grid)
    b★ = CenterField(grid)
    interior(u★) .= interior(model.velocities.u)
    interior(v★) .= interior(model.velocities.v)
    interior(b★) .= interior(model.tracers.b)
    return (; u★, v★, b★)
end

observations = observe(model)

# ## Cost function and its gradient
#
# The cost function is the normalized mean square difference between
# the final state of the model and the observations,

function cost(x, model, bᵢ, observations, Δt, Nt)
    run_column!(model, x, bᵢ, Δt, Nt)

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

# To compute the gradient, we differentiate `cost` in reverse mode. Enzyme accumulates the
# gradient of the cost with respect to the parameters into the "shadow" parameters `∂J∂x`.
# The model is also mutated by `cost`, so we give it a shadow as well.

function cost_and_gradient!(∂J∂x, x, model, shadow, bᵢ, observations, Δt, Nt)
    mode = Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal)
    _, J = Enzyme.autodiff(mode, cost, Enzyme.Active,
                           Enzyme.Duplicated(x, ∂J∂x),
                           Enzyme.Duplicated(model, shadow),
                           Enzyme.Const(bᵢ),
                           Enzyme.Const(observations),
                           Enzyme.Const(Δt),
                           Enzyme.Const(Nt))
    return J
end

# We start from an initial guess that's quite different from the nature run,

x₀ = [0.6, 1.8, -0.5, 0.5]

# and compile the cost and its gradient. Compiling the gradient takes a few minutes.

shadow = Enzyme.make_zero(model)
∂J∂x = parameters(zeros(4))

compiled_cost = @compile raise=true raise_first=true sync=true cost(parameters(x₀), model, bᵢ, observations, Δt, Nt)
compiled_cost_and_gradient! = @compile raise=true raise_first=true sync=true cost_and_gradient!(∂J∂x, parameters(x₀), model, shadow, bᵢ, observations, Δt, Nt)

J(x) = Float64(compiled_cost(parameters(x), model, bᵢ, observations, Δt, Nt))

function cost_and_gradient(x)
    shadow = Enzyme.make_zero(model)
    ∂J∂x = parameters(zeros(4))
    Jx = compiled_cost_and_gradient!(∂J∂x, parameters(x), model, shadow, bᵢ, observations, Δt, Nt)
    return Float64(Jx), Float64.(collect(∂J∂x))
end

# ## Minimizing the cost
#
# We minimize the cost with [BFGS](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm),
# a quasi-Newton flavor of gradient descent. Plain gradient descent steps along `-∇J`. BFGS instead
# steps along `-H ∇J`, where `H` is an estimate of the inverse Hessian of `J` that BFGS builds from the
# change in the gradient between iterations. We start with `H = α I`, which makes the first iteration
# a plain gradient descent step. Each step is shortened by a backtracking line search until the cost
# decreases sufficiently.

identity_matrix(N) = [i == j ? 1.0 : 0.0 for i in 1:N, j in 1:N]

function bfgs(x₀; iterations=25, α=1)
    x = copy(x₀)
    Jⁿ, ∇J = cost_and_gradient(x)
    H = α * identity_matrix(length(x))
    history = [(x=copy(x), J=Jⁿ)]

    for n = 1:iterations
        d = - H * ∇J

        t = 1.0
        x′ = x .+ t .* d
        J′ = J(x′)
        while J′ > Jⁿ + 1e-4 * t * dot(∇J, d)
            t /= 2
            x′ = x .+ t .* d
            J′ = J(x′)
        end

        J′, ∇J′ = cost_and_gradient(x′)

        ## BFGS update of the inverse Hessian estimate
        s = x′ .- x
        y = ∇J′ .- ∇J
        if dot(s, y) > 0
            ρ = 1 / dot(s, y)
            A = identity_matrix(length(x)) .- ρ .* s * y'
            H = A * H * A' .+ ρ .* s * s'
        end

        x, Jⁿ, ∇J = x′, J′, ∇J′
        push!(history, (x=copy(x), J=Jⁿ))
        @info @sprintf("iteration %2d: J = %.2e, x / x★ = %s", n, Jⁿ, string(round.(x ./ x★, digits=3)))
    end

    return history
end

history = bfgs(x₀)

# ## Visualizing the parameter estimation
#
# We compute the final state of the model at each iteration,

z = Array(znodes(grid, Center()))

function final_state(x)
    compiled_run_column!(model, parameters(x), bᵢ, Δt, Nt)
    u = Array(interior(model.velocities.u))[:]
    v = Array(interior(model.velocities.v))[:]
    b = Array(interior(model.tracers.b))[:] .- N² .* z
    return (; u, v, b)
end

nature_state = final_state(x★)
iteration_states = [final_state(h.x) for h in history]

# and plot the parameters, the cost, and the profiles of velocity and the buoyancy
# anomaly `b - N² z`. We plot the state of the optimization at iteration `n`, starting
# with the final iteration.

fig = Figure(size=(1200, 800))
top = fig[2, 1] = GridLayout()
bottom = fig[3, 1] = GridLayout()

n = Observable(length(history))

title = @lift @sprintf("Iteration %d, J = %.1e", $n - 1, history[$n].J)
Label(fig[1, 1], title, fontsize=20, tellwidth=false)

iterations = 0:length(history)-1
labels = ["Cu₀", "Cc₀", "τˣ", "Jᵇ"]

ax = Axis(top[1, 1]; xlabel="Iteration", ylabel="Parameter / nature run value", title="Parameters")
hlines!(ax, 1; color=:gray, linestyle=:dash)
for i in 1:4
    ratio = [h.x[i] / x★[i] for h in history]
    points = @lift Point2f.(iterations[1:$n], ratio[1:$n])
    scatterlines!(ax, points; label=labels[i])
end
xlims!(ax, -0.5, length(history) - 0.5)
ylims!(ax, 0, 2)
axislegend(ax, position=:rt)

costs = [h.J for h in history]
ax = Axis(top[1, 2]; xlabel="Iteration", ylabel="J", yscale=log10, title="Cost")
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
