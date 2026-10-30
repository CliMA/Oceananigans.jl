# # [Online learning of an eddy closure for a wind-driven double gyre](@id online_learning_double_gyre_example)
#
# This example trains a neural network closure for a coarse resolution ocean model by differentiating
# *through* the model ("online learning"), following
# [Maddison (2026), "Online learning in idealized ocean gyres"](https://doi.org/10.1029/2024MS004883).
# A barotropic double gyre is simulated at 62.5 km resolution, too coarse to resolve the eddies that
# drive and shape the jet. A convolutional neural network forces the coarse model at every time step,
# and its weights are trained so that the coarse model's vorticity follows a coarse-grained
# high resolution "truth" over one-day windows. The truth was computed ahead of time at 3.9 km
# resolution with the same equations, WENO advection and no explicit closure, and is downloaded as an artifact.
#
# This example demonstrates:
#
#   * How to differentiate a `NonhydrostaticModel` in a closed basin, whose pressure solver uses discrete
#     cosine transforms built from FFTs, with Reactant.
#   * How to force a model with a neural network evaluated at every Runge-Kutta stage, via a callback.
#   * How to compute the gradient of a trajectory loss with respect to the network's weights
#     with Reactant and Enzyme.
#   * How to train with AdamW from Optimisers.jl, and compare free runs with and without the closure.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Enzyme, Reactant, CUDA, Optimisers, JLD2, CairoMakie"
# ```
#
# Most of this example's run time is spent compiling with Reactant. Starting Julia with `julia -O0` cuts the
# compile time by about a fifth without slowing down the compiled code, which XLA optimizes on its own.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState, on_architecture
using Oceananigans.Operators: ∂xᶜᶠᶜ, ∂yᶠᶜᶜ, ℑxyᶠᶠᵃ, ζ₃ᶠᶠᶜ
using Oceananigans.Grids: ynode
using Reactant
using Enzyme
using CUDA
using Optimisers
using JLD2
using CairoMakie
using Statistics
using Random
using Printf

# ## The coarse model
#
# Like Maddison's, our coarse model solves the barotropic vorticity equation: two-dimensional incompressible
# flow in a 4000 km square basin on a β-plane, forced by a double gyre wind stress acting on a 500 m deep
# layer and damped by linear bottom drag. A `NonhydrostaticModel` on a `(Bounded, Bounded, Flat)` grid solves
# exactly these equations, in their momentum form, with a rigid lid. Like the truth, the coarse model has no
# explicit viscosity: a ninth-order upwind-biased advection scheme removes grid-scale noise. On its own, the
# coarse model puts too much energy into its mean flow and too little into its eddies.

Lx = 4000kilometers
H = 500
Nx = 64
Δx = Lx / Nx
Δt = 30minutes
ρ₀ = 1000
τ₀ = 0.1
β = 2e-11

grid = RectilinearGrid(ReactantState(); size=(Nx, Nx), x=(0, Lx), y=(0, Lx), halo=(5, 5),
                       topology=(Bounded, Bounded, Flat))

# The network produces a scalar potential `Φ` on cell corners, which is zero on the walls.
# Its rotated gradient forces the momentum equations, so the network's forcing is divergence-free and
# produces no flow through the walls.

Φ = Field{Face, Face, Center}(grid)
parameters = (; τ₀, Lx, ρ₀, H, r=1e-7, Φ)

@inline function u_forcing_function(i, j, k, grid, clock, fields, p)
    y = ynode(j, grid, Center())
    wind = - p.τ₀ * cos(2π * y / p.Lx) / (p.ρ₀ * p.H)
    return wind - p.r * @inbounds(fields.u[i, j, k]) + ∂yᶠᶜᶜ(i, j, k, grid, p.Φ) / p.H
end

@inline v_forcing_function(i, j, k, grid, clock, fields, p) =
    - p.r * @inbounds(fields.v[i, j, k]) - ∂xᶜᶠᶜ(i, j, k, grid, p.Φ) / p.H

u_forcing = Forcing(u_forcing_function, discrete_form=true; parameters)
v_forcing = Forcing(v_forcing_function, discrete_form=true; parameters)

model = NonhydrostaticModel(grid;
                            advection = UpwindBiased(order=9),
                            coriolis = BetaPlane(f₀=7e-5, β=β),
                            forcing = (u=u_forcing, v=v_forcing),
                            timestepper = :RungeKutta3)

# ## The neural network
#
# A small convolutional network with five 3 × 3 convolutions and 16 hidden channels maps
# the velocities and the vorticity to the potential `Φ`. Like Maddison's ResNet,
# it also has a linear path from its inputs to its output, which can represent a viscosity directly
# (`Φ ∝ ζ` is a Laplacian viscosity) and so learns one in a few steps. We write the convolutions as
# nine shifted matrix multiplications, which Reactant compiles to a handful of fused kernels.

elu(x) = ifelse(x > 0, x, exp(x) - 1)

function convolve(x, W)
    N₁, N₂, Cin = size(x)
    Cout = size(W, 4)
    padded = similar(x, N₁ + 2, N₂ + 2, Cin)
    fill!(padded, 0)
    padded[2:N₁+1, 2:N₂+1, :] .= x
    output = reshape(padded[1:N₁, 1:N₂, :], N₁ * N₂, Cin) * W[1, 1, :, :]
    for a in 1:3, b in 1:3
        (a, b) == (1, 1) && continue
        output = output .+ reshape(padded[a:a+N₁-1, b:b+N₂-1, :], N₁ * N₂, Cin) * W[a, b, :, :]
    end
    return reshape(output, N₁, N₂, Cout)
end

bias(b) = reshape(b, 1, 1, length(b))

function network(θ, input)
    x = elu.(convolve(input, θ.W₁) .+ bias(θ.b₁))
    x = elu.(convolve(x, θ.W₂) .+ bias(θ.b₂))
    x = elu.(convolve(x, θ.W₃) .+ bias(θ.b₃))
    x = elu.(convolve(x, θ.W₄) .+ bias(θ.b₄))
    return convolve(x, θ.W₅) .+ convolve(input, θ.L)  # the second term is the linear path
end

glorot(rng, dims...) = sqrt(6 / (dims[1] * dims[2] * (dims[3] + dims[4]))) .* (2 .* rand(rng, dims...) .- 1)

# The last layer and the linear path start at zero, so that training starts from the model without a closure.

rng = Xoshiro(2026)
initial_weights = (W₁ = glorot(rng, 3, 3, 4, 16),  b₁ = zeros(16),
                   W₂ = glorot(rng, 3, 3, 16, 16), b₂ = zeros(16),
                   W₃ = glorot(rng, 3, 3, 16, 16), b₃ = zeros(16),
                   W₄ = glorot(rng, 3, 3, 16, 16), b₄ = zeros(16),
                   W₅ = zeros(3, 3, 16, 1),        L = zeros(3, 3, 4, 1))

@info "The network has $(sum(length, initial_weights)) weights"

# The inputs and output are scaled by the natural units of the gyre (Maddison's equation 6,
# translated to velocity and a transport potential).

L = Lx / 2
velocity_scale = β * L * Δx
potential_scale = τ₀ * π / (ρ₀ * H * L) * H * Δx^2

# The callback below evaluates the network on the current state and writes `Φ`. Registered at the
# `UpdateStateCallsite`, it runs after every Runge-Kutta stage.
#
# The network sees the velocities and the vorticity interpolated to cell centers. Its output is interpolated
# to the cell corners, where `Φ` lives, and set to zero on the walls.

@inline on_interior_corner(i, j, grid) = (i > 1) & (i <= grid.Nx) & (j > 1) & (j <= grid.Ny)

@inline interior_vorticity(i, j, k, grid, u, v) = ifelse(on_interior_corner(i, j, grid), ζ₃ᶠᶠᶜ(i, j, k, grid, u, v), zero(grid))

interior_vorticity(u, v) = Field(KernelFunctionOperation{Face, Face, Center}(interior_vorticity, u.grid, u, v))

@inline interior_corner_average(i, j, k, grid, c) =
    ifelse(on_interior_corner(i, j, grid), ℑxyᶠᶠᵃ(i, j, k, grid, c), zero(grid))

function update_closure!(model, θ)
    grid = model.grid
    u, v, w = model.velocities

    inputs = (Field(@at((Center, Center, Center), u) / velocity_scale),
              Field(@at((Center, Center, Center), v) / velocity_scale),
              Field(@at((Center, Center, Center), ∂x(v) - ∂y(u)) / (β * L)))
    wet = one.(interior(first(inputs), :, :, 1))
    x = cat((interior(input, :, :, 1) for input in inputs)..., wet; dims=3)
    y = set!(CenterField(grid), network(θ, x))
    Φ = KernelFunctionOperation{Face, Face, Center}(interior_corner_average, grid, y)
    set!(model.forcing.u.parameters.Φ, potential_scale * Φ)

    return nothing
end

# Each training window starts from a truth state. After `set!` (which also makes the state divergence-free),
# we update the closure and recompute the tendencies, so that the first time step uses this window's `Φ`.

using Oceananigans.TimeSteppers: update_state!

function start_window!(model, θ, u₀, v₀)
    set!(model, u=u₀, v=v₀)
    update_closure!(model, θ)
    update_state!(model)
    return [Callback(update_closure!; parameters=θ, callsite=Oceananigans.UpdateStateCallsite())]
end

# ## Truth data
#
# The truth is a 1024² (3.9 km) `NonhydrostaticModel` simulation of the same gyre with `WENO(order=9)` and no
# explicit closure, spun up for four years. Like Maddison, we coarse-grain it with a Gaussian filter of width
# 62.5 km. The filter acts on the streamfunction, so that the coarse velocities on the 64² grid are
# divergence-free with no flow through the walls. We download two months of hourly coarse velocities, one for
# training and one for validation, and statistics from twelve years of daily coarse velocities.

truth_directory = get(ENV, "DOUBLE_GYRE_TRUTH", joinpath(@__DIR__, "double_gyre_truth"))
artifact_url = "https://github.com/glwagner/OceananigansArtifacts.jl/raw/main/double_gyre_online_learning"

for filename in ("truth_hourly_month01.jld2", "truth_hourly_month02.jld2", "truth_statistics.jld2")
    path = joinpath(truth_directory, filename)
    isfile(path) || (mkpath(truth_directory); download(joinpath(artifact_url, filename), path))
end

training_truth, validation_truth = map(("truth_hourly_month01.jld2", "truth_hourly_month02.jld2")) do filename
    path = joinpath(truth_directory, filename)
    (u = FieldTimeSeries(path, "u"), v = FieldTimeSeries(path, "v"))
end

# The loss compares the model's vorticity with the truth's on the interior cell corners, computed with the
# same operator:

truth_vorticity(truth) = stack(1:length(truth.u.times)) do n
    interior(interior_vorticity(truth.u[n], truth.v[n]), :, :, 1)
end

training_vorticity = truth_vorticity(training_truth)
validation_vorticity = truth_vorticity(validation_truth)

# ## The loss and its gradient
#
# The loss is Maddison's equation 7: the mean squared vorticity error over the hourly outputs of a
# window, normalized by ``β L``. The window length is the number of hourly targets `ζ★` holds, which is
# fixed when the loss is compiled.

steps_per_hour = Int(1hour / Δt)

function window_loss(θ, model, u₀, v₀, ζ★, misfit)
    callbacks = start_window!(model, θ, u₀, v₀)
    u, v, w = model.velocities
    ζ = interior_vorticity(u, v)
    window_hours = size(ζ★, 3)
    fill!(misfit, 0)
    @trace track_numbers=false mincut=true checkpointing=Reactant.Periodic(1) for hour in 1:window_hours
        @trace track_numbers=false for step in 1:steps_per_hour
            time_step!(model, Δt; callbacks)
        end
        compute!(ζ)
        misfit .+= sum(abs2, interior(ζ, :, :, 1) .- ζ★[:, :, hour])
    end
    return 1e4 / (β * L)^2 * Δx^2 / (2L)^2 * sum(misfit) / window_hours
end

function loss_and_gradient!(∂θ, θ, model, ∂model, u₀, v₀, ζ★, misfit, ∂misfit)
    _, loss = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), window_loss, Enzyme.Active,
                              Enzyme.Duplicated(θ, ∂θ), Enzyme.Duplicated(model, ∂model),
                              Enzyme.Const(u₀), Enzyme.Const(v₀), Enzyme.Const(ζ★),
                              Enzyme.Duplicated(misfit, ∂misfit))
    return loss
end

# One-day windows start at the beginning of an hour; their initial states and targets are uploaded to the
# device as Reactant arrays.

window_hours = 24
to_device(a) = on_architecture(ReactantState(), a)
to_host(a) = on_architecture(CPU(), a)
zero_like(θ) = map(x -> to_device(zero(x)), θ)
scalar() = to_device(zeros(1))

function window(truth, vorticity, start)
    hours = start+1:start+window_hours
    ## A snapshot's interior is a view into the whole time series, so we copy it before uploading
    return (u₀ = to_device(copy(interior(truth.u[start]))),
            v₀ = to_device(copy(interior(truth.v[start]))),
            ζ★ = to_device(vorticity[:, :, hours]))
end

θ = map(to_device, initial_weights)
w = window(training_truth, training_vorticity, 1)
compile_start = time_ns()
compiled_gradient! = @compile raise=true raise_first=true sync=true loss_and_gradient!(
    zero_like(θ), θ, model, Enzyme.make_zero(model), w.u₀, w.v₀, w.ζ★, scalar(), scalar())
@info @sprintf("Compiled the loss gradient in %.1f s", (time_ns() - compile_start) / 1e9)

function loss_and_gradient(weights, truth, vorticity, start)
    w = window(truth, vorticity, start)
    ∂θ = zero_like(weights)
    loss = compiled_gradient!(∂θ, map(to_device, weights), model, Enzyme.make_zero(model),
                              w.u₀, w.v₀, w.ζ★, scalar(), scalar())
    return Float64(loss), map(to_host, ∂θ)
end

# ## Training
#
# We train with AdamW from Optimisers.jl (with Keras' default parameters except for a larger learning rate)
# on batches of four one-day windows drawn at random from the first month; the second month is held out to
# measure the loss on unseen windows. A training this short only starts the job: the paper trains for 300
# epochs over a curriculum of window lengths from 1 to 60 days, re-initializing the optimizer at each window
# length.

training_starts = 1:length(training_truth.u.times) - window_hours
validation_starts = 1:168:length(validation_truth.u.times) - window_hours
validation_loss(weights) = mean(first(loss_and_gradient(weights, validation_truth, validation_vorticity, start))
                                for start in validation_starts)

weights = deepcopy(initial_weights)
optimizer = Optimisers.setup(AdamW(eta=1e-2, beta=(0.9, 0.999), lambda=0.004, epsilon=1e-7), weights)
training_steps = 120
training_losses = Float64[]
validation_losses = [validation_loss(weights)]
training_start = time_ns()

for step in 1:training_steps
    results = [loss_and_gradient(weights, training_truth, training_vorticity, start)
               for start in rand(rng, training_starts, 4)]
    push!(training_losses, mean(first, results))
    gradient = map((gs...) -> mean(gs), (last(r) for r in results)...)
    Optimisers.update!(optimizer, weights, gradient)
end

push!(validation_losses, validation_loss(weights))
@info @sprintf("Validation loss %.3e → %.3e after %d steps, %.0f s", validation_losses..., training_steps,
               (time_ns() - training_start) / 1e9)

# ## Free runs with and without the closure
#
# We run the coarse model from the first truth state, with the trained network and without any network
# (zero weights). After the coarse model has adjusted for half a year, we average for a year and compare its
# statistics with the truth's. One compiled call starts the run and another advances the model one day.

function start_free_run!(θ, model, u₀, v₀)
    start_window!(model, θ, u₀, v₀)
    return nothing
end

function simulate_day!(θ, model)
    callbacks = [Callback(update_closure!; parameters=θ, callsite=Oceananigans.UpdateStateCallsite())]
    @trace track_numbers=false for step in 1:24steps_per_hour
        time_step!(model, Δt; callbacks)
    end
    return nothing
end

compile_start = time_ns()
compiled_start! = @compile raise=true raise_first=true sync=true start_free_run!(θ, model, w.u₀, w.v₀)
compiled_day! = @compile raise=true raise_first=true sync=true simulate_day!(θ, model)
@info @sprintf("Compiled the free run in %.1f s", (time_ns() - compile_start) / 1e9)

kinetic_energy(u, v) = (sum(u^2) / length(u) + sum(v^2) / length(v)) / 2

function free_run(weights; spinup_days=180, averaging_days=365)
    w = window(training_truth, training_vorticity, 1)
    θ = map(to_device, weights)
    compiled_start!(θ, model, w.u₀, w.v₀)

    cpu_grid = on_architecture(CPU(), grid)
    u, v = XFaceField(cpu_grid), YFaceField(cpu_grid)
    ū, v̄ = XFaceField(cpu_grid), YFaceField(cpu_grid)
    daily_kinetic_energy = Float64[]
    for day in 1:spinup_days + averaging_days
        compiled_day!(θ, model)
        set!(u, model.velocities.u)
        set!(v, model.velocities.v)
        push!(daily_kinetic_energy, kinetic_energy(u, v))
        if day > spinup_days
            set!(ū, ū + u / averaging_days)
            set!(v̄, v̄ + v / averaging_days)
        end
    end

    total = mean(daily_kinetic_energy[spinup_days+1:end])
    mean_flow = kinetic_energy(ū, v̄)
    return (; daily_kinetic_energy, kinetic_energy = total, mean_kinetic_energy = mean_flow,
              eddy_kinetic_energy = total - mean_flow, ζ = interior_vorticity(u, v))
end

no_closure = free_run(map(zero, weights))
learned = free_run(weights)
truth_statistics = load(joinpath(truth_directory, "truth_statistics.jld2"))

for (name, run) in (("no closure", no_closure), ("learned closure", learned))
    @info @sprintf("%-16s KE %.2e, mean KE %.2e, eddy KE %.2e m² s⁻²  (truth: %.2e, %.2e, %.2e)", name,
                   run.kinetic_energy, run.mean_kinetic_energy, run.eddy_kinetic_energy,
                   truth_statistics["kinetic_energy"], truth_statistics["mean_kinetic_energy"],
                   truth_statistics["eddy_kinetic_energy"])
end

# After this short training, the closure reduces the vorticity error on the held-out month by about 30%.
# The free runs are another matter. Without a closure, the coarse model has too much energy in its mean flow
# and too little in its eddies. Trained on one-day windows, the closure mostly learns to dissipate: it brings
# the mean flow closer to the truth's, but it also damps the eddies. Matching long-term statistics requires
# longer training on longer windows, as in Maddison's 300-epoch curriculum. In our experiments with ten months
# of data and a curriculum of 1-, 2- and 4-day windows (3,500 optimizer steps), the closure cut the error over
# 4-day windows by three quarters and, for three of six initial weights, reproduced the truth's kinetic energy
# and eddy kinetic energy within 20% in free runs of one to five years. The other networks were too energetic
# or unstable, like Maddison's at short window lengths.
#
# ## Visualization

fig = Figure(size=(1200, 800))

ax = Axis(fig[1, 1:2], xlabel="optimizer step", ylabel="loss", yscale=log10, title="Training")
lines!(ax, 1:training_steps, training_losses, label="training batches")
scatter!(ax, [0, training_steps], validation_losses, label="validation windows")
axislegend(ax)

ax = Axis(fig[1, 3:4], xlabel="day", ylabel="kinetic energy (m² s⁻²)", title="Free runs")
lines!(ax, no_closure.daily_kinetic_energy, label="no closure")
lines!(ax, learned.daily_kinetic_energy, label="learned closure")
hlines!(ax, truth_statistics["kinetic_energy"], color=:black, linestyle=:dash, label="truth (time mean)")
axislegend(ax, position=:lt)

ζₜ = interior_vorticity(validation_truth.u[end], validation_truth.v[end])

ζmax = 3e-5
for (n, (title, ζ)) in enumerate((("truth", ζₜ), ("no closure", no_closure.ζ), ("learned closure", learned.ζ)))
    panel = Axis(fig[2, n], aspect=1, title="vorticity, $title", xlabel="x (m)", ylabel="y (m)")
    heatmap!(panel, ζ, colormap=:balance, colorrange=(-ζmax, ζmax))
end

save("online_learning_double_gyre.png", fig)
nothing #hide

# ![](online_learning_double_gyre.png)
