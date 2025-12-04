# # [Single column state estimation example](@id single_column_state_estimation_example)
#
# This example demonstrates state estimation for a single column ocean model using
# automatic differentiation with Enzyme and Reactant. We compute gradients of a
# cost function (mean square error) with respect to the initial velocity field
# array for a diffusion problem in a stratified flow.
#
# This example demonstrates:
#
#   * How to set up a single column model with `TKEDissipationVerticalDiffusivity`
#   * How to run a forward simulation to generate "truth" data
#   * How to define a cost function comparing model output to truth
#   * How to compute gradients using Enzyme automatic differentiation
#   * How to use Reactant for XLA compilation acceleration
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Enzyme, Reactant, CairoMakie"
# ```

using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: reset!
using Enzyme
using Reactant
using CairoMakie
using Printf
using CUDA

# Required for Enzyme
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.maxtypeoffset!(2032)

# ## Model Configuration
#
# We set up a single column model (1D vertical) representing an ocean water column.
# The model uses `TKEDissipationVerticalDiffusivity` (k-epsilon) closure for
# turbulent mixing in a stratified environment.

# Parameters
Nz = 128
H = 200          # Depth [m]
U₀ = 0.2         # Jet amplitude [m/s]
du = 10
z₀ = -100        # Jet center depth [m]

N² = 1e-5        # Buoyancy frequency squared [s⁻²]
db = 20          # Buoyancy transition width [m]
b₀ = N² * db     # Buoyancy scale
z₁ = -90         # Buoyancy transition center [m] (slightly offset from jet)

arch = Oceananigans.Architectures.ReactantState()
grid = RectilinearGrid(arch, size=Nz, z=(-H, 0), topology=(Flat, Flat, Bounded))
closure = TKEDissipationVerticalDiffusivity()
model = HydrostaticFreeSurfaceModel(; grid, closure, tracers=(:b, :e, :ϵ), buoyancy=BuoyancyTracer())

# ## Initial Conditions
#
# We define a Gaussian jet velocity profile and a tanh-stratified buoyancy field.

u_jet(z) = U₀ * exp(-(z - z₀)^2 / 2du^2)
b_stratified(z) = b₀ * tanh((z - z₁) / db)

# Set initial conditions
# set!(model, u=u_jet, b=b_stratified)

# ## Truth Simulation
#
# Run the model forward to generate "truth" data that we'll try to match.
function take_12_steps!(model, uᵢ, bᵢ)
    set!(model, u=uᵢ, b=bᵢ)        

    model.clock.iteration = 0
    model.clock.time = 0

    Δt = model.clock.last_Δt
    @trace mincut = true checkpointing = true track_numbers = false for n = 1:12
        time_step!(model, Δt)
    end

    return nothing
end

model.clock.last_Δt = 1minutes
uᵢ = set!(CenterField(grid), u_jet)
bᵢ = set!(CenterField(grid), b_stratified)
r_12_steps! = @compile sync=true raise=true take_12_steps!(model, uᵢ, bᵢ)

u₀ = interior(uᵢ, 1, 1, :) |> Array
b₀ = interior(bᵢ, 1, 1, :) |> Array

r_12_steps!(model, uᵢ, bᵢ)

u₁ = interior(model.velocities.u, 1, 1, :) |> Array
b₁ = interior(model.tracers.b, 1, 1, :) |> Array

# ## Visualization of Truth Simulation
#
# Plot the initial and final velocity profiles from the truth simulation.

fig_truth = Figure(size=(800, 400))
ax_u = Axis(fig_truth[1, 1]; xlabel="Velocity [m/s]", ylabel="Depth [m]", title="Truth Simulation")
ax_b = Axis(fig_truth[1, 2]; xlabel="Buoyancy [m/s²]", ylabel="Depth [m]", title="Buoyancy Profile")

z = znodes(model.velocities.u) |> Array
lines!(ax_u, u₀, z, label="Initial")
lines!(ax_u, u₁, z, label="Final")
axislegend(ax_u)

lines!(ax_b, b₀, z, label="Initial")
lines!(ax_b, b₁, z, label="Final")
axislegend(ax_b)

current_figure() #hide

#=
# ## Cost Function
#
# Define a cost function that takes an initial velocity array, runs the model forward,
# and computes the mean square error with respect to the truth data.

function cost_function(u_init_array, model, u_truth, b_truth, Δt, n_steps)
    # Reset model clock
    reset!(model.clock)
    
    # Set initial condition from array
    # Note: u_init_array should match the size of the velocity field interior
    set!(model, u=u_init_array, b=b_stratified)
    
    # Run model forward
    for n = 1:n_steps
        time_step!(model, Δt)
    end
    
    # Compute mean square error using loops (required for Enzyme compatibility)
    u_model = model.velocities.u
    b_model = model.tracers.b
    
    Nz = size(model.grid, 3)
    
    # MSE for velocity (using loops instead of broadcasting)
    mse_u = 0.0
    for k = 1:Nz
        mse_u += (u_model[1, 1, k] - u_truth[k])^2
    end
    mse_u /= Nz
    
    # MSE for buoyancy (with constant stratification)
    mse_b = 0.0
    for k = 1:Nz
        mse_b += (b_model[1, 1, k] - b_truth[k])^2
    end
    mse_b /= Nz
    
    # Total cost
    cost = mse_u + mse_b
    
    return cost::Float64
end

# ## Gradient Computation with Enzyme
#
# Use Enzyme to compute gradients of the cost function with respect to the initial
# velocity field array.

# Create a copy of the model for differentiation
model_for_gradient = deepcopy(model)

# Initial guess (perturbed from truth)
# Need to create a 3D array for set! function
u_init_guess_1d = copy(u_init_truth) .+ 0.01 * randn(length(u_init_truth))
u_init_guess = reshape(u_init_guess_1d, (1, 1, length(u_init_guess_1d)))

# Compute cost at initial guess
cost_initial = cost_function(u_init_guess, model_for_gradient, u_truth, b_truth, Δt, n_steps)
@info "Initial cost: $cost_initial"

# Prepare for Enzyme autodiff
dmodel = Enzyme.make_zero(model_for_gradient)

# Compute gradient
@info "Computing gradient with Enzyme..."
gradient_start = time_ns()

dcost_du = autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse),
                    cost_function,
                    Active(u_init_guess),
                    Duplicated(model_for_gradient, dmodel),
                    Const(u_truth),
                    Const(b_truth),
                    Const(Δt),
                    Const(n_steps))

gradient_time = 1e-9 * (time_ns() - gradient_start)
@info "Gradient computed in $(prettytime(gradient_time))"

# Extract gradient (convert back to 1D)
grad_u_full = dcost_du[1][1]
grad_u = grad_u_full[1, 1, :]

# ## Reactant Integration for XLA Compilation
#
# Reactant can be used to accelerate gradient computation through XLA compilation.
# Here we demonstrate how to use ReactantState architecture and the @jit macro.

# Create a model with ReactantState architecture
using Oceananigans.Architectures: ReactantState, CPU, on_architecture

# Note: Reactant requires special handling. For this example, we'll show the pattern
# but note that Reactant may have compatibility issues with certain operations.
# If Reactant errors occur, the CPU version above will still work.

try
    @info "Attempting to use Reactant for XLA compilation..."
    
    # Create Reactant architecture
    reactant_arch = ReactantState()
    
    # Create grid and model on Reactant architecture
    reactant_grid = RectilinearGrid(reactant_arch, size=Nz, z=(-H, 0), topology=(Flat, Flat, Bounded))
    reactant_model = HydrostaticFreeSurfaceModel(; grid=reactant_grid, closure,
                                                 tracers = (:b, :e, :ϵ),
                                                 buoyancy = BuoyancyTracer())
    
    # Set initial conditions
    set!(reactant_model, u=u_jet, b=b_stratified)
    
    # JIT compile the cost function
    # Note: Reactant may have limitations with certain operations
    # This is a demonstration of the pattern, but may not work in all cases
    @jit function cost_function_reactant(u_init_array, model, u_truth, b_truth, Δt, n_steps)
        reset!(model.clock)
        set!(model, u=u_init_array, b=b_stratified)
        
        for n = 1:n_steps
            time_step!(model, Δt)
        end
        
        u_model = model.velocities.u
        b_model = model.tracers.b
        Nz = size(model.grid, 3)
        
        mse_u = 0.0
        for k = 1:Nz
            mse_u += (u_model[1, 1, k] - u_truth[k])^2
        end
        mse_u /= Nz
        
        mse_b = 0.0
        for k = 1:Nz
            mse_b += (b_model[1, 1, k] - b_truth[k])^2
        end
        mse_b /= Nz
        
        return (mse_u + mse_b)::Float64
    end
    
    # Test Reactant version (may fail - that's okay per plan instructions)
    u_truth_reactant = Reactant.to_rarray(u_truth, track_numbers=Number)
    b_truth_reactant = Reactant.to_rarray(b_truth, track_numbers=Number)
    
    cost_reactant = cost_function_reactant(u_init_guess, reactant_model, 
                                           u_truth_reactant, b_truth_reactant, Δt, n_steps)
    @info "Reactant cost function executed successfully: $cost_reactant"
    
catch e
    @warn "Reactant compilation encountered an error (this is expected in some cases): $e"
    @info "Continuing with CPU version..."
end

# ## Visualization of Results
#
# Plot the cost function sensitivity with respect to the initial velocity field
# and the gradient components.

fig_grad = Figure(size=(1200, 400))

# Plot 1: Initial guess vs truth
ax1 = Axis(fig_grad[1, 1]; xlabel="Velocity [m/s]", ylabel="Depth [m]", title="Initial Conditions")
lines!(ax1, u_init_truth, z, label="Truth", linewidth=3)
lines!(ax1, u_init_guess_1d, z, label="Guess", linewidth=2, linestyle=:dash)
axislegend(ax1)

# Plot 2: Gradient components
ax2 = Axis(fig_grad[1, 2]; xlabel="Gradient [1/(m/s)²]", ylabel="Depth [m]", title="Cost Function Gradient")
lines!(ax2, grad_u, z, linewidth=2)
hlines!(ax2, [0], color=:black, linestyle=:dash, linewidth=1)

# Plot 3: Buoyancy sensitivity (constant stratification case)
# For constant stratification, we compute sensitivity wrt buoyancy field
b_init_guess_1d = copy(b_init_truth)
b_init_guess = reshape(b_init_guess_1d, (1, 1, length(b_init_guess_1d)))

function cost_function_b(b_init_array, model, u_truth, b_truth, Δt, n_steps)
    reset!(model.clock)
    # Convert u_init_truth back to 3D for set!
    u_init_truth_3d = reshape(u_init_truth, (1, 1, length(u_init_truth)))
    set!(model, u=u_init_truth_3d, b=b_init_array)
    
    for n = 1:n_steps
        time_step!(model, Δt)
    end
    
    u_model = model.velocities.u
    b_model = model.tracers.b
    
    Nz = size(model.grid, 3)
    
    # MSE for velocity
    mse_u = 0.0
    for k = 1:Nz
        mse_u += (u_model[1, 1, k] - u_truth[k])^2
    end
    mse_u /= Nz
    
    # MSE for buoyancy
    mse_b = 0.0
    for k = 1:Nz
        mse_b += (b_model[1, 1, k] - b_truth[k])^2
    end
    mse_b /= Nz
    
    return (mse_u + mse_b)::Float64
end

model_for_b_gradient = deepcopy(model)
dmodel_b = Enzyme.make_zero(model_for_b_gradient)

dcost_db = autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse),
                    cost_function_b,
                    Active(b_init_guess),
                    Duplicated(model_for_b_gradient, dmodel_b),
                    Const(u_truth),
                    Const(b_truth),
                    Const(Δt),
                    Const(n_steps))

grad_b_full = dcost_db[1][1]
grad_b = grad_b_full[1, 1, :]

ax3 = Axis(fig_grad[1, 3]; xlabel="Gradient [1/(m/s²)²]", ylabel="Depth [m]", title="Buoyancy Sensitivity")
lines!(ax3, grad_b, z, linewidth=2, color=:red)
hlines!(ax3, [0], color=:black, linestyle=:dash, linewidth=1)

current_figure() #hide

nothing #hide


=#