# # Wind and convection-driven mixing in an ocean surface boundary layer
#
# This example simulates mixing by three-dimensional turbulence in an ocean surface
# boundary layer driven by atmospheric winds and convection. It demonstrates:
#
#   * how to use the `SeawaterBuoyancy` model for buoyancy with a linear equation of state;
#   * how to use a turbulence closure for large eddy simulation;
#   * how to use a function to impose a boundary condition;
#   * how to use user-defined `model.parameters` in a boundary condition function.
#
# In addition to `Oceananigans.jl` we need `Plots` for plotting, `Random` for
# generating random initial conditions, and `Printf` for printing progress messages.
# We also need `Oceananigans.OutputWriters` and `Oceananigans.Diagnostics` to access
# some nice features for writing output data to disk.

using Random, Printf, Plots
using Oceananigans, Oceananigans.OutputWriters, Oceananigans.Diagnostics, Oceananigans.Utils

# ## Model parameters
#
# Here we use an isotropic, cubic grid with `Nz` grid points and grid spacing
# `Δz = 1` meter. We specify fluxes of heat, momentum, and salinity via
#
#   1. A temperature flux `Qᵀ` at the top of the domain, which is related to heat flux
#       by `Qᵀ = Qʰ / (ρ₀ * cᴾ)`, where `Qʰ` is the heat flux, `ρ₀` is a reference density,
#       and `cᴾ` is the heat capacity of seawater. With a reference density
#       `ρ₀ = 1026 kg m⁻³`and heat capacity `cᴾ = 3991`, our chosen temperature flux of
#       `Qᵀ = 5 × 10⁻⁵ K m⁻¹ s⁻¹` corresponds to a heat flux of `Qʰ = 204.7 W m⁻²`, a
#       relatively powerful cooling rate.
#
#   2. A velocity flux `Qᵘ` at the top of the domain, which is related
#       to the `x` momentum flux `τˣ` via `τˣ = ρ₀ * Qᵘ`, where `ρ₀` is a reference density.
#       Our chosen value of `Qᵘ = -2 × 10⁻⁵ m² s⁻²` roughly corresponds to atmospheric winds
#       of `uᵃ = 2.9 m s⁻¹` in the positive `x`-direction, using the parameterization
#       `τ = 0.0025 * |uᵃ| * uᵃ`.
#
#   3. An evaporation rate `evaporation = 10⁻⁷ m s⁻¹`, or approximately 0.1 millimeter per
#       hour.
#
# Finally, we use an initial temperature gradient of `∂T/∂z = 0.005 K m⁻¹`,
# which implies an iniital buoyancy frequency `N² = α * g * ∂T/∂z = 9.8 × 10⁻⁶ s⁻²`
# with a thermal expansion coefficient `α = 2 × 10⁻⁴ K⁻¹` and gravitational acceleration
# `g = 9.81 s⁻²`. Note that, by default, the `SeawaterBuoyancy` model uses a gravitational
# acceleration `gᴱᵃʳᵗʰ = 9.80665 s⁻²`.

         Nz = 32       # Number of grid points in x, y, z
         Δz = 1.0      # Grid spacing in x, y, z (meters)
         Qᵀ = 5e-5     # Temperature flux at surface
         Qᵘ = -2e-5    # Velocity flux at surface
       ∂T∂z = 0.005    # Initial vertical temperature gradient
evaporation = 1e-7     # Mass-specific evaporation rate [m s⁻¹]
          f = 1e-4     # Coriolis parameter
          α = 2e-4     # Thermal expansion coefficient
          β = 8e-4     # Haline contraction coefficient
nothing # hide

# ## Boundary conditions
#
# Here we define `Flux` boundary conditions at the surface for `u`, `T`, and `S`,
# and a `Gradient` boundary condition on `T` that maintains a constant stratification
# at the bottom. Our flux boundary condition for salinity uses a function that calculates
# the salinity flux in terms of the evaporation rate.

grid = RegularCartesianGrid(size=(Nz, Nz, Nz), length=(Δz*Nz, Δz*Nz, Δz*Nz))

u_bcs = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᵘ))

T_bcs = TracerBoundaryConditions(grid,    top = BoundaryCondition(Flux, Qᵀ),
                                       bottom = BoundaryCondition(Gradient, ∂T∂z))

## Salinity flux: Qˢ = - E * S
@inline Qˢ(i, j, grid, time, iter, U, C, p) = @inbounds -p.evaporation * C.S[i, j, 1]

S_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qˢ))
nothing # hide

# ## Model instantiation
#
# We instantiate a horizontally-periodic `Model` on the CPU with on a `RegularCartesianGrid`,
# using a `FPlane` model for rotation (constant rotation rate), a linear equation
# of state for temperature and salinity, the Anisotropic Minimum Dissipation closure
# to model the effects of unresolved turbulence, and the previously defined boundary
# conditions for `u`, `T`, and `S`. We also pass the evaporation rate to the container
# model.parameters for use in the boundary condition function that calculates the salinity
# flux.

model = Model(
         architecture = CPU(),
                 grid = grid,
             coriolis = FPlane(f=f),
             buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=α, β=β)),
              closure = AnisotropicMinimumDissipation(),
  boundary_conditions = SolutionBoundaryConditions(grid, u=u_bcs, T=T_bcs, S=S_bcs),
           parameters = (evaporation = evaporation,)
)
nothing # hide

# Notes:
#
# * To use the Smagorinsky-Lilly turbulence closure (with a constant model coefficient) rather than
#   `AnisotropicMinimumDissipation`, use `closure = ConstantSmagorinsky()` in the model constructor.
#
# * To change the `architecture` to `GPU`, replace the `architecture` keyword argument with
#   `architecture = GPU()``

# ## Initial conditions
#
# Our initial condition for temperature consists of a linear stratification superposed with
# random noise damped at the walls, while our initial condition for velocity consists
# only of random noise.

## Random noise damped at top and bottom
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise

## Temperature initial condition: a stable density tradient with random noise superposed.
T₀(x, y, z) = 20 + ∂T∂z * z + ∂T∂z * model.grid.Lz * 1e-6 * Ξ(z)

## Velocity initial condition: random noise scaled by the friction velocity.
u₀(x, y, z) = sqrt(abs(Qᵘ)) * 1e-1 * Ξ(z)

set!(model, u=u₀, w=u₀, T=T₀, S=35)

# ## Set up output
#
# We set up an output writer that saves all velocity fields, tracer fields, and the subgrid
# turbulent diffusivity associated with `model.closure`. The `prefix` keyword argument
# to `JLD2OutputWriter` indicates that output will be saved in
# `ocean_wind_mixing_and_convection.jld2`.

## Create a NamedTuple containing all the fields to be outputted.
fields_to_output = merge(model.velocities, model.tracers, (νₑ=model.diffusivities.νₑ,))

## Instantiate a JLD2OutputWriter to write fields. We will add it to the simulation before
## running it.
field_writer = JLD2OutputWriter(model, FieldOutputs(fields_to_output); interval=hour/4,
                                prefix="ocean_wind_mixing_and_convection", force=true)

# ## Running the simulation
#
# To run the simulation, we instantiate a `TimeStepWizard` to ensure stable time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 0.2.

wizard = TimeStepWizard(cfl=0.2, Δt=1.0, max_change=1.1, max_Δt=5.0)
nothing # hide

# A diagnostic that returns the maximum absolute value of `w` by calling
# `wmax(model)`:

wmax = FieldMaximum(abs, model.velocities.w)
nothing # hide

# Finally, we set up and run the the simulation.

simulation = Simulation(model, Δt=wizard, stop_iteration=0, progress_frequency=10)
simulation.output_writers[:fields] = field_writer

anim = @animate for i in 1:100
    ## Run the simulation forward
    simulation.stop_iteration += 10
    walltime = @elapsed run!(simulation)

    ## Print a progress message
    @printf("i: %04d, t: %s, Δt: %s, wmax = %.1e ms⁻¹, wall time: %s\n",
            model.clock.iteration, prettytime(model.clock.time), prettytime(wizard.Δt),
            wmax(model), prettytime(walltime))

    ## Coordinate arrays for plotting
    xC, zF, zC = model.grid.zC, model.grid.zF[1:Nz], model.grid.zC

    ## Slices to plots.
    jhalf = floor(Int, model.grid.Ny/2)
    w = Array(interior(model.velocities.w))[:, jhalf, :]
    T = Array(interior(model.tracers.T))[:, jhalf, :]
    S = Array(interior(model.tracers.S))[:, jhalf, :]

    ## Plot the slices.
    w_plot = heatmap(xC, zF, w', xlabel="x (m)", ylabel="z (m)", color=:balance, clims=(-3e-2, 3e-2))
    T_plot = heatmap(xC, zC, T', xlabel="x (m)", ylabel="z (m)", color=:thermal, clims=(19.75, 20))
    S_plot = heatmap(xC, zC, S', xlabel="x (m)", ylabel="z (m)", color=:haline, clims=(34.99, 35.01))

    ## Arrange the plots side-by-side.
    plot(w_plot, T_plot, S_plot, layout=(1, 3), size=(1600, 400),
         title=["vertical velocity (m/s)" "temperature (C)" "salinity (g/kg)"])
end

mp4(anim, "ocean_wind_mixing_and_convection.mp4", fps = 15) # hide
