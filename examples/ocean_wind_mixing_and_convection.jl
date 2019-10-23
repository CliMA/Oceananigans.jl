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
# In addition to `Oceananigans.jl` we need `PyPlot` for plotting, `Random` for
# generating random initial conditions, and `Printf` for printing progress messages.

using Oceananigans, PyPlot, Random, Printf

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

         Nz = 48       # Number of grid points in x, y, z
         Δz = 1.0      # Grid spacing in x, y, z (meters)
         Qᵀ = 5e-5     # Temperature flux at surface
         Qᵘ = -2e-5    # Velocity flux at surface
       ∂T∂z = 0.005    # Initial vertical temperature gradient
evaporation = 1e-7     # Mass-specific evaporation rate [m s⁻¹]
   end_time = 2hour    # End time for the simulation
          f = 1e-4     # Coriolis parameter
          α = 2e-4     # Thermal expansion coefficient
          β = 8e-4     # Haline contraction coefficient

# ## Boundary conditions
#
# Here we define `Flux` boundary conditions at the surface for `u`, `T`, and `S`,
# and a `Gradient` boundary condition on `T` that maintains a constant stratification
# at the bottom. Our flux boundary condition for salinity uses a function that calculates
# the salinity flux in terms of the evaporation rate.

u_bcs = HorizontallyPeriodicBCs(top = BoundaryCondition(Flux, Qᵘ))

T_bcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Flux, Qᵀ),
                                bottom = BoundaryCondition(Gradient, ∂T∂z))

## Salinity flux: Qˢ = - E * S
@inline Qˢ(i, j, grid, time, iter, U, C, p) = @inbounds -p.evaporation * C.S[i, j, 1]

S_bcs = HorizontallyPeriodicBCs(top = BoundaryCondition(Flux, Qˢ))

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
                 grid = RegularCartesianGrid(size=(Nz, Nz, Nz), grid=(Δz*Nz, Δz*Nz, Δz*Nz)),
             coriolis = FPlane(f=f),
             buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=α, β=β)),
              closure = AnisotropicMinimumDissipation(),
  boundary_conditions = BoundaryConditions(u=u_bcs, T=T_bcs, S=S_bcs),
           parameters = (evaporation = evaporation,)
)

# To use the Smagorinsky-Lilly turbulence closure (with a constant model coefficient), use
#
# ```julia
# closure = ConstantSmagorinsky()
# ```
#
# To change the `architecture` to `GPU`, replace the `architecture` keyword argument with
#
# ```julia
# architecture = GPU()
# ```
#
# ## Initial conditions
#
# Out initial condition for temperature consists of a linear stratification superposed with
# random noise damped at the walls, while our initial condition for velocity consists
# only of random noise.

## Random noise damped at top and bottom
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise

## Temperature initial condition: a stable density tradient with random noise superposed.
T₀(x, y, z) = 20 + ∂T∂z * z + ∂T∂z * model.grid.Lz * 1e-1 * Ξ(z)

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

## Instantiate a JLD2OutputWriter to write fields.
field_writer = JLD2OutputWriter(model, FieldOutputs(fields_to_output); interval=hour/4,
                                prefix="ocean_wind_mixing_and_convection", force=true)

## Add the output writer to the models `output_writers`.
model.output_writers[:fields] = field_writer

# ## Running the simulation
#
# To run the simulation, we instantiate a `TimeStepWizard` to ensure stable time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 0.2.

wizard = TimeStepWizard(cfl=0.2, Δt=1.0, max_change=1.1, max_Δt=5.0)

# A diagnostic that returns the maximum absolute value of `w` by calling
# `wmax(model)`:

wmax = FieldMaximum(abs, model.velocities.w)

# We also create a figure and define a plotting function for live plotting of results.

fig, axs = subplots(ncols=3, figsize=(12, 5))

"""
    makeplot!(axs, model)

Make a triptych of x-z slices of vertical velocity, temperature, and salinity
associated with `model` in `axs`.
"""
function makeplot!(axs, model)
    jhalf = floor(Int, model.grid.Nz/2)

    ## Coordinate arrays for plotting
    xC = repeat(model.grid.xC, 1, model.grid.Nz)
    zF = repeat(reshape(model.grid.zF[1:end-1], 1, model.grid.Nz), model.grid.Nx, 1)
    zC = repeat(reshape(model.grid.zC, 1, model.grid.Nz), model.grid.Nx, 1)

    sca(axs[1]); cla()
    title("Vertical velocity")
    pcolormesh(xC, zF, data(model.velocities.w)[:, jhalf, :])
    xlabel("\$ x \$ (m)"); ylabel("\$ z \$ (m)")

    sca(axs[2]); cla()
    title("Temperature")
    pcolormesh(xC, zC, data(model.tracers.T)[:, jhalf, :])
    xlabel("\$ x \$ (m)")

    sca(axs[3]); cla()
    title("Salinity")
    pcolormesh(xC, zC, data(model.tracers.S)[:, jhalf, :])
    xlabel("\$ x \$ (m)")

    [ax.set_aspect(1) for ax in axs]
    pause(0.01)

    return nothing
end

# Finally, we run the the model in a `while` loop.

while model.clock.time < end_time

    ## Update the time step associated with `wizard`.
    update_Δt!(wizard, model)

    ## Time step the model forward
    walltime = @elapsed time_step!(model, 10, wizard.Δt)

    ## Print a progress message
    @printf("i: %04d, t: %s, Δt: %s, wmax = %.1e ms⁻¹, wall time: %s\n",
            model.clock.iteration, prettytime(model.clock.time), prettytime(wizard.Δt),
            wmax(model), prettytime(walltime))

    model.architecture == CPU() && makeplot!(axs, model)
end
