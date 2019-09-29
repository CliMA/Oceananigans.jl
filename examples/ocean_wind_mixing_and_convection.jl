# # Wind and convection-driven mixing in an ocean surface boundary layer
#
# This example simulates mixing by three-dimensional turbulence in an ocean surface
# boundary layer driven by atmospheric winds and convection. It demonstrates:
#
#   * how to use the `SeawaterBuoyancy` model for buoyancy with an equation of state;
#   * how to use a turbulence closure for large eddy simulation.

using Oceananigans, Random, Printf, PyPlot

# ## Model parameters
#
# Here we use an isotropic, cubic grid with `Nz` grid points and grid spacing
# `Δz = 1` meter. We specify the temperature flux `QT` at the top of the domain,
# which is related to the heat flux at the top via
#
# ```math
# Q_T = \frac{Q_h}{\rho_0 c_P} \, ,
# ```
#
# where ``Q_h`` is the heat flux, ``\rho_0`` is a reference density, and ``c_P`` is 
# the heat capacity of seawater. With a reference density ``\rho_0 = 1026`` and heat 
# capacity ``c_P = 3991``, our chosen temperature flux of 
# ``Q_T = 10^{-4} \, \mathrm{K \, m^{-1} \, s^{-1}}`` corresponds to a heat flux of
# ``Q_h = 409.4 \, \mathrm{W \, m^{-2}}``.
#
# Finally, we use an initial temperature gradient of ``\partial_z T = 0.005``,
# which implies an iniital buoyancy frequency
#
# ```math
# N\^2 = \alpha g \partial_z T = 9.8 \times 10^{-6} \, \mathrm{s^{-2}}
# ```
#
# with a thermal expansion coefficient ``\alpha = 2 \times 10^{-4} \, \mathrm{K^{-1}}`` 
# and gravitational acceleration ``g = 9.81 \, \mathrm{m \, s^{-2}}``. Note that, by default,
# the `SeawaterBuoyancy` model uses a gravitational acceleration 
# ``g_\mathrm{Earth} = 9.80665 \, \mathrm{m \, s^{-2}}``. 

      Nz = 48       # Number of grid points in x, y, z
      Δz = 1.0      # Grid spacing in x, y, z (meters)         
      QT = 1e-4     # Temperature flux at surface
      Qu = -5e-5    # Velocity flux at surface
    ∂T∂z = 0.005    # Initial vertical temperature gradient
       f = 1e-4     # Coriolis parameter
       α = 2e-4     # Thermal expansion coefficient
end_time = 2hour    # End time for the simulation

# ## Boundary conditions
#
# Here we define `Flux` boundary conditions at the surface for both `u` and `T`,
# and a `Gradient` boundary condition on `T` that maintains a constant stratification
# at the bottom.

ubcs = HorizontallyPeriodicBCs(top = BoundaryCondition(Flux, Qu))

Tbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, QT),
                                bottom = BoundaryCondition(Gradient, ∂T∂z))

# ## Model instantiation
#
# We instantiate a horizontally-periodic `Model` on the CPU with on a `RegularCartesianGrid`, 
# using a `FPlane` model for rotation (constant rotation rate), a linear equation 
# of state for temperature and salinity, the Anisotropic Minimum Dissipation closure 
# to model the effects of unresolved turbulence, and the previously defined boundary
# conditions for `u` and `T`.

model = Model(
           architecture = CPU(),
                   grid = RegularCartesianGrid(N=(Nz, Nz, Nz), L=(Δz*Nz, Δz*Nz, Δz*Nz)),
               coriolis = FPlane(f=f),
               buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=α)),
                closure = AnisotropicMinimumDissipation(),
    boundary_conditions = BoundaryConditions(u=ubcs, T=Tbcs)
)

# To use the Smagorinsky-Lilly turbulence closure 
# (with a constant model coefficient), use
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
T₀(x, y, z) = 20 + ∂T∂z * z + ∂T∂z * model.grid.Lz * 1e-6 * Ξ(z)

## Velocity initial condition: random noise scaled by the friction velocity.
u₀(x, y, z) = sqrt(abs(Qu)) * 1e-3 * Ξ(z)

set!(model, u=u₀, w=u₀, T=T₀)

# ## Set up output
#
# We set up an output writer that saves all velocity fields, tracer fields, and the subgrid
# turbulent diffusivity associated with `model.closure`. The `prefix` keyword argument 
# to `JLD2OutputWriter` indicates that output will be saved in `wind_mixing_and_convection.jld2`.

## Create a NamedTuple containing all the fields to be outputted.
fields_to_output = merge(model.velocities, model.tracers, (νₑ=model.diffusivities.νₑ,))

## Instantiate a JLD2OutputWriter to write fields.
field_writer = JLD2OutputWriter(model, FieldOutputs(fields_to_output); interval=hour/4, 
                                prefix="wind_mixing_and_convection", force=true)
                                
## Add the output writer to the models `output_writers`.
model.output_writers[:fields] = field_writer

# ## Running the simulation
#
# To run the simulation, we instantiate a `TimeStepWizard` to ensure stable time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 0.2. In addition, we create a diagnostic
# for calculating the maximum vertical velocity, and create a figure for live plotting
# of simulation results.

wizard = TimeStepWizard(cfl=0.2, Δt=10.0, max_change=1.1, max_Δt=20.0)

wmax = FieldMaximum(abs, model.velocities.w)

fig, axs = subplots()
x = repeat(model.grid.xC, 1, model.grid.Nz)
z = repeat(reshape(model.grid.zF[1:end-1], 1, model.grid.Nz), model.grid.Nx, 1)

# Run the model
while model.clock.time < end_time

    ## Update the time step associated with `wizard`.
    update_Δt!(wizard, model)

    ## Time step the model forward
    walltime = @elapsed time_step!(model, 10, wizard.Δt)

    ## Print a progress message
    @printf("i: %04d, t: %s, Δt: %s, wmax = %.1e ms⁻¹, wall time: %s\n", model.clock.iteration, 
            prettytime(model.clock.time), prettytime(wizard.Δt), wmax(model), prettytime(walltime))

    ## Make a plot
    sca(axs); cla(); axs.set_aspect(1)
    pcolormesh(x, z, data(model.velocities.w)[:, floor(Int, Nz/2), :])
    xlabel("\$ x \$ (m)")
    ylabel("\$ z \$ (m)")
    title("Vertical velocity")
    pause(0.01)
end
