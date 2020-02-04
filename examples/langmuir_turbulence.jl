# # Langmuir turbulence example
#
# This example implements the Langmuir turbulence simulation reported in section
# 4 of McWilliams, J. C. et al., "Langmuir Turbulence in the ocean," Journal of 
# Fluid Mechanics (1997). This example demonstrates:
#   
#   * how to run a simulation with surface wave effects via the Craik-Leibovich
#     approximation
#
# In addition to `Oceananigans.jl` we need `PyPlot` for plotting, `Random` for
# generating random initial conditions, and `Printf` for printing progress messages.

using Oceananigans, Oceananigans.Diagnostics, Oceananigans.OutputWriters, Random, Printf
using Oceananigans: g_Earth
using Oceananigans.SurfaceWaves: UniformStokesDrift

# ## Model parameters
#
# Here we use an anisotropic grid with `Nh` horizontal grid points, `Nz` vertical
# grid points, `Δh` horizontal grid spacing, and `Δz` vertical grid spacing. 
# We specify fluxes of buouyancy and momentum that match the specifications in
# McWilliams and Sullivan (1997) via
#
#   1. A buoyancy flux `Qᵇ` at the top of the domain.
#
#   2. A velocity flux `Qᵘ` at the top of the domain, which is related
#       to the `x` momentum flux `τˣ` via `τˣ = ρ₀ * Qᵘ`, where `ρ₀` is a reference density.
#
# We also choose the wavenumber and amplitude of the surface wave field, and use an
# initial buoyancy gradient of `N² = 1.936e-5`.

      Nh = 32       # Number of grid points in x, y, z
      Nz = 48       # Number of grid points in x, y, z
      Δh = 4.0      # [m] Grid spacing in x, y, z (meters)
      Δz = 2.0      # [m] Grid spacing in x, y, z (meters)
      Qᵘ = -3.72e-5 # [m² s⁻²] Velocity flux / stress at surface
      Qᵇ = 2.307e-9 # [m³ s⁻²] Buoyancy flux at surface
      N² = 1.936e-5 # [s⁻²] Initial buoyancy gradient
       f = 1e-4     # [s⁻¹] Coriolis parameter
end_time = 2hour    # [s] End time for the simulation

# Surface wave stokes drift profile: 
#
# $ uˢ(z) = Uˢ exp(2kˢʷ z) $
#
# and
#
# $ ∂_z uˢ(z) = 2 kˢʷ Uˢ exp(2kˢʷ z) $
#
# Notes:
#
#  * We use `const`s so Stokes drift functions compile on GPU.
#
#  * The `Oceananigans.jl` implementation of Craik-Leibovich uses the Lagrangian-mean
#    formulation, which means that the vertical derivative or 'Stokes shear' is specified.
#    If the surface wave field Stokes drift changes in time, the time-derivative of the
#    Stokes drift profile must be provided as well.

const kˢʷ = 0.105    # [m⁻¹] Surface wave wavenumber
const aˢʷ = 0.8      # [m] Surface wave amplitude
const Uˢ = aˢʷ^2 * kˢʷ * sqrt(g_Earth * kˢʷ)

## Oceananigans
∂z_uˢ(z, t) = 2kˢʷ * Uˢ * exp(2kˢʷ * z) 

# We use the Stokes drift profile for the initial condition
uˢ(z) = Uˢ * exp(2kˢʷ * z)

# ## Boundary conditions
#
# Here we define `Flux` boundary conditions at the surface for `u`, `T`, and `S`,
# and a `Gradient` boundary condition on `T` that maintains a constant stratification
# at the bottom. Our flux boundary condition for salinity uses a function that calculates
# the salinity flux in terms of the evaporation rate.

u_bcs = HorizontallyPeriodicBCs(top = BoundaryCondition(Flux, Qᵘ))

b_bcs = HorizontallyPeriodicBCs(top = BoundaryCondition(Flux, Qᵇ), 
                                bottom = BoundaryCondition(Gradient, N²))

# ## Model instantiation
#
# We instantiate a horizontally-periodic `Model` on the CPU with on a `RegularCartesianGrid`,
# using a `FPlane` model for rotation (constant rotation rate), the Anisotropic Minimum 
# Dissipation closure to model the effects of unresolved turbulence, and the previously defined 
# boundary conditions for `u` and `b`.

model = Model(
         architecture = CPU(),
                 grid = RegularCartesianGrid(size=(Nh, Nh, Nz), length=(Δh*Nh, Δh*Nh, Δz*Nz)),
             buoyancy = BuoyancyTracer(), tracers = :b,
              closure = AnisotropicMinimumDissipation(),
        surface_waves = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
  boundary_conditions = BoundaryConditions(u=u_bcs, b=b_bcs),
)

# ## Initial conditions
#
# Out initial condition for buoyancy consists of a linear stratification superposed with
# random noise damped at the walls.

## Random noise damped at top and bottom
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise

## Temperature initial condition: a stable density tradient with random noise superposed.
b₀(x, y, z) = N² * z + 1e-1 * Ξ(z) * N² * model.grid.Lz

# Velocity initial condition: note that McWilliams and Sullivan (1997) use a model
# that is formulated in terms of the Eulerian-mean velocity field. Due to this, 
# their 'resting' initial condition actually consists of a sheared and 
# unbalanced Lagrangian-mean velocity field uˢ(z). To this we add noise.
u₀(x, y, z) = uˢ(z) + sqrt(abs(Qᵘ)) * 1e-1 * Ξ(z)

# Finally, we introduce noise into the vertical velocity:
w₀(x, y, z) = sqrt(abs(Qᵘ)) * 1e-1 * Ξ(z)

set!(model, u=u₀, w=w₀, b=b₀)

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
                                prefix="langmuir_turbulence", force=true)

## Add the output writer to the models `output_writers`.
model.output_writers[:fields] = field_writer;

# ## Running the simulation
#
# To run the simulation, we instantiate a `TimeStepWizard` to ensure stable time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 0.2.

wizard = TimeStepWizard(cfl=0.2, Δt=5.0, max_change=1.1, max_Δt=10.0)

# Diagnostic that returns the maximum absolute value of `u, v, w` by calling
# `umax(), vmax(), `wmax()`:

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

# We also create a figure and define a plotting function for live plotting of results.
makeplot = false

using PyPlot

fig, axs = subplots(ncols=3, figsize=(12, 5))

"""
    makeplot!(axs, model)

Make a triptych of x-z slices of vertical velocity, temperature, and salinity
associated with `model` in `axs`.
"""
function makeplot!(axs, model)
    jhalf = floor(Int, model.grid.Ny/2)

    ## Coordinate arrays for plotting
    xCˣᶻ = repeat(model.grid.xC, 1, model.grid.Nz)
    xFˣᶻ = repeat(model.grid.xF[1:end-1], 1, model.grid.Nz)
    zCˣᶻ = repeat(reshape(model.grid.zC, 1, model.grid.Nz), model.grid.Nx, 1)
    zFˣᶻ = repeat(reshape(model.grid.zF[1:end-1], 1, model.grid.Nz), model.grid.Nx, 1)

    xCˣʸ = repeat(model.grid.xC, 1, model.grid.Ny)
    yCˣʸ = repeat(reshape(model.grid.yC, 1, model.grid.Ny), model.grid.Nx, 1)

    sca(axs[1]); cla()
    title("Horizontal velocity")
    pcolormesh(xFˣᶻ, zCˣᶻ, Array(interior(model.velocities.u))[:, jhalf, :])
    xlabel("\$ x \$ (m)"); ylabel("\$ z \$ (m)")

    sca(axs[2]); cla()
    title("Vertical velocity")
    pcolormesh(xCˣᶻ, zFˣᶻ, Array(interior(model.velocities.w))[:, jhalf, :])
    xlabel("\$ x \$ (m)")

    sca(axs[3]); cla()
    title("Vertical velocity at \$ z = $(model.grid.zF[Nz-2]) \$ meters")
    pcolormesh(xCˣʸ, yCˣʸ, Array(interior(model.velocities.w))[:, :, Nz-2])
    xlabel("\$ x \$ (m)"); ylabel("\$ y \$ (m)")

    axs[3].yaxis.set_label_position("right")
    axs[3].tick_params(right=true, labelright=true, left=false, labelleft=false)

    [ax.set_aspect(1) for ax in axs]

    tight_layout()

    pause(0.01)

    return nothing
end

# Finally, we run the the model in a `while` loop.

while model.clock.time < end_time

    ## Update the time step associated with `wizard`.
    update_Δt!(wizard, model)

    ## Time step the model forward
    walltime = @elapsed time_step!(model, 100, wizard.Δt)

    ## Print a progress message
    @printf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
            model.clock.iteration, prettytime(model.clock.time), prettytime(wizard.Δt),
            umax(), vmax(), wmax(), prettytime(walltime))

    makeplot && makeplot!(axs, model)
end

# Show the reults in a plot

makeplot!(axs, model)
gcf()
