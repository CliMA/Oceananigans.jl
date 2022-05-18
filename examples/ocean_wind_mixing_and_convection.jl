# # [Wind- and convection-driven mixing in an ocean surface boundary layer](@id gpu_example)
#
# This example simulates mixing by three-dimensional turbulence in an ocean surface
# boundary layer driven by atmospheric winds and convection. It demonstrates:
#
#   * How to set-up a grid with varying spacing in the vertical direction
#   * How to use the `SeawaterBuoyancy` model for buoyancy with a linear equation of state.
#   * How to use a turbulence closure for large eddy simulation.
#   * How to use a function to impose a boundary condition.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# ```

# We start by importing all of the packages and functions that we'll need for this
# example.

using Random
using Printf
using CairoMakie

using Oceananigans
using Oceananigans.Units: minute, minutes, hour

# ## The grid
#
# We use 32²×24 grid points with 2 m grid spacing in the horizontal and
# varying spacing in the vertical, with higher resolution closer to the
# surface. Here we use a stretching function for the vertical nodes that
# maintains relatively constant vertical spacing in the mixed layer, which
# is desirable from a numerical standpoint:

Nz = 24          # number of points in the vertical direction
Lz = 32          # (m) domain depth

refinement = 1.2 # controls spacing near surface (higher means finer spaced)
stretching = 12  # controls rate of stretching at bottom

## Normalized height ranging from 0 to 1
h(k) = (k - 1) / Nz

## Linear near-surface generator
ζ₀(k) = 1 + (h(k) - 1) / refinement

## Bottom-intensified stretching function 
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

## Generating function
z_faces(k) = Lz * (ζ₀(k) * Σ(k) - 1)

grid = RectilinearGrid(size = (32, 32, Nz), 
                          x = (0, 64),
                          y = (0, 64),
                          z = z_faces)

# We plot vertical spacing versus depth to inspect the prescribed grid stretching:

plot(grid.Δzᵃᵃᶜ[1:grid.Nz], grid.zᵃᵃᶜ[1:grid.Nz],
     marker = :circle,
     ylabel = "Depth (m)",
     xlabel = "Vertical spacing (m)",
     legend = nothing)

# ## Buoyancy that depends on temperature and salinity
#
# We use the `SeawaterBuoyancy` model with a linear equation of state,

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 2e-4,
                                                                    haline_contraction = 8e-4))

# ## Boundary conditions
#
# We calculate the surface temperature flux associated with surface heating of
# 200 W m⁻², reference density `ρₒ`, and heat capacity `cᴾ`,

Qʰ = 200  # W m⁻², surface _heat_ flux
ρₒ = 1026 # kg m⁻³, average density at the surface of the world ocean
cᴾ = 3991 # J K⁻¹ kg⁻¹, typical heat capacity for seawater

Qᵀ = Qʰ / (ρₒ * cᴾ) # K m s⁻¹, surface _temperature_ flux

# Finally, we impose a temperature gradient `dTdz` both initially and at the
# bottom of the domain, culminating in the boundary conditions on temperature,

dTdz = 0.01 # K m⁻¹

T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵀ),
                                bottom = GradientBoundaryCondition(dTdz))

# Note that a positive temperature flux at the surface of the ocean
# implies cooling. This is because a positive temperature flux implies
# that temperature is fluxed upwards, out of the ocean.
#
# For the velocity field, we imagine a wind blowing over the ocean surface
# with an average velocity at 10 meters `u₁₀`, and use a drag coefficient `cᴰ`
# to estimate the kinematic stress (that is, stress divided by density) exerted
# by the wind on the ocean:

u₁₀ = 10    # m s⁻¹, average wind velocity 10 meters above the ocean
cᴰ = 2.5e-3 # dimensionless drag coefficient
ρₐ = 1.225  # kg m⁻³, average density of air at sea-level

Qᵘ = - ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀) # m² s⁻²

# The boundary conditions on `u` are thus

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

# For salinity, `S`, we impose an evaporative flux of the form

@inline Qˢ(x, y, t, S, evaporation_rate) = - evaporation_rate * S # [salinity unit] m s⁻¹
nothing # hide

# where `S` is salinity. We use an evporation rate of 1 millimeter per hour,

evaporation_rate = 1e-3 / hour # m s⁻¹

# We build the `Flux` evaporation `BoundaryCondition` with the function `Qˢ`,
# indicating that `Qˢ` depends on salinity `S` and passing
# the parameter `evaporation_rate`,

evaporation_bc = FluxBoundaryCondition(Qˢ, field_dependencies=:S, parameters=evaporation_rate)

# The full salinity boundary conditions are

S_bcs = FieldBoundaryConditions(top=evaporation_bc)

# ## Model instantiation
#
# We fill in the final details of the model here: upwind-biased 5th-order
# advection for momentum and tracers, 3rd-order Runge-Kutta time-stepping,
# Coriolis forces, and the `AnisotropicMinimumDissipation` closure
# for large eddy simulation to model the effect of turbulent motions at
# scales smaller than the grid scale that we cannot explicitly resolve.

model = NonhydrostaticModel(advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            grid = grid,
                            tracers = (:T, :S),
                            coriolis = FPlane(f=1e-4),
                            buoyancy = buoyancy,
                            closure = AnisotropicMinimumDissipation(),
                            boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))

# Notes:
#
# * To use the Smagorinsky-Lilly turbulence closure (with a constant model coefficient) rather than
#   `AnisotropicMinimumDissipation`, use `closure = SmagorinskyLilly()` in the model constructor.
#
# * To change the `architecture` to `GPU`, replace `architecture = CPU()` with
#   `architecture = GPU()`.

# ## Initial conditions
#
# Our initial condition for temperature consists of a linear stratification superposed with
# random noise damped at the walls, while our initial condition for velocity consists
# only of random noise.

## Random noise damped at top and bottom
Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise

## Temperature initial condition: a stable density gradient with random noise superposed.
Tᵢ(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 1e-6 * Ξ(z)

## Velocity initial condition: random noise scaled by the friction velocity.
uᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)

## `set!` the `model` fields using functions or constants:
set!(model, u=uᵢ, w=uᵢ, T=Tᵢ, S=35)

# ## Setting up a simulation
#
# We set-up a simulation with an initial time-step of 10 seconds
# that stops at 40 minutes, with adaptive time-stepping and progress printing.

simulation = Simulation(model, Δt=10.0, stop_time=40minutes)

# The `TimeStepWizard` helps ensure stable time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 1.0.

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1minute)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Nice progress messaging is helpful:

## Print a progress message
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim),
                                prettytime(sim),
                                prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.w),
                                prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

# We then set up the simulation:

# ## Output
#
# We use the `JLD2OutputWriter` to save ``x, z`` slices of the velocity fields,
# tracer fields, and eddy diffusivities. The `prefix` keyword argument
# to `JLD2OutputWriter` indicates that output will be saved in
# `ocean_wind_mixing_and_convection.jld2`.

## Create a NamedTuple with eddy viscosity
eddy_viscosity = (; νₑ = model.diffusivity_fields.νₑ)

filename = "ocean_wind_mixing_and_convection"

simulation.output_writers[:slices] =
    JLD2OutputWriter(model, merge(model.velocities, model.tracers, eddy_viscosity),
                     filename = filename * ".jld2",
                     indices = (:, grid.Ny/2, :),
                     schedule = TimeInterval(1minute),
                     overwrite_existing = true)

# We're ready:

run!(simulation)

# ## Turbulence visualization
#
# We animate the data saved in `ocean_wind_mixing_and_convection.jld2`.
# We prepare for animating the flow by loading the data into
# FieldTimeSeries and defining functions for computing colorbar limits.

filepath = filename * ".jld2"

time_series = (w = FieldTimeSeries(filepath, "w"),
               T = FieldTimeSeries(filepath, "T"),
               S = FieldTimeSeries(filepath, "S"),
               νₑ = FieldTimeSeries(filepath, "νₑ"))

## Coordinate arrays
xw, yw, zw = nodes(time_series.w)
xT, yT, zT = nodes(time_series.T)

""" Return colorbar levels equispaced between `(-clim, clim)` and encompassing the extrema of `c`. """
function divergent_levels(c, clim, nlevels=21)
    cmax = maximum(abs, c)
    levels = clim > cmax ? range(-clim, stop=clim, length=nlevels) : range(-cmax, stop=cmax, length=nlevels)

    return (levels[1], levels[end]), levels
end

""" Return colorbar levels equispaced between `clims` and encompassing the extrema of `c`."""
function sequential_levels(c, clims, nlevels=20)
    levels = range(clims[1], stop=clims[2], length=nlevels)
    cmin, cmax = minimum(c), maximum(c)
    cmin < clims[1] && (levels = vcat([cmin], levels))
    cmax > clims[2] && (levels = vcat(levels, [cmax]))

    return clims, levels
end
nothing # hide

# We start the animation at ``t = 10minutes`` since things are pretty boring till then:

times = time_series.w.times
intro = searchsortedfirst(times, 10minutes)

# We are now ready to animate using Makie. We use Makie's `Observable` to animate
# the data. To dive into how `Observable`s work we refer to
# [Makie.jl's Documentation](https://makie.juliaplots.org/stable/documentation/nodes/index.html).

n = Observable(intro)

 wₙ = @lift interior(time_series.w[$n],  :, 1, :)
 Tₙ = @lift interior(time_series.T[$n],  :, 1, :)
 Sₙ = @lift interior(time_series.S[$n],  :, 1, :)
νₑₙ = @lift interior(time_series.νₑ[$n], :, 1, :)

fig = Figure(resolution = (1000, 500))

axis_kwargs = (xlabel="x (m)",
               ylabel="z (m)",
               aspect = AxisAspect(grid.Lx/grid.Lz),
               limits = ((0, grid.Lx), (-grid.Lz, 0)))

ax_w = Axis(fig[2, 1]; title = "vertical velocity", axis_kwargs...)

ax_T = Axis(fig[2, 3]; title = "temperature", axis_kwargs...)

ax_S = Axis(fig[3, 1]; title = "salinity", axis_kwargs...)

ax_νₑ = Axis(fig[3, 3]; title = "eddy viscocity", axis_kwargs...)

title = @lift @sprintf("t = %s", prettytime(times[$n]))

wlims = (-0.05, 0.05)
Tlims = (19.7, 19.99)
Slims = (35, 35.005)
νₑlims = (1e-6, 5e-3)

hm_w = heatmap!(ax_w, xw, zw, wₙ; colormap = :balance, colorrange = wlims)
Colorbar(fig[2, 2], hm_w; label = "m s⁻¹")

hm_T = heatmap!(ax_T, xT, zT, Tₙ; colormap = :thermal, colorrange = Tlims)
Colorbar(fig[2, 4], hm_T; label = "ᵒC")

hm_S = heatmap!(ax_S, xT, zT, Sₙ; colormap = :haline, colorrange = Slims)
Colorbar(fig[3, 2], hm_S; label = "g / kg")

hm_νₑ = heatmap!(ax_νₑ, xT, zT, νₑₙ; colormap = :thermal, colorrange = νₑlims)
Colorbar(fig[3, 4], hm_νₑ; label = "m s⁻²")

fig[1, :] = Label(fig, title, textsize=24, tellwidth=false)

# And now record a movie.

frames = intro:length(times)

record(fig, filename * ".mp4", frames, framerate=8) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end
nothing #hide

# ![](ocean_wind_mixing_and_convection.mp4)
