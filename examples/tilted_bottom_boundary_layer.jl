# # Tilted bottom boundary layer example
#
# This example simulates a two-dimensional (x-z) tilted oceanic bottom boundary layer with a
# constant background alongslope (y-direction) velocity. It demonstrates how to simulate a domain
# tilt by
#
#   * Changing the direction of the buoyancy acceleration
#   * Changing the axis of rotation for Coriolis
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, Plots"
# ```

# ## Load `Oceananigans.jl` and define problem constants
#
# We set a named tuple that defines the necessary simulation parameters:

using Oceananigans
using Oceananigans.Units
using Printf
using CUDA

params = (f₀ = 1e-4, #1/s
          V∞ = 0.1, # m/s
          N²∞ = 1e-5, # 1/s²
          θ_rad = 0.05,
          ν = 5e-4, # m²/s
          z₀ = 0.1, # m (roughness length)
          Lx = 1000, # m
          Lz = 100, # m
          )


# Here `f₀` is the Coriolis frequency, `V∞` in the constant interior `v`-velocity, `N²∞` is the
# interior stratification, `θ_rad` is the bottom slope in radians, `ν` is the eddy viscosity, and
# `z₀` is the roughness length (needed for the drag at the bottom). `Lx` and `Lz` are the domain
# length and height.
#
#
# ## Creating the grid
#
# Since we anticipate most of the energy in the flow to be located close to the bottom, we
# can save computational resources by creating a grid that has finer spacings near the bottom
# which gets progressively coarser near the top. Such a grid can be achieved with the following
# equations:

Nx = 128
Nz = 32

refinement = 1.8 # controls spacing near surface (higher means finer spaced)
stretching = 10   # controls rate of stretching at bottom 

h₁(k) = ((-k+Nz)+1) / Nz

## Linear near-surface generator
ζ₁(k) = 1 + (h₁(k) - 1) / refinement

## Bottom-intensified stretching function 
Σ₁(k) = (1 - exp(-stretching * h₁(k))) / (1 - exp(-stretching))

## Generating function
z_faces(k) = -params.Lz * (ζ₁(k) * Σ₁(k) - 1)


grid = RectilinearGrid(topology=(Periodic, Flat, Bounded),
                       size=(Nx, Nz),
                       x=(0, params.Lx), z=z_faces,
                       halo=(3,3),
                       )

# We plot vertical spacing versus depth to inspect the prescribed grid stretching. Note that
# the spacing near the bottom is relatively uniform with height, which is a desired property from
# a numerical perspective.
#

using Plots

plot(grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz],
     marker = :circle,
     ylabel = "Depth (m)",
     xlabel = "Vertical spacing (m)",
     legend = nothing)


# ## Setting up a buoyancy model in a tilted domain
#
# We set-up our domain in a way that the coordinates align with the tilted bottom. That means that,
# from the coordinate's point of view, gravity is tilted by `θ_rad` radians, which we can simulate
# by passing the `gravity_unit_vector` parameter to `Buoyancy()`:

ĝ = [sin(params.θ_rad), 0, cos(params.θ_rad)]
buoyancy = Buoyancy(model=BuoyancyTracer(), gravity_unit_vector=ĝ)




# ## Background fields
#
# Since we are simulating a period domain that is tilted, we need to set-up the background
# stratification as a `BackgroundField` and solve for the buoyancy perturbation. This avoids mean
# horizontal buoyant gradients, which cannot exist in a horizontally-periodic domain:

b∞(x, y, z, t, p) = p.N²∞ * (x * sin(p.θ_rad) + z * cos(p.θ_rad))
B_field = BackgroundField(b∞, parameters=(; params.N²∞, params.θ_rad))


# We also set the constant interior velocity `V∞` as a background field in order to avoid inertial
# oscillations due to an unbalanced resolved flow:
#

V_bg(x, y, z, t, p) = p.V∞
V_field = BackgroundField(V_bg, parameters=(; V∞=params.V∞))



# ## Bottom drag
#
# We set-up a bottom drag that follows Monin-Obukhov theory. Note that we need to include`V∞`
# in the velocity, since we will model it as a background field.
#

const κ = 0.4 # von Karman constant
z₁ = CUDA.@allowscalar znodes(Center, grid)[1] # Closest grid center to the bottom
cᴰ = (κ / log(z₁/params.z₀))^2 # Drag coefficient

@inline drag_u(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * u
@inline drag_v(x, y, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * (v + p.V∞)

drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(cᴰ=cᴰ, V∞=params.V∞))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(cᴰ=cᴰ, V∞=params.V∞))

u_bcs = FieldBoundaryConditions(bottom = drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom = drag_bc_v)



# ## Create the `NonhydrostaticModel`
#
# Since the coordinate system is aligned with the tilted domain, we can use a traditional f-plane
# approximation (only keep the rotation component that's aligned with gravity) by also tilting the
# rotation from the coordinate system's perpective by `θ_rad` radians using
# `ConstantCartesianCoriolis` and passing the `rotation_axis` argument:
#

coriolis = ConstantCartesianCoriolis(f=params.f₀, rotation_axis=ĝ)

#
# We are now ready to create the model. We create a `NonhydrostaticModel` with an
# `UpwindBiasedFifthOrder` advection scheme and a `RungeKutta3` timestepper.  For simplicity, we use
# a constant-diffusivity turbulence closure:

model = NonhydrostaticModel(grid = grid, timestepper = :RungeKutta3,
                            advection = UpwindBiasedFifthOrder(),
                            buoyancy = buoyancy,
                            coriolis = coriolis,
                            tracers = :b,
                            closure = ScalarDiffusivity(ν=params.ν, κ=params.ν),
                            boundary_conditions = (u=u_bcs, v=v_bcs,),
                            background_fields = (b=B_field, v=V_field,),
                           )


# ## Create and run a simulation
#
# We are now ready to create the simulation. We begin by setting the initial time step
# conservatively, based on the smallest grid size of our domain and set-up a 

using Oceananigans.Grids: min_Δz
simulation = Simulation(model, 
                        Δt=0.5*min_Δz(grid)/params.V∞,
                        stop_time=3days, 
                        )

# We now add callbacks to adjust the time-step and to display the simulation progress:

wizard = TimeStepWizard(max_change=1.1, cfl=0.7)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

start_time = time_ns() # so we can print the total elapsed wall time
progress_message(sim) =
    @printf("i: %04d, t: %s, Δt: %s, wmax = %.1e ms⁻¹, wall time: %s\n",
            iteration(sim), prettytime(time(sim)),
            prettytime(sim.Δt), maximum(abs, sim.model.velocities.w),
            prettytime((time_ns() - start_time) * 1e-9))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))


# ## Add outputs to the simulation
#
# We add outputs to our model using the `NetCDFOutputWriter`

u, v, w = model.velocities

b_tot = Field(model.tracers.b + model.background_fields.tracers.b)
v_tot = Field(v + model.background_fields.velocities.v)
ω_y = Field(∂z(u)-∂x(w))

fields = (; u, v_tot, w, b_tot, ω_y)

simulation.output_writers[:fields] = NetCDFOutputWriter(model, fields, 
                                                        filepath = joinpath(@__DIR__, "out.tilted_bbl.nc"),
                                                        schedule = TimeInterval(20minutes),
                                                        mode = "c")

# Now we just run it!

run!(simulation)


# ## Visualize the results
#
# First we load the required package to load NetCDF output files and define the coordinates for
# plotting using existing objects:
#

using NCDatasets
xᶠ, zᶠ = xnodes(ω_y), znodes(ω_y);
xᶜ, zᶜ = xnodes(v_tot), znodes(v_tot) ;

# Define keyword arguments for plotting function
#
kwargs = (xlabel = "x",
          zlabel = "z",
          fill = true,
          levels = 30,
          linewidth = 0,
          color = :balance,
          colorbar = true,
          xlim = (0, params.Lx),
          zlim = (0, params.Lz)
         )


# Read in the output_writer for the two-dimensional fields and then create an animation showing the
# vorticity.

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

anim = @animate for (iter, t) in enumerate(ds["time"])
    ω_y = ds["ω_y"][:,1,:, iter]
    v_tot = ds["v_tot"][:,1,:,iter]

    ω_max = maximum(abs, ω_y)
    v_max = maximum(abs, v_tot)

    plot_ω = contour(xᶠ, zᶠ, ω_y',
                     clim = (-0.015, +0.015),
                     title = @sprintf("y-vorticity, ω_y, at t = %.1f", t); kwargs...)

    plot_v = contour(xᶜ, zᶜ, v_tot',
                     clim = (-params.V∞, +params.V∞),
                     title = @sprintf("Total along-slope velocity, at t = %.1f", t); kwargs...)

    plot(plot_ω, plot_v, layout = (2, 1), size = (800, 440))
end

close(ds)

mp4(anim, "tilted_bottom_boundary_layer.mp4", fps=15)
