# # Tilted bottom boundary layer example
#
# This example simulates a two-dimensional tilted oceanic bottom boundary layer based 
# on Wenegrat et al. (2020). It demonstrates how to tilt the domain by
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
          Lx = 1000, # m
          Lz = 100, # m
          Nx = 64,
          Nz = 64,
          ν = 5e-4, # m²/s
          sponge_frac = 1/5,
          sponge_rate = √1e-5, # 1/s
          z₀ = 0.1, # m (roughness length)
          )

arch = CPU()


# Here `f₀` is the Coriolis frequency, `V∞` in the interior `v`-velocity, `N²∞` is the
# interior stratification, `θ_rad` is the bottom slope in radians, `ν` is the eddy viscosity,
# and `z₀` is the roughness length (needed for the drag at the bottom).
#
#
# ## Creating the grid
#
# Since we anticipate most of the energy in the flow to be located close to the bottom, we
# can save computational resources by creating a grid that has finer spacings near the bottom
# which gets progressively coarser near the top. Such a grid can be achieved with the following
# equations:
#


refinement = 1.8 # controls spacing near surface (higher means finer spaced)
stretching = 10   # controls rate of stretching at bottom 

h₁(k) = ((-k+params.Nz)+1) / params.Nz

## Linear near-surface generator
ζ₁(k) = 1 + (h₁(k) - 1) / refinement

## Bottom-intensified stretching function 
Σ₁(k) = (1 - exp(-stretching * h₁(k))) / (1 - exp(-stretching))

## Generating function
z_faces(k) = -params.Lz * (ζ₁(k) * Σ₁(k) - 1)


grid = VerticallyStretchedRectilinearGrid(topology=topo,
                                          architecture = arch,
                                          size=(params.Nx, params.Nz),
                                          x=(0, params.Lx), z_faces=z_faces,
                                          halo=(3,3),
                                          )
@info grid


# We plot vertical spacing versus depth to inspect the prescribed grid stretching. Note that
# the spacing near the bottom is relatively uniform with height, which is a desired property from
# a numerical perspective.
#
#
# ## Setting up a buoyancy model in a tilted domain
#
# We set-up our domain in a way that the coordinates align with the bottom. That means that, from
# the coordinate's point of view, gravity needs to be tilted by `θ_rad` radians, which we can do by
# passing the `vertical_units_vector` parameter to `Buoyancy()`:

ĝ = [sin(params.θ_rad), 0, cos(params.θ_rad)]
buoyancy = Buoyancy(model=BuoyancyTracer(), vertical_unit_vector=ĝ)

# Since we are simulating a period domain that is tilted, we need to set-up the background
# stratification as a `BackgroundField` and solve for the buoyancy perturbation. This avoids mean
# horizontal buoyant gradients, which cannot exist in a horizontally-periodic domain:

b∞(x, y, z, t, p) = p.N²∞ * (x * sin(p.θ_rad) + z * cos(p.θ_rad))
B_field = BackgroundField(b∞, parameters=(; params.N²∞, params.θ_rad))


# ## Bottom drag
#
# We also set-up a bottom drag that follows Monin-Obukhov theory, which can be implemented following https://doi.org/10.1029/2005WR004685.
# Note that we need to include`V∞` in the velocity, since we will model it as a background field.
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

V_bg(x, y, z, t, p) = p.V∞
V_field = BackgroundField(V_bg, parameters=(; V∞=params.V∞))
#-----


# ## Sponge layers
#
# In order to damp internal wave propagation, we use a sponge layer in the upper part of the domain.
# One such sponge layer can be defined as

@inline heaviside(X) = ifelse(X < 0, zero(X), one(X))

const sp_frac = params.sponge_frac
const Lz = params.Lz

function top_mask_2nd(x, y, z)
    z₁ = +Lz; z₀ = z₁ - Lz * sp_frac 
    return heaviside((z - z₀)/(z₁ - z₀)) * ((z - z₀)/(z₁ - z₀))^2
end

full_sponge_0 = Relaxation(rate=params.sponge_rate, mask=top_mask_2nd, target=0)
forcing = (u=full_sponge_0, v=full_sponge_0, w=full_sponge_0,)
#----



# ## Create the `NonhydrostaticModel`
#
# We create model with a `WENO5` advection scheme and a `RungeKutta3` timestepper. For simplicity,
# we use a constant-diffusivity turbulence closure:

model = NonhydrostaticModel(grid = grid, timestepper = :RungeKutta3,
                            architecture = arch,
                            advection = WENO5(),
                            buoyancy = buoyancy,
                            coriolis = ConstantCartesianCoriolis(f=params.f₀, rotation_axis=ĝ),
                            tracers = :b,
                            closure = IsotropicDiffusivity(ν=params.ν, κ=params.ν),
                            boundary_conditions = (u=u_bcs, v=v_bcs,),
                            background_fields = (b=B_field, v=V_field,),
                            forcing=forcing,
                           )


# ## Create a simulation
#
#

using Oceananigans.Grids: min_Δz
if ndims==3
    cfl=0.85
else
    cfl=0.5
end
wizard = TimeStepWizard(max_change=1.03, cfl=cfl)

# Print a progress message
progress_message(sim) =
    @printf("i: %04d, t: %s, Δt: %s, wmax = %.1e ms⁻¹, wall time: %s\n",
            iteration(sim), prettytime(time(sim)),
            prettytime(sim.Δt), maximum(abs, sim.model.velocities.w),
            prettytime((time_ns() - start_time) * 1e-9))

simulation = Simulation(model, Δt=0.5*min_Δz(grid)/params.V∞, 
                        stop_time=3days, 
                       )
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))
#----


# ## Add outputs to the simulation:
#

u, v, w = model.velocities
b_tot = ComputedField(model.tracers.b + model.background_fields.tracers.b)
v_tot = ComputedField(v + model.background_fields.velocities.v)
ω_y = ComputedField(∂z(u)-∂x(w))
fields = merge((; u, v_tot, w, b_tot, ω_y))
simulation.output_writers[:fields] =
NetCDFOutputWriter(model, fields, filepath = "out.tilted_bbl.nc",
                   schedule = TimeInterval(20minutes),
                   mode = "c")
#----
#

start_time = time_ns() # so we can print the total elapsed wall time
run!(simulation)
