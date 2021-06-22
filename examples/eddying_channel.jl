# using Pkg
# pkg"add Oceananigans GLMakie"

using Printf
using Statistics
using GLMakie

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries

# # Vertically-stretched grid
#
# We build a vertically stretched grid with cell interfaces
# clustered near the surface, where mesoscale eddies are most vigorous.
#
# The domain is rectangular and twice as wide north-south as east-west.

const Lx = 1000kilometers # east-west extent [m]
const Ly = 2000kilometers # north-south extent [m]
const Lz = 3kilometers    # depth [m]

# We use a resolution that implies O(10 km) grid spacing in the horizontal
# and a vertical grid spacing that varies from O(10 m) to O(100 m),

Nx = 32
Ny = 2Nx
Nz = 16

#=
# Vertical stretching is accomplished with an exponential "stretching function",

s = 1.5 # stretching factor
z_faces(k) = - Lz * (1 - tanh(s * (k - 1) / Nz) / tanh(s))

@show grid = VerticallyStretchedRectilinearGrid(topology = (Periodic, Bounded, Bounded),
                                                size = (Nx, Ny, Nz),
                                                halo = (3, 3, 3),
                                                x = (-Lx/2, Lx/2),
                                                y = (0, Ly),
                                                z_faces = z_faces)

# We visualize the cell interfaces by plotting the cell height
# as a function of depth,

fig = Figure(resolution=(400, 600))

ax = Axis(fig[1, 1],
          xlabel = "Cell height Δz (m)",
          ylabel = "z (m)",
          xscale = log10,
          xticks = [20, 50, 100, 200, 500])

scatter!(ax, grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz])

display(fig)
=#

@show grid = RegularRectilinearGrid(topology = (Periodic, Bounded, Bounded),
                                        size = (Nx, Ny, Nz),
                                        halo = (3, 3, 3),
                                           x = (-Lx/2, Lx/2),
                                           y = (0, Ly),
                                           z = (-Lz, 0))

# # Boundary conditions
#
# A channel-centered jet and overturning circulation are driven by wind stress
# and an alternating pattern of surface cooling and surface heating with
# parameters

Qᵇ = 0.0 #1e-8       # buoyancy flux magnitude [m² s⁻³]
y_shutoff = 5/6 * Ly # shutoff location for buoyancy flux [m]
τ = 1e-4             # surface kinematic wind stress [m² s⁻²]
μ = 1 / 100days      # bottom drag damping time-scale [s⁻¹]

# The buoyancy flux has a sinusoidal pattern in `y`,

@inline buoyancy_flux(x, y, t, p) = ifelse(y < p.y_shutoff, p.Qᵇ * cos(3π * y / p.Ly), 0)

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, parameters=(Ly=grid.Ly, y_shutoff=y_shutoff, Qᵇ=Qᵇ))

# At the surface we impose a wind stress with sinusoidal variation in `y`,

@inline u_stress(x, y, t, p) = - p.τ * sin(π * y / p.Ly)

u_stress_bc = FluxBoundaryCondition(u_stress, parameters=(τ=τ, Ly=grid.Ly))

# Linear bottom drag on `u` and `v` provides a sink of momentum

@inline u_drag(x, y, t, u, μ) = - μ * u
@inline v_drag(x, y, t, v, μ) = - μ * v

u_drag_bc = FluxBoundaryCondition(u_drag, field_dependencies=:u, parameters=μ)
v_drag_bc = FluxBoundaryCondition(v_drag, field_dependencies=:v, parameters=μ)

# To summarize,

b_bcs = TracerBoundaryConditions(grid, top = buoyancy_flux_bc)
u_bcs = UVelocityBoundaryConditions(grid, top = u_stress_bc, bottom = u_drag_bc)
v_bcs = VVelocityBoundaryConditions(grid, bottom = v_drag_bc)

# # Coriolis
#
# We use a ``β``-plane model to capture the effect of meridional
# variations in the planetary vorticity,

#coriolis = BetaPlane(latitude=-45)
coriolis = FPlane(latitude=-45)

# # Sponge layer and initial condition
#
# We use a geostrophically-balanced initial condition with a
# linear meridional buoyancy gradient and linear vertical shear.
#
# The geostrophic streamfunction is
#
# ```math
# ψ(y, z) = - α y (z + L_z) \, ,
# ```
#
# with parameters
#
const α = 1e-3         # geostrophic shear [s⁻¹]
#
# corresponding to a barotropic streamfunction ``Ψ = - α y L_z / 2``,
# a free surface displacement
#
# ```math
# η = - \frac{f₀ α y L_z}{2 g} \, ,
# ```
#
# with

#const f₀ = coriolis.f₀ # background planetary vorticity [s⁻¹]
const f₀ = coriolis.f # background planetary vorticity [s⁻¹]
g = 9.81 # m s⁻²

# The coriolis parameter

@show f₀ 

# is < 0 due to our austral focus.
# The geostrophic buoyancy field ``b = f₀ ∂_z ψ′``, where ``ψ′``
# is the baroclinic component of ``ψ``, is then

@inline b_geostrophic(y) = - α * f₀ * y

# and the zonal velocity ``u = - ∂_y ψ`` is

u_geostrophic(z) = α * (z + Lz/2)

# We also impose an initial stratification with surface buoyancy gradient
# and scale height

const N² = 1e-5               # surface vertical buoyancy gradient [s⁻²]
const h = 1kilometer          # decay scale of stable stratification [m]

@inline b_stratification(z) = N² * h * exp(z / h)

# We introduce a sponge layer adjacent the northern boundary to restore
# the buoyancy field on a time-scale of 30 days to the initial condition.
# The sponge layer, surface forcing, and net transport by the eddy field
# leads to the development of a diabatic overturning circulation.
# We impose the sponge layer with a ramp function that decays to zero within
# `y_sponge` of the northern boundary.

const y_sponge = 9/10 * Ly # southern boundary of sponge layer [m]

## Mask that limits sponge layer to a thin region near the northern boundary
@inline northern_mask(x, y, z) = max(0, y - y_sponge) / (Ly - y_sponge)

## Target and initial buoyancy profile
@inline b_target(x, y, z, t) = b_geostrophic(y) + b_stratification(z)

b_forcing = Relaxation(target=b_target, mask=northern_mask, rate=1/10days)

# The annotations `const` on global variables above ensure that our forcing functions
# compile on the GPU, while the annotation `@inline` ensures efficient execution.
#
# # Turbulence closures
#
# A horizontally Laplacian diffusivity destroys enstrophy and buoyancy variance
# created by mesoscale turbulence, while a convective adjustment scheme creates
# a surface mixed layer due to surface cooling.

horizontal_diffusivity = AnisotropicDiffusivity(νh = 10, κh = 10)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 1.0,
                                                                background_κz = 1e-3,
                                                                background_νz = 1e-3)

# # Model building
#
# We build a model on a BetaPlane with an ImplicitFreeSurface.

model = HydrostaticFreeSurfaceModel(
           architecture = CPU(),                                           
                   grid = grid,
           free_surface = ImplicitFreeSurface(gravitational_acceleration = g),
     momentum_advection = UpwindBiasedThirdOrder(),
       tracer_advection = UpwindBiasedThirdOrder(),
               buoyancy = BuoyancyTracer(),
               coriolis = coriolis,
                closure = (convective_adjustment, horizontal_diffusivity),
                tracers = :b,
    #boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
    #            forcing = (b=b_forcing,),
)

#=
model = IncompressibleModel(
           architecture = CPU(),                                           
                   grid = grid,
              advection = UpwindBiasedThirdOrder(),
               buoyancy = BuoyancyTracer(),
               coriolis = coriolis,
                closure = horizontal_diffusivity,
                tracers = :b,
    #boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
    #            forcing = (b=b_forcing,),
)
=#

# # InitiaL conditions
#
# Our initial condition superposes the previously discussed geostrophic flow
# with surface-concentrated random noise scaled by the total velocity
# jump in the vertical and concentrated in the upper tenth of the domain.

## Random noise
u★ = 1e-3 * α * Lz
ϵ(x, y, z) = u★ * exp(- (y - Ly/2)^2 / (2 * (0.1Ly)^2)) * randn()

ηᵢ(x, y) = - f₀ * α * (y - Ly/2) * Lz / (2g)
uᵢ(x, y, z) = u_geostrophic(z) + ϵ(x, y, z)
bᵢ(x, y, z) = b_geostrophic(y) + b_stratification(z)
set!(model, u=uᵢ, b=bᵢ, η=ηᵢ)

#=
uᵢ(x, y, z) = ϵ(x, y, z)
bᵢ(x, y, z) = b_stratification(z)
set!(model, u=uᵢ, b=bᵢ)
=#

# # Simulation setup
#
# We set up a simulation with adaptive time-stepping and a simple progress message.

using Oceananigans.Diagnostics: accurate_cell_advection_timescale

wizard = TimeStepWizard(cfl=0.1, Δt=5minutes, max_change=1.1, max_Δt=30minutes, min_Δt=1minute,
                        cell_advection_timescale = accurate_cell_advection_timescale)

print_progress(sim) = @printf("[%05.2f%%] i: %d, t: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
                              100 * (sim.model.clock.time / sim.stop_time),
                              sim.model.clock.iteration,
                              prettytime(sim.model.clock.time),
                              maximum(abs, sim.model.velocities.u),
                              maximum(abs, sim.model.velocities.v),
                              maximum(abs, sim.model.velocities.w),
                              prettytime(sim.Δt.Δt))

simulation = Simulation(model, Δt=wizard, stop_time=10days, progress=print_progress, iteration_interval=10)

u, v, w = model.velocities
b = model.tracers.b

ζ = ComputedField(∂x(v) - ∂y(u))

B = AveragedField(b, dims=(1, 2))
χ_op = @at (Center, Center, Center) ∂x(b - B)^2 + ∂y(b - B)^2 + ∂z(b - B)^2
χ = ComputedField(χ_op)

outputs = merge(model.velocities, model.tracers, (ζ=ζ, χ=χ))

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(100days),
                                                        prefix = "eddying_channel",
                                                        force = true)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(1hour),
                                                      prefix = "eddying_channel",
                                                      field_slicer = nothing,
                                                      force = true)

try
    run!(simulation, pickup=false)
catch err
    showerror(stdout, err)
end

# # Visualizing the solution with GLMakie
#
# We make a volume rendering of the solution using GLMakie.

u_timeseries = FieldTimeSeries("eddying_channel.jld2", "u")
v_timeseries = FieldTimeSeries("eddying_channel.jld2", "v")
w_timeseries = FieldTimeSeries("eddying_channel.jld2", "w")
b_timeseries = FieldTimeSeries("eddying_channel.jld2", "b")
ζ_timeseries = FieldTimeSeries("eddying_channel.jld2", "ζ")
χ_timeseries = FieldTimeSeries("eddying_channel.jld2", "χ")

xζ, yζ, zζ = nodes((Face, Face, Center), grid)
xu, yu, zu = nodes((Face, Center, Center), grid)
xv, yv, zv = nodes((Center, Face, Center), grid)
xw, yw, zw = nodes((Center, Center, Face), grid)
xc, yc, zc = nodes((Center, Center, Center), grid)

kwargs = (algorithm=:absorption, absorption=0.5, colormap=:balance, show_axis=true)

iter = Node(1)

u′ = @lift interior(u_timeseries[$iter])
v′ = @lift interior(v_timeseries[$iter])
w′ = @lift interior(w_timeseries[$iter])
b′ = @lift interior(b_timeseries[$iter]) .- mean(interior(b_timeseries[$iter]))
ζ′ = @lift interior(ζ_timeseries[$iter])
χ′ = @lift interior(χ_timeseries[$iter])

function symmetric_lims(q, iter)
    q_max = maximum(abs, interior(q[iter]))
    return (-q_max, q_max)
end

u_lims = @lift symmetric_lims(u_timeseries, $iter)
w_lims = @lift symmetric_lims(w_timeseries, $iter)
b_lims = @lift symmetric_lims(b_timeseries, $iter)
ζ_lims = @lift symmetric_lims(ζ_timeseries, $iter)
χ_lims = @lift symmetric_lims(χ_timeseries, $iter)

fig = Figure(resolution = (2000, 1600))

ax_u = fig[1, 1] = LScene(fig)
ax_v = fig[1, 2] = LScene(fig)
ax_w = fig[2, 1] = LScene(fig)
ax_b = fig[2, 2] = LScene(fig)

volume!(ax_u, xu * 1e-3, yu * 1e-3, zu / 10, u′; colorrange=u_lims, kwargs...) 
volume!(ax_v, xv * 1e-3, yv * 1e-3, zu / 10, v′; colorrange=u_lims, kwargs...) 
volume!(ax_w, xw * 1e-3, yw * 1e-3, zu / 10, w′; colorrange=w_lims, kwargs...) 
volume!(ax_b, xc * 1e-3, yc * 1e-3, zc / 10, b′; colorrange=b_lims, kwargs...) 

#volume!(ax_ζ, xζ * 1e-3, yζ * 1e-3, zζ / 10, ζ′; colorrange=ζ_lims, kwargs...) 
#volume!(ax_χ, xc * 1e-3, yc * 1e-3, zc / 10, χ′; colorrange=χ_lims, kwargs...) 

nframes = length(ζ_timeseries.times)

record(fig, "eddying_channel.mp4", 1:nframes, framerate=12) do i
    @info "Plotting frame $i of $nframes..."
    iter[] = i
end
