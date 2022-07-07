# # Langmuir turbulence example
#
# This example implements a Langmuir turbulence simulation reported in section
# 4 of
#
# > [Wagner et al., "Near-inertial waves and turbulence driven by the growth of swell", Journal of Physical Oceanography (2021)](https://journals.ametsoc.org/view/journals/phoc/51/5/JPO-D-20-0178.1.xml)
#
# This example demonstrates
#
#   * How to run large eddy simulations with surface wave effects via the
#     Craik-Leibovich approximation.
#
#   * How to specify time- and horizontally-averaged output.

# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# ```

using Oceananigans
using Oceananigans.Units: minute, minutes, hours

# ## Model set-up
#
# To build the model, we specify the grid, Stokes drift, boundary conditions, and
# Coriolis parameter.
#
# ### Domain and numerical grid specification
#
# We use a modest resolution and the same total extent as Wagner et al. 2021,

grid = RectilinearGrid(size=(32, 32, 32), extent=(128, 128, 64))

# ### The Stokes Drift profile
#
# The surface wave Stokes drift profile prescribed in Wagner et al. 2021,
# corresponds to a 'monochromatic' (that is, single-frequency) wave field.
#
# A monochromatic wave field is characterized by its wavelength and amplitude
# (half the distance from wave crest to wave trough), which determine the wave
# frequency and the vertical scale of the Stokes drift profile.

using Oceananigans.BuoyancyModels: g_Earth

 amplitude = 0.8 # m
wavelength = 60 # m
wavenumber = 2π / wavelength # m⁻¹
 frequency = sqrt(g_Earth * wavenumber) # s⁻¹

## The vertical scale over which the Stokes drift of a monochromatic surface wave
## decays away from the surface is `1/2wavenumber`, or
const vertical_scale = wavelength / 4π

## Stokes drift velocity at the surface
const Uˢ = amplitude^2 * wavenumber * frequency # m s⁻¹

# The `const` declarations ensure that Stokes drift functions compile on the GPU.
# To run this example on the GPU, write `architecture = GPU()` in the constructor
# for `NonhydrostaticModel` below.
#
# The Stokes drift profile is

uˢ(z) = Uˢ * exp(z / vertical_scale)

# which we'll need for the initial condition.
#
# !!! info "The Craik-Leibovich equations in Oceananigans"
#     Oceananigans implements the Craik-Leibovich approximation for surface wave effects
#     using the _Lagrangian-mean_ velocity field as its prognostic momentum variable.
#     In other words, `model.velocities.u` is the Lagrangian-mean ``x``-velocity beneath surface
#     waves. This differs from models that use the _Eulerian-mean_ velocity field
#     as a prognostic variable, but has the advantage that ``u`` accounts for the total advection
#     of tracers and momentum, and that ``u = v = w = 0`` is a steady solution even when Coriolis
#     forces are present. See the
#     [physics documentation](https://clima.github.io/OceananigansDocumentation/stable/physics/surface_gravity_waves/)
#     for more information.
#
# The vertical derivative of the Stokes drift is

∂z_uˢ(z, t) = 1 / vertical_scale * Uˢ * exp(z / vertical_scale)

# Finally, we note that the time-derivative of the Stokes drift must be provided
# if the Stokes drift and surface wave field undergoes _forced_ changes in time.
# In this example, the Stokes drift is constant
# and thus the time-derivative of the Stokes drift is 0.

# ### Boundary conditions
#
# At the surface at ``z=0``, Wagner et al. 2021 impose

Qᵘ = -3.72e-5 # m² s⁻², surface kinematic momentum flux

u_boundary_conditions = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

# Wagner et al. 2021 impose a linear buoyancy gradient `N²` at the bottom
# along with a weak, destabilizing flux of buoyancy at the surface to faciliate
# spin-up from rest.

Qᵇ = 2.307e-8 # m² s⁻³, surface buoyancy flux
N² = 1.936e-5 # s⁻², initial and bottom buoyancy gradient

b_boundary_conditions = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ),
                                                bottom = GradientBoundaryCondition(N²))

# !!! info "The flux convention in Oceananigans"
#     Note that Oceananigans uses "positive upward" conventions for all fluxes. In consequence,
#     a negative flux at the surface drives positive velocities, and a positive flux of
#     buoyancy drives cooling.

# ### Coriolis parameter
#
# Wagner et al. (2021) use

coriolis = FPlane(f=1e-4) # s⁻¹

# which is typical for mid-latitudes on Earth.

# ## Model instantiation
#
# We are ready to build the model. We use a fifth-order Weighted Essentially
# Non-Oscillatory (WENO) advection scheme and the `AnisotropicMinimumDissipation`
# model for large eddy simulation. Because our Stokes drift does not vary in ``x, y``,
# we use `UniformStokesDrift`, which expects Stokes drift functions of ``z, t`` only.

model = NonhydrostaticModel(; grid, coriolis,
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=∂z_uˢ),
                            boundary_conditions = (u=u_boundary_conditions, b=b_boundary_conditions))

# ## Initial conditions
#
# We make use of random noise concentrated in the upper 4 meters
# for buoyancy and velocity initial conditions,

Ξ(z) = randn() * exp(z / 4)
nothing # hide

# Our initial condition for buoyancy consists of a surface mixed layer 33 m deep,
# a deep linear stratification, plus noise,

initial_mixed_layer_depth = 33 # m
stratification(z) = z < - initial_mixed_layer_depth ? N² * z : N² * (-initial_mixed_layer_depth)

bᵢ(x, y, z) = stratification(z) + 1e-1 * Ξ(z) * N² * model.grid.Lz

# The simulation we reproduce from Wagner et al. (2021) is zero Lagrangian-mean velocity.
# This initial condition is consistent with a wavy, quiescent ocean suddenly impacted
# by winds. To this quiescent state we add noise scaled by the friction velocity to ``u`` and ``w``.

u★ = sqrt(abs(Qᵘ))
uᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
wᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, b=bᵢ)

# ## Setting up the simulation

simulation = Simulation(model, Δt=45.0, stop_time=4hours)

# We use the `TimeStepWizard` for adaptive time-stepping
# with a Courant-Freidrichs-Lewy (CFL) number of 1.0,

wizard = TimeStepWizard(cfl=1.0, max_change=1.1, max_Δt=1minute)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# ### Nice progress messaging
#
# We define a function that prints a helpful message with
# maximum absolute value of ``u, v, w`` and the current wall clock time.

using Printf

function progress(simulation)
    u, v, w = simulation.model.velocities

    ## Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

# ## Output
#
# ### A field writer
#
# We set up an output writer for the simulation that saves all velocity fields,
# tracer fields, and the subgrid turbulent diffusivity.

output_interval = 5minutes

fields_to_output = merge(model.velocities, model.tracers, (; νₑ=model.diffusivity_fields.νₑ))

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, fields_to_output,
                     schedule = TimeInterval(output_interval),
                     filename = "langmuir_turbulence_fields.jld2",
                     overwrite_existing = true)

# ### An "averages" writer
#
# We also set up output of time- and horizontally-averaged velocity field and
# momentum fluxes,

u, v, w = model.velocities
b = model.tracers.b

 U = Average(u, dims=(1, 2))
 V = Average(v, dims=(1, 2))
 B = Average(b, dims=(1, 2))
wu = Average(w * u, dims=(1, 2))
wv = Average(w * v, dims=(1, 2))

simulation.output_writers[:averages] =
    JLD2OutputWriter(model, (; U, V, B, wu, wv),
                     schedule = AveragedTimeInterval(output_interval, window=2minutes),
                     filename = "langmuir_turbulence_averages.jld2",
                     overwrite_existing = true)

# ## Running the simulation
#
# This part is easy,

run!(simulation)

# # Making a neat movie
#
# We look at the results by loading data from file with FieldTimeSeries,
# and plotting vertical slices of ``u`` and ``w``, and a horizontal
# slice of ``w`` to look for Langmuir cells.

using CairoMakie

time_series = (;
     w = FieldTimeSeries("langmuir_turbulence_fields.jld2", "w"),
     u = FieldTimeSeries("langmuir_turbulence_fields.jld2", "u"),
     B = FieldTimeSeries("langmuir_turbulence_averages.jld2", "B"),
     U = FieldTimeSeries("langmuir_turbulence_averages.jld2", "U"),
     V = FieldTimeSeries("langmuir_turbulence_averages.jld2", "V"),
    wu = FieldTimeSeries("langmuir_turbulence_averages.jld2", "wu"),
    wv = FieldTimeSeries("langmuir_turbulence_averages.jld2", "wv"))

times = time_series.w.times
xw, yw, zw = nodes(time_series.w)
xu, yu, zu = nodes(time_series.u)
nothing # hide

# We are now ready to animate using Makie. We use Makie's `Observable` to animate
# the data. To dive into how `Observable`s work we refer to
# [Makie.jl's Documentation](https://makie.juliaplots.org/stable/documentation/nodes/index.html).

n = Observable(1)

wxy_title = @lift string("w(x, y, t) at z=-8 m and t = ", prettytime(times[$n]))
wxz_title = @lift string("w(x, z, t) at y=0 m and t = ", prettytime(times[$n]))
uxz_title = @lift string("u(x, z, t) at y=0 m and t = ", prettytime(times[$n]))

fig = Figure(resolution = (850, 850))

ax_B = Axis(fig[1, 4];
            xlabel = "Buoyancy (m s⁻²)",
            ylabel = "z (m)")

ax_U = Axis(fig[2, 4];
            xlabel = "Velocities (m s⁻¹)",
            ylabel = "z (m)",
            limits = ((-0.07, 0.07), nothing))

ax_fluxes = Axis(fig[3, 4];
                 xlabel = "Momentum fluxes (m² s⁻²)",
                 ylabel = "z (m)",
                 limits = ((-3.5e-5, 3.5e-5), nothing))

ax_wxy = Axis(fig[1, 1:2];
              xlabel = "x (m)",
              ylabel = "y (m)",
              aspect = DataAspect(),
              limits = ((0, grid.Lx), (0, grid.Ly)),
              title = wxy_title)

ax_wxz = Axis(fig[2, 1:2];
              xlabel = "x (m)",
              ylabel = "z (m)",
              aspect = AxisAspect(2),
              limits = ((0, grid.Lx), (-grid.Lz, 0)),
              title = wxz_title)

ax_uxz = Axis(fig[3, 1:2];
              xlabel = "x (m)",
              ylabel = "z (m)",
              aspect = AxisAspect(2),
              limits = ((0, grid.Lx), (-grid.Lz, 0)),
              title = uxz_title)

nothing #hide

wₙ = @lift time_series.w[$n]
uₙ = @lift time_series.u[$n]
Bₙ = @lift time_series.B[$n][1, 1, :]
Uₙ = @lift time_series.U[$n][1, 1, :]
Vₙ = @lift time_series.V[$n][1, 1, :]
wuₙ = @lift time_series.wu[$n][1, 1, :]
wvₙ = @lift time_series.wv[$n][1, 1, :]

k = searchsortedfirst(grid.zᵃᵃᶠ[:], -8)
wxyₙ = @lift interior(time_series.w[$n], :, :, k)
wxzₙ = @lift interior(time_series.w[$n], :, 1, :)
uxzₙ = @lift interior(time_series.u[$n], :, 1, :)

wlims = (-0.03, 0.03)
ulims = (-0.05, 0.05)

lines!(ax_B, Bₙ, zu)

lines!(ax_U, Uₙ, zu; label = L"\bar{u}")
lines!(ax_U, Vₙ, zu; label = L"\bar{v}")
axislegend(ax_U; position = :rb)

lines!(ax_fluxes, wuₙ, zw; label = L"mean $wu$")
lines!(ax_fluxes, wvₙ, zw; label = L"mean $wv$")
axislegend(ax_fluxes; position = :rb)

hm_wxy = heatmap!(ax_wxy, xw, yw, wxyₙ;
                  colorrange = wlims,
                  colormap = :balance)

Colorbar(fig[1, 3], hm_wxy; label = "m s⁻¹")

hm_wxz = heatmap!(ax_wxz, xw, zw, wxzₙ;
                  colorrange = wlims,
                  colormap = :balance)

Colorbar(fig[2, 3], hm_wxz; label = "m s⁻¹")

ax_uxz = heatmap!(ax_uxz, xu, zu, uxzₙ;
                  colorrange = ulims,
                  colormap = :balance)

Colorbar(fig[3, 3], ax_uxz; label = "m s⁻¹")

# And, finally, we record a movie.

frames = 1:length(times)

record(fig, "langmuir_turbulence.mp4", frames, framerate=8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
nothing #hide

# ![](langmuir_turbulence.mp4)
