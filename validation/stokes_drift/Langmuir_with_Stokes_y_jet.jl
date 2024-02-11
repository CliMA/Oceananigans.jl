# This validation case simulates Langmuir turbulence under a spatially-varying wave field
# The code is based on the existing Oceananigans Langmuir turbulence example

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

grid = RectilinearGrid(size=(64, 32, 32), extent=(256, 128, 64))

# ### The Stokes Drift profile
#
# We utilize the same monochromatic wave parameters as Wagner et al. 2021,

using Oceananigans.BuoyancyModels: g_Earth

 amplitude = 0.8 # m
wavelength = 60  # m
wavenumber = 2π / wavelength # m⁻¹
 frequency = sqrt(g_Earth * wavenumber) # s⁻¹

## The vertical scale over which the Stokes drift of a monochromatic surface wave
## decays away from the surface is `1/2wavenumber`, or
const vertical_scale = wavelength / 4π

## Stokes drift velocity at the surface
const Uˢ = amplitude^2 * wavenumber * frequency # m s⁻¹

stokes_jet_center = 70
stokes_jet_central_width = 40
stokes_jet_edge_width = 40

# The `const` declarations ensure that Stokes drift functions compile on the GPU.
# To run this example on the GPU, include `GPU()` in the
# constructor for `RectilinearGrid` above.
#
# The Stokes drift profile at the core of the jet is
# 
# ```
# vˢ(x, y, z, t) = Uˢ * exp(z / vertical_scale) * exp( - (x - stokes_jet_center)^2 / (2 * stokes_jet_width^2) ) * 0.5 * ( 1 + 0.1 * cos(2 * pi * (y - grid.Ly/2) / grid.Ly ) )
# ```

# Create a Stokes drift field that is a cosine function within a subregion of the domain.
# This function peaks at `y = stokes_jet_center` with a  value of `2*Uˢ`, reaches zero at a distance of 
# `stokes_jet_width` either side of the peak, and is zero beyond those regions. 
# The zeroing of regions outside the jet is achieved through application of a Heaviside function
# 
# ```
# vˢ(x, y, z, t) = Uˢ * exp(z / vertical_scale) * 0.5 * (1 + cos(π * (y - stokes_jet_center) / stokes_jet_width)) * 0.5 * (sign(y - stokes_jet_center + stokes_jet_width)  -  sign(y - stokes_jet_center - stokes_jet_width) )
# ```

vˢ(x, y, z, t) = Uˢ * exp(z / vertical_scale) * 0.5 * (
    (1 + cos(π * (x - stokes_jet_center + stokes_jet_central_width / 2) / stokes_jet_edge_width)) *
    0.5 * (sign(x - stokes_jet_center + stokes_jet_central_width / 2 + stokes_jet_edge_width)  -  sign(x - stokes_jet_center + stokes_jet_central_width / 2) )
    + (1 + cos(π * (x - stokes_jet_center - stokes_jet_central_width / 2) / stokes_jet_edge_width)) *
    0.5 * (sign(x - stokes_jet_center - stokes_jet_central_width / 2) - sign(x - stokes_jet_center - stokes_jet_central_width / 2 - stokes_jet_edge_width) )
    + (sign(x - stokes_jet_center + stokes_jet_central_width / 2)  -  sign(x - stokes_jet_center - stokes_jet_central_width / 2)) ) *
    0.5 * ( 1 + 0.1 * cos(2π * (y - grid.Ly/2) / grid.Ly ) )


# and its `z`-derivative is

∂z_vˢ(x, y, z, t) = 1 / vertical_scale * Uˢ * exp(z / vertical_scale) * 0.5 * (
    (1 + cos(π * (x - stokes_jet_center + stokes_jet_central_width / 2) / stokes_jet_edge_width)) *
    0.5 * (sign(x - stokes_jet_center + stokes_jet_central_width / 2 + stokes_jet_edge_width)  -  sign(x - stokes_jet_center + stokes_jet_central_width / 2) )
    + (1 + cos(π * (x - stokes_jet_center - stokes_jet_central_width / 2) / stokes_jet_edge_width)) *
    0.5 * (sign(x - stokes_jet_center - stokes_jet_central_width / 2) - sign(x - stokes_jet_center - stokes_jet_central_width / 2 - stokes_jet_edge_width) )
    + (sign(x - stokes_jet_center + stokes_jet_central_width / 2)  -  sign(x - stokes_jet_center - stokes_jet_central_width / 2)) ) *
    0.5 * ( 1 + 0.1 * cos(2π * (y - grid.Ly/2) / grid.Ly ) )

∂x_vˢ(x, y, z, t) = - π / stokes_jet_edge_width * Uˢ * exp(z / vertical_scale) * 0.5 * (
    sin(π * (x - stokes_jet_center + stokes_jet_central_width / 2) / stokes_jet_edge_width) *
    0.5 * (sign(x - stokes_jet_center + stokes_jet_central_width / 2 + stokes_jet_edge_width)  -  sign(x - stokes_jet_center + stokes_jet_central_width / 2) )
    + sin(π * (x - stokes_jet_center - stokes_jet_central_width / 2) / stokes_jet_edge_width) *
    0.5 * (sign(x - stokes_jet_center - stokes_jet_central_width / 2) - sign(x - stokes_jet_center - stokes_jet_central_width / 2 - stokes_jet_edge_width) ) ) *
    0.5 * ( 1 + 0.1 * cos(2π * (y - grid.Ly/2) / grid.Ly ) )

∂y_vˢ(x, y, z, t) = - 2π / grid.Ly * Uˢ * exp(z / vertical_scale) * 0.5 * (
    sin(π * (x - stokes_jet_center + stokes_jet_central_width / 2) / stokes_jet_edge_width) *
    0.5 * (sign(x - stokes_jet_center + stokes_jet_central_width / 2 + stokes_jet_edge_width)  -  sign(x - stokes_jet_center + stokes_jet_central_width / 2) )
    + sin(π * (x - stokes_jet_center - stokes_jet_central_width / 2) / stokes_jet_edge_width) *
    0.5 * (sign(x - stokes_jet_center - stokes_jet_central_width / 2) - sign(x - stokes_jet_center - stokes_jet_central_width / 2 - stokes_jet_edge_width) ) ) *
    0.5 * 0.1 * sin(2π * (y - grid.Ly/2) / grid.Ly )

# Now diagnose the w component of Stokes drift using incompressibility and the surface boundary condition `wˢ(z=0)=0`

wˢ(x, y, z, t) = 2π / grid.Ly *vertical_scale * Uˢ * ( exp(z / vertical_scale) - 1 ) * 0.5 * (
    sin(π * (x - stokes_jet_center + stokes_jet_central_width / 2) / stokes_jet_edge_width) *
    0.5 * (sign(x - stokes_jet_center + stokes_jet_central_width / 2 + stokes_jet_edge_width)  -  sign(x - stokes_jet_center + stokes_jet_central_width / 2) )
    + sin(π * (x - stokes_jet_center - stokes_jet_central_width / 2) / stokes_jet_edge_width) *
    0.5 * (sign(x - stokes_jet_center - stokes_jet_central_width / 2) - sign(x - stokes_jet_center - stokes_jet_central_width / 2 - stokes_jet_edge_width) ) ) *
    0.5 * 0.1 * sin(2π * (y - grid.Ly/2) / grid.Ly )


# and its `z`-derivative is

∂z_wˢ(x, y, z, t) = 2π / grid.Ly * Uˢ * exp(z / vertical_scale) * 0.5 * (
    sin(π * (x - stokes_jet_center + stokes_jet_central_width / 2) / stokes_jet_edge_width) *
    0.5 * (sign(x - stokes_jet_center + stokes_jet_central_width / 2 + stokes_jet_edge_width)  -  sign(x - stokes_jet_center + stokes_jet_central_width / 2) )
    + sin(π * (x - stokes_jet_center - stokes_jet_central_width / 2) / stokes_jet_edge_width) *
    0.5 * (sign(x - stokes_jet_center - stokes_jet_central_width / 2) - sign(x - stokes_jet_center - stokes_jet_central_width / 2 - stokes_jet_edge_width) ) ) *
    0.5 * 0.1 * sin(2π * (y - grid.Ly/2) / grid.Ly )

∂x_wˢ(x, y, z, t) = 2π^2 / (grid.Ly * stokes_jet_edge_width) * vertical_scale * Uˢ * ( exp(z / vertical_scale) - 1 ) * 0.5 * (
    cos(π * (x - stokes_jet_center + stokes_jet_central_width / 2) / stokes_jet_edge_width) *
    0.5 * (sign(x - stokes_jet_center + stokes_jet_central_width / 2 + stokes_jet_edge_width)  -  sign(x - stokes_jet_center + stokes_jet_central_width / 2) )
    + cos(π * (x - stokes_jet_center - stokes_jet_central_width / 2) / stokes_jet_edge_width) *
    0.5 * (sign(x - stokes_jet_center - stokes_jet_central_width / 2) - sign(x - stokes_jet_center - stokes_jet_central_width / 2 - stokes_jet_edge_width) ) ) *
    0.5 * 0.1 * sin(2π * (y - grid.Ly/2) / grid.Ly )

∂y_wˢ(x, y, z, t) = - 4π^2 / (grid.Ly)^2 *vertical_scale * Uˢ * ( exp(z / vertical_scale) - 1 ) * 0.5 * (
    sin(π * (x - stokes_jet_center + stokes_jet_central_width / 2) / stokes_jet_edge_width) *
    0.5 * (sign(x - stokes_jet_center + stokes_jet_central_width / 2 + stokes_jet_edge_width)  -  sign(x - stokes_jet_center + stokes_jet_central_width / 2) )
    + sin(π * (x - stokes_jet_center - stokes_jet_central_width / 2) / stokes_jet_edge_width) *
    0.5 * (sign(x - stokes_jet_center - stokes_jet_central_width / 2) - sign(x - stokes_jet_center - stokes_jet_central_width / 2 - stokes_jet_edge_width) ) ) *
    0.5 * 0.1 * cos(2π * (y - grid.Ly/2) / grid.Ly )

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
# Finally, we note that the time-derivative of the Stokes drift must be provided
# if the Stokes drift and surface wave field undergoes _forced_ changes in time.
# In this example, the Stokes drift is constant **in time**
# and thus the time-derivative of the Stokes drift is 0.

# ### Boundary conditions
#
# At the surface at ``z=0``, Wagner et al. 2021 impose

Qᵛ = -3.72e-5 # m² s⁻², surface kinematic momentum flux

v_boundary_conditions = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵛ))

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
# No Coriolis force, to ensure localized "jet" structure of Stokes drift is clear

coriolis = FPlane(f=0) # s⁻¹

# which is typical for mid-latitudes on Earth.

# ## Model instantiation
#
# We are ready to build the model. We use a fifth-order Weighted Essentially
# Non-Oscillatory (WENO) advection scheme and the `AnisotropicMinimumDissipation`
# model for large eddy simulation. Because our Stokes drift does not vary in ``x, y``,
# we use `UniformStokesDrift`, which expects Stokes drift functions of ``z, t`` only.

model = NonhydrostaticModel(; grid, coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = StokesDrift(∂x_wˢ=∂x_wˢ,∂x_vˢ=∂x_vˢ,∂y_wˢ=∂y_wˢ,∂z_vˢ=∂z_vˢ),
                            boundary_conditions = (v=v_boundary_conditions, b=b_boundary_conditions))

# ## Initial conditions
#
# We make use of random noise concentrated in the upper 4 meters
# for buoyancy and velocity initial conditions,

Ξ(z) = randn() * exp(z / 4)
nothing #hide

# Our initial condition for buoyancy consists of a surface mixed layer 33 m deep,
# a deep linear stratification, plus noise,

initial_mixed_layer_depth = 33 # m
stratification(z) = z < - initial_mixed_layer_depth ? N² * z : N² * (-initial_mixed_layer_depth)

bᵢ(x, y, z) = stratification(z) + 1e-1 * Ξ(z) * N² * model.grid.Lz

# The simulation we reproduce from Wagner et al. (2021) is zero Lagrangian-mean velocity.
# This initial condition is consistent with a wavy, quiescent ocean suddenly impacted
# by winds. To this quiescent state we add noise scaled by the friction velocity to ``u`` and ``w``.

u★ = sqrt(abs(Qᵘ))
vᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
wᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)

set!(model, v=vᵢ, w=wᵢ, b=bᵢ)

# ## Setting up the simulation

simulation = Simulation(model, Δt=45.0, stop_time=30minutes)

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
                     filename = "Stokes_drift_jet_fields.jld2",
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
                     filename = "Stokes_drift_jet_averages.jld2",
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
     w = FieldTimeSeries("Stokes_drift_jet_fields.jld2", "w"),
     u = FieldTimeSeries("Stokes_drift_jet_fields.jld2", "u"),
     B = FieldTimeSeries("Stokes_drift_jet_averages.jld2", "B"),
     U = FieldTimeSeries("Stokes_drift_jet_averages.jld2", "U"),
     V = FieldTimeSeries("Stokes_drift_jet_averages.jld2", "V"),
    wu = FieldTimeSeries("Stokes_drift_jet_averages.jld2", "wu"),
    wv = FieldTimeSeries("Stokes_drift_jet_averages.jld2", "wv"))

times = time_series.w.times
xw, yw, zw = nodes(time_series.w)
xu, yu, zu = nodes(time_series.u)
nothing #hide

# We are now ready to animate using Makie. We use Makie's `Observable` to animate
# the data. To dive into how `Observable`s work we refer to
# [Makie.jl's Documentation](https://makie.juliaplots.org/stable/documentation/nodes/index.html).

n = Observable(1)

wxy_title = @lift string("w(x, y, t) at z=-8 m and t = ", prettytime(times[$n]))
wyz_title = @lift string("w(y, z, t) at x=0 m and t = ", prettytime(times[$n]))
uyz_title = @lift string("u(y, z, t) at x=0 m and t = ", prettytime(times[$n]))

fig = Figure(size = (1020, 1320))

ax_x_stokes = Axis(fig[2, 4];
                   xlabel = "y (m)",
                   ylabel = string("vˢ(y=", stokes_jet_center + 0.5* (stokes_jet_central_width + stokes_jet_edge_width), "m, z=-2m) terms ([m] s⁻¹)"))

ax_y_stokesu = Axis(fig[1, 1];
                    xlabel = string("vˢ(x=", 3*grid.Lx/8, "m, z=-2m) terms ([m] s⁻¹)"),
                    ylabel = "x (m)")

ax_xy_stokesu = Axis(fig[1, 2];
                     xlabel = "x (m)",
                     ylabel = "y (m)",
                     aspect = AxisAspect(1),
                     limits = ((0, grid.Lx), (0, grid.Ly)),
                     title = "vˢ(z=-2m)")

ax_xy_stokesw = Axis(fig[1, 4];
                     xlabel = "x (m)",
                     ylabel = "y (m)",
                     aspect = AxisAspect(1),
                     limits = ((0, grid.Lx), (0, grid.Ly)),
                     title = "wˢ(z=-2m) x 100")

ax_y_stokesw = Axis(fig[2, 1];
                    xlabel = string("wˢ(x=", 3*grid.Lx/8, "m, z=-2m) terms ([m] s⁻¹)"),
                    ylabel = "y (m)")

ax_U = Axis(fig[3, 4];
            xlabel = "Velocities (m s⁻¹)",
            ylabel = "z (m)",
            limits = ((-0.07, 0.07), nothing))

ax_fluxes = Axis(fig[4, 4];
                 xlabel = "Momentum fluxes (m² s⁻²)",
                 ylabel = "z (m)",
                 limits = ((-3.5e-5, 3.5e-5), nothing))

ax_wxy = Axis(fig[2, 2];
              xlabel = "x (m)",
              ylabel = "y (m)",
              aspect = AxisAspect(1),
              limits = ((0, grid.Lx), (0, grid.Ly)),
              title = wxy_title)

ax_wyz = Axis(fig[3, 1:2];
              xlabel = "y (m)",
              ylabel = "z (m)",
              aspect = AxisAspect(2),
              limits = ((0, grid.Ly), (-grid.Lz, 0)),
              title = wyz_title)

ax_uyz = Axis(fig[4, 1:2];
              xlabel = "y (m)",
              ylabel = "z (m)",
              aspect = AxisAspect(2),
              limits = ((0, grid.Ly), (-grid.Lz, 0)),
              title = uyz_title)

nothing #hide

wₙ = @lift time_series.w[$n]
uₙ = @lift time_series.u[$n]
Bₙ = @lift time_series.B[$n][1, 1, :]
Uₙ = @lift time_series.U[$n][1, 1, :]
Vₙ = @lift time_series.V[$n][1, 1, :]
wuₙ = @lift time_series.wu[$n][1, 1, :]
wvₙ = @lift time_series.wv[$n][1, 1, :]

k_index = searchsortedfirst(grid.zᵃᵃᶠ[:], -8)
wxyₙ = @lift interior(time_series.w[$n], :, :, k_index)
wyzₙ = @lift interior(time_series.w[$n], 1, :, :)
uyzₙ = @lift interior(time_series.u[$n], 1, :, :)

wlims = (-0.03, 0.03)
ulims = (-0.05, 0.05)
stokeslims = (-0.025, 0.025)

global ii = 1
vˢ_xvariation = Array{Float32}(undef, size(xu, 1))
∂z_vˢ_xvariation = Array{Float32}(undef, size(xu, 1))
∂y_vˢ_xvariation = Array{Float32}(undef, size(xu, 1))
∂x_vˢ_xvariation = Array{Float32}(undef, size(xu, 1))
wˢ_xvariation = Array{Float32}(undef, size(xu, 1))
∂z_wˢ_xvariation = Array{Float32}(undef, size(xu, 1))
∂y_wˢ_xvariation = Array{Float32}(undef, size(xu, 1))
∂x_wˢ_xvariation = Array{Float32}(undef, size(xu, 1))
vˢ_yvariation = Array{Float32}(undef, size(yu, 1))
wˢ_yvariation = Array{Float32}(undef, size(yu, 1))
vˢ_map = Array{Float32}(undef, size(xu, 1), size(yu, 1))
wˢ_map = Array{Float32}(undef, size(xu, 1), size(yu, 1))

while ii <= size(xu,1) 
    vˢ_xvariation[ii] = vˢ(xu[ii], 3*grid.Ly/8, -2, 0)  
    ∂z_vˢ_xvariation[ii] = ∂z_vˢ(xu[ii], 3*grid.Ly/8, -2, 0) 
    ∂y_vˢ_xvariation[ii] = ∂y_vˢ(xu[ii], 3*grid.Ly/8, -2, 0)
    ∂x_vˢ_xvariation[ii] = ∂x_vˢ(xu[ii], 3*grid.Ly/8, -2, 0)
    wˢ_xvariation[ii] = wˢ(xu[ii], 3*grid.Ly/8, -2, 0)  
    ∂z_wˢ_xvariation[ii] = ∂z_wˢ(xu[ii], 3*grid.Ly/8, -2, 0) 
    ∂y_wˢ_xvariation[ii] = ∂y_wˢ(xu[ii], 3*grid.Ly/8, -2, 0)
    ∂x_wˢ_xvariation[ii] = ∂x_wˢ(xu[ii], 3*grid.Ly/8, -2, 0) 

    global jj = 1
    while jj <= size(yu,1) 
        vˢ_yvariation[jj] = vˢ(stokes_jet_center + 0.5 * (stokes_jet_central_width + stokes_jet_edge_width), yu[jj], -2, 0)
        wˢ_yvariation[jj] = wˢ(stokes_jet_center + 0.5 * (stokes_jet_central_width + stokes_jet_edge_width), yu[jj], -2, 0)
        vˢ_map[ii,jj] = vˢ(xu[ii], yu[jj], -2, 0)
        wˢ_map[ii,jj] = wˢ(xu[ii], yu[jj], -2, 0)
        global jj += 1
    end

    global ii += 1
end 

lines!(ax_y_stokesu, vˢ_xvariation, xu; label = L"v^s")
lines!(ax_y_stokesu, ∂z_vˢ_xvariation, xu; label = L"\partial_z v^s")
lines!(ax_y_stokesu, ∂y_vˢ_xvariation * 100, xu; label = L"\partial_y v^s \times 100")
lines!(ax_y_stokesu, ∂x_vˢ_xvariation, xu; label = L"\partial_x v^s")
axislegend(ax_y_stokesu; position = :rt)

lines!(ax_y_stokesw, wˢ_xvariation * 100, xu; label = L"w^s \times 100")
lines!(ax_y_stokesw, ∂z_wˢ_xvariation * 100, xu; label = L"\partial_z w^s \times 100")
lines!(ax_y_stokesw, ∂y_wˢ_xvariation * 100, xu; label = L"\partial_y w^s \times 100")
lines!(ax_y_stokesw, ∂x_wˢ_xvariation * 100, xu; label = L"\partial_x w^s \times 100")
axislegend(ax_y_stokesw; position = :rt)

lines!(ax_x_stokes, vˢ_yvariation, yu; label = L"v^s")
lines!(ax_x_stokes, wˢ_yvariation*100, yu; label = L"w^s \times 100")
axislegend(ax_x_stokes; position = :lb)

#lines!(ax_B, Bₙ, zu)

lines!(ax_U, Uₙ, zu; label = L"\bar{u}")
lines!(ax_U, Vₙ, zu; label = L"\bar{v}")
axislegend(ax_U; position = :rb)

lines!(ax_fluxes, wuₙ, zw; label = L"mean $wu$")
lines!(ax_fluxes, wvₙ, zw; label = L"mean $wv$")
axislegend(ax_fluxes; position = :rb)

hm_xy_stokesu = heatmap!(ax_xy_stokesu, xw, yw, vˢ_map;
                         colorrange = stokeslims,
                         colormap = :balance)

Colorbar(fig[1, 3], hm_xy_stokesu; label = "m s⁻¹")

hm_xy_stokesw = heatmap!(ax_xy_stokesw, xw, yw, wˢ_map*100;
                         colorrange = stokeslims,
                         colormap = :balance)

hm_wxy = heatmap!(ax_wxy, xw, yw, wxyₙ;
                  colorrange = wlims,
                  colormap = :balance)

Colorbar(fig[2, 3], hm_wxy; label = "m s⁻¹")

hm_wyz = heatmap!(ax_wyz, yw, zw, wyzₙ;
                  colorrange = wlims,
                  colormap = :balance)

Colorbar(fig[3, 3], hm_wyz; label = "m s⁻¹")

ax_uyz = heatmap!(ax_uyz, yu, zu, uyzₙ;
                  colorrange = ulims,
                  colormap = :balance)

Colorbar(fig[4, 3], ax_uyz; label = "m s⁻¹")

current_figure() #hide
fig

# And, finally, we record a movie.

frames = 1:length(times)

record(fig, "Stokes_drift_y_jet.mp4", frames, framerate=8) do i
    n[] = i
end
nothing #hide

# ![](Stokes_drift_y_jet.mp4)
