# # Double Gyre
#
# This example simulates a double gyre following:
# https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode, znode, halo_size
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using CairoMakie
using Printf

const Lx = 4000kilometers # east-west extent [m]
const Ly = 6000kilometers # north-south extent [m]
const Lz = 1.8kilometers  # depth [m]

Δt₀ = 11minutes
stop_time = 365days

Nx = 160
Ny = 240
Nz = 50

σ = 1.2 # stretching factor
hyperbolically_spaced_faces(k) = - Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ))

grid = RectilinearGrid(CPU();
                       size = (Nx, Ny, Nz),
                          x = (-Lx/2, Lx/2),
                          y = (-Ly/2, Ly/2),
                          z = hyperbolically_spaced_faces,
                   topology = (Bounded, Bounded, Bounded))

# We plot vertical spacing versus depth to inspect the prescribed grid stretching:

Hx, Hy, Hz = halo_size(grid)
fig = Figure(resolution=(1200, 800))
ax = Axis(fig[1, 1], ylabel = "Depth (m)", xlabel = "Vertical spacing (m)")
lines!(ax, grid.Δzᵃᵃᶜ[1:grid.Nz], grid.zᵃᵃᶜ[1:grid.Nz])
scatter!(ax, grid.Δzᵃᵃᶜ[1:Nz], grid.zᵃᵃᶜ[1:Nz])

Δz_cpu = Array(grid.Δzᵃᵃᶜ.parent)[1+Hz:Nz+Hz]

save("double_gyre_grid_spacing.svg", fig)
nothing #hide

# ![](double_gyre_grid_spacing.svg)

α  = 2e-4 # [K⁻¹] thermal expansion coefficient
g  = 9.81 # [m s⁻²] gravitational constant
cᵖ = 3991 # [J K⁻¹ kg⁻¹] heat capacity for seawater
ρ₀ = 1028 # [kg m⁻³] reference density

parameters = (Ly = Ly,
              Lz = Lz,
               τ = 0.1 / ρ₀,     # surface kinematic wind stress [m² s⁻²]
               μ = 1 / 30days,   # bottom drag damping time-scale [s⁻¹]
              Δb = 30 * α * g,   # surface vertical buoyancy gradient [s⁻²]
               λ = 30days        # relaxation time scale [s]
              )

# ## Boundary conditions
#
# ### Wind stress

@inline u_stress(x, y, t, p) = - p.τ * cos(2π * y / p.Ly)
u_stress_bc = FluxBoundaryCondition(u_stress; parameters)

# ### Bottom drag
@inline u_drag(x, y, t, u, p) = - p.μ * p.Lz * u
@inline v_drag(x, y, t, v, p) = - p.μ * p.Lz * v

u_drag_bc = FluxBoundaryCondition(u_drag; field_dependencies=:u, parameters)
v_drag_bc = FluxBoundaryCondition(v_drag; field_dependencies=:v, parameters)

u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
v_bcs = FieldBoundaryConditions(                   bottom = v_drag_bc)


# ### Buoyancy relaxation

@inline buoyancy_relaxation(x, y, z, t, b, p) = - 1 / p.λ * (b - p.Δb * y / p.Ly)
Fb = Forcing(buoyancy_relaxation; parameters, field_dependencies=:b)

# ## Turbulence closure
horizontal_diffusive_closure = HorizontalScalarDiffusivity(ν=5000, κ=1000)
vertical_diffusive_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(),
                                                       ν=1e-2, κ=1e-5)

# ## Model building

model = HydrostaticFreeSurfaceModel(; grid,
                                    free_surface = ImplicitFreeSurface(),
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = BetaPlane(latitude=45),
                                    closure = (vertical_diffusive_closure, horizontal_diffusive_closure),
                                    tracers = :b,
                                    boundary_conditions = (u=u_bcs, v=v_bcs),
                                    forcing = (b=Fb,)
                                    )

# ## Initial conditions

bᵢ(x, y, z) = parameters.Δb * (1 + z / grid.Lz)

set!(model, b=bᵢ)

# ## Simulation setup

simulation = Simulation(model, Δt=Δt₀, stop_time=stop_time)

# add timestep wizard callback
max_Δt = 1 / 5model.coriolis.f₀
wizard = TimeStepWizard(cfl=0.15, max_change=1.1, max_Δt=max_Δt)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))

# add progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(200))


# ## Output

u, v, w = model.velocities
b = model.tracers.b

speed = Field(u^2 + v^2)
buoyancy_variance = Field(b^2)

outputs = merge(model.velocities, model.tracers, (speed=speed, b²=buoyancy_variance))

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(7days),
                                                      filename = "double_gyre",
                                                      indices = (:, :, model.grid.Nz),
                                                      overwrite_existing = true)

barotropic_u = Field(Average(model.velocities.u, dims=3))
barotropic_v = Field(Average(model.velocities.v, dims=3))

simulation.output_writers[:barotropic_velocities] =
    JLD2OutputWriter(model, (u=barotropic_u, v=barotropic_v),
                     schedule = AveragedTimeInterval(30days, window=10days),
                     filename = "double_gyre_circulation",
                     overwrite_existing = true)

run!(simulation)

# # A neat movie

# We open the JLD2 file, and extract the `grid` and the iterations we ended up saving at.

using JLD2

filename = "double_gyre.jld2"

u_timeseries = FieldTimeSeries(filename, "u"; architecture = CPU())
v_timeseries = FieldTimeSeries(filename, "v"; architecture = CPU())
s_timeseries = FieldTimeSeries(filename, "speed"; architecture = CPU())

times = u_timeseries.times

xu, yu, zu = nodes(u_timeseries[1])
xv, yv, zv = nodes(v_timeseries[1])
xs, ys, zs = nodes(s_timeseries[1])

xlims = (-grid.Lx/2 * 1e-3, grid.Lx/2 * 1e-3)
ylims = (-grid.Ly/2 * 1e-3, grid.Ly/2 * 1e-3)

xu_km, yu_km = xu * 1e-3, yu * 1e-3
xv_km, yv_km = xv * 1e-3, yv * 1e-3
xs_km, ys_km = xs * 1e-3, ys * 1e-3

# These utilities are handy for calculating nice contour intervals:

""" Returns colorbar levels equispaced from `(-clim, clim)` and encompassing the extrema of `c`. """
function divergent_levels(c, clim, nlevels=21)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    return ((-clim, clim), clim > cmax ? levels : levels = vcat([-cmax], levels, [cmax]))
end

""" Returns colorbar levels equispaced between `clims` and encompassing the extrema of `c`."""
function sequential_levels(c, clims, nlevels=20)
    levels = range(clims[1], stop=clims[2], length=nlevels)
    cmin, cmax = minimum(c), maximum(c)
    cmin < clims[1] && (levels = vcat([cmin], levels))
    cmax > clims[2] && (levels = vcat(levels, [cmax]))
    return clims, levels
end

# Finally, we're ready to animate.

@info "Making an animation from the saved data..."

using CairoMakie

n = Observable(1)

u = @lift interior(u_timeseries[$n], :, :)
v = @lift interior(v_timeseries[$n], :, :)
s = @lift interior(s_timeseries[$n], :, :)

extrema_reduction_factor = 0.8

ulims = extrema(u_timeseries.data) .* extrema_reduction_factor
vlims = extrema(v_timeseries.data) .* extrema_reduction_factor
slims = extrema(s_timeseries.data) .* extrema_reduction_factor

fig = Figure(resolution=(1500,1250)) 

title_u = @lift "Zonal Velocity after " *string(round(times[$n]/day, digits=1))*" days"
ax_u = Axis(fig[1,1]; xlabel = "x (km)", ylabel = "y (km)", aspect = 1.0, title = title_u)
hm_u = heatmap!(ax_u, xu_km, yu_km, u; colorrange = ulims, colormap = :balance)
Colorbar(fig[1,2], hm_u; label = "Zonal velocity (m s⁻¹)")

title_v = @lift "Meridional Velocity after " *string(round(times[$n]/day, digits=1))*" days"
ax_v = Axis(fig[1,3]; xlabel = "x (km)", ylabel = "y (km)", aspect = 1.0, title = title_v)
hm_v = heatmap!(ax_v, xv_km, yv_km, v; colorrange = vlims, colormap = :balance)
Colorbar(fig[1,4], hm_v; label = "Meridional velocity (m s⁻¹)")

title_s = @lift "Speed after " *string(round(times[$n]/day, digits=1))*" days"
ax_s = Axis(fig[2,1]; xlabel = "x (km)", ylabel = "y (km)", aspect = 1.0, title = title_s)
hm_s = heatmap!(ax_s, xs_km, ys_km, v; colorrange = slims, colormap = :balance)
Colorbar(fig[2,2], hm_s; label = "Speed (m s⁻¹)")

frames = 1:length(times)

record(fig, filename * ".mp4", frames, framerate=8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end

nothing #hide

# Plot the barotropic circulation

filename_barotropic = "double_gyre_circulation.jld2"

U_timeseries = FieldTimeSeries(filename_barotropic, "u"; grid = grid)
V_timeseries = FieldTimeSeries(filename_barotropic, "v"; grid = grid)

# Average for the last `n_years`

n_years = 5

using Statistics

U = mean(interior(U_timeseries)[:, :, :, end:end], dims=4)[:, :, 1, 1]
V = mean(interior(V_timeseries)[:, :, :, end:end], dims=4)[:, :, 1, 1]

U_plot = contourf(xu_km, yu_km, U, xlims=xlims, ylims=ylims,
                  linewidth=0, color=:balance, aspectratio=:equal)

V_plot = contourf(xv_km, yv_km, V, xlims=xlims, ylims=ylims,
                  linewidth=0, color=:balance, aspectratio=:equal)

Ψ = cumsum(V, dims=1) * grid.Δxᶜᵃᵃ * grid.Lz * 1e-6

Ψlims, Ψlevels = divergent_levels(Ψ, 45)

Ψ_plot = contourf(xv_km, yv_km, Ψ, xlims=xlims, ylims=ylims, levels = Ψlevels, clims = Ψlims,
                  linewidth=0, color=:balance, aspectratio=:equal)

plot(U_plot, V_plot, Ψ_plot, layout = Plots.grid(1, 3), size=(1200, 400),
     title=["Depth- and time-averaged \$ u \$" "Depth- and time-averaged \$ v \$" "Barotropic streamfunction"])

savefig("double_gyre_circulation.png"); nothing # hide

![](assets/double_gyre_circulation.svg)
