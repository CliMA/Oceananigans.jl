# # Double Gyre
#
# This example simulates a double gyre following:
# https://mitgcm.readthedocs.io/en/latest/examples/baroclinic_gyre/baroclinic_gyre.html

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: on_architecture
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using Oceananigans.Coriolis: ActiveCellEnstrophyConservingScheme
using Oceananigans.Operators
using JLD2
using CairoMakie
using Statistics
using Printf

const λ_west = -30 # [°] longitude of west boundary
const λ_east = +30 # [°] longitude of east boundary
const φ_south = 15 # [°] latitude of south boundary
const φ_north = 75 # [°] latitude of north boundary

φ₀ = 0.5(φ_south + φ_north) # [°] latitude of the center of the domain

Lλ = λ_east - λ_west   # [°] longitude extent of the domain
Lφ = φ_north - φ_south # [°] latitude extent of the domain

const Lz = 1.8kilometers # depth [m]

Δt₀ = 11minutes
stop_time = 365days

Nλ = 160
Nφ = 240
Nz = 50

σ = 1.2 # stretching factor
hyperbolically_spaced_faces(k) = - Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ))

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nλ, Nφ, Nz),
                        longitude = (λ_west, λ_east),
                         latitude = (φ_south, φ_north),
                                z = hyperbolically_spaced_faces,
                         topology = (Bounded, Bounded, Bounded),
                             halo = (4, 4, 4))

# We plot vertical spacing versus depth to inspect the prescribed grid stretching.

grid_CPU = on_architecture(CPU(), grid)
zspacings_CPU = zspacings(grid_CPU, Center(), Center(), Center())
znodes_CPU = znodes(grid_CPU, Center(), Center(), Center())

fig = Figure(resolution = (750, 750))
ax = Axis(fig[1, 1], xlabel = "Vertical spacing (m)", ylabel = "Depth (m)", xlabelsize = 22.5, ylabelsize = 22.5, 
           xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1, 
           title = "Variation of Vertical Spacing with Depth", titlesize = 27.5, titlegap = 15, titlefont = :bold,
           xminorgridvisible = true, yminorgridvisible = true)
scatterlines!(ax, zspacings_CPU, znodes_CPU, linewidth = 2.0, color = :black, marker = :circle, markersize = 12)

save("double_gyre_grid_spacing.pdf", fig)

nothing # hide

# ![](double_gyre_grid_spacing.svg)

α  = 2e-4 # [K⁻¹] thermal expansion coefficient
g  = 9.81 # [m s⁻²] gravitational constant
cᵖ = 3991 # [J K⁻¹ kg⁻¹] heat capacity for seawater
ρ₀ = 1028 # [kg m⁻³] reference density

Δzₛ = minimum_zspacing(grid) # vertical spacing at the surface [m] 

parameters = (Lφ = Lφ,
              Lz = Lz,
              φ₀ = φ₀,         # latitude of the center of the domain [°]
               τ = 0.1 / ρ₀,   # surface kinematic wind stress [m² s⁻²]
               μ = 0.001,      # bottom drag damping parameter [ms⁻¹]
              Δb = 30 * α * g, # surface vertical buoyancy gradient [s⁻²]
       timescale = 30days,     # relaxation time scale [s]  
              vˢ = Δzₛ/30days) # buoyancy pumping velocity [ms⁻¹]

# ## Boundary conditions
#
# ### Wind stress
@inline u_stress(λ, φ, t, p) = p.τ * sin(2π * (φ - p.φ₀) / p.Lφ)

# ### Buoyancy relaxation
@inline surface_buoyancy(φ, p)             = p.Δb * (φ - p.φ₀) / p.Lφ
@inline buoyancy_relaxation(λ, φ, t, b, p) = p.vˢ * (b - surface_buoyancy(φ, p))

# ### Plotting surface forcing functions
φ_array = range(φ_south, φ_north, length = 100)
u_stress_array = u_stress.(0, φ_array, 0, Ref(parameters))
surface_buoyancy_array = surface_buoyancy.(φ_array, Ref(parameters))

fig = Figure(resolution = (750, 750))
ax  = Axis(fig[1, 1], xlabel = "Buoyancy Profile", ylabel = "Latitude (Degree)", xlabelsize = 22.5, ylabelsize = 22.5, 
           xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1, 
           title = "Surface Buoyancy Forcing", titlesize = 27.5, titlegap = 15, titlefont = :bold, 
           xminorgridvisible = true, yminorgridvisible = true)
scatterlines!(ax, surface_buoyancy_array, φ_array, linewidth = 2.0, color = :black, marker = :circle, markersize = 12)

save("SurfaceBuoyancyForcing.pdf", fig)

fig = Figure(resolution = (750, 750))
ax = Axis(fig[1, 1], xlabel = "Wind Stress Profile", ylabel = "Latitude (Degree)", xlabelsize = 22.5, ylabelsize = 22.5, 
          xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, aspect = 1, 
          title = "Surface Wind Stress", titlesize = 27.5, titlegap = 15, titlefont = :bold, xminorgridvisible = true, 
          yminorgridvisible = true)
scatterlines!(ax, u_stress_array, φ_array, linewidth = 2.0, color = :black, marker = :circle, markersize = 12)

save("SurfaceWindStress.pdf", fig)

# ### Bottom drag
@inline u_drag(λ, φ, t, u, p) = - p.μ * u
@inline v_drag(λ, φ, t, v, p) = - p.μ * v

u_stress_bc = FluxBoundaryCondition(u_stress; parameters)
b_relax_bc  = FluxBoundaryCondition(buoyancy_relaxation; field_dependencies = :b, parameters)
u_drag_bc   = FluxBoundaryCondition(u_drag; field_dependencies = :u, parameters)
v_drag_bc   = FluxBoundaryCondition(v_drag; field_dependencies = :v, parameters)

u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
v_bcs = FieldBoundaryConditions(                   bottom = v_drag_bc)
b_bcs = FieldBoundaryConditions(top = b_relax_bc)

# ## Turbulence closure
boundary_layer_closure     = RiBasedVerticalDiffusivity()
vertical_diffusive_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ = 3e-5, ν = 5e-4)

closures = (boundary_layer_closure, vertical_diffusive_closure)

Δt_max = 10minutes
gravitational_acceleration = 9.81 # [m s⁻²]
gravity_wave_speed = sqrt(gravitational_acceleration*grid.Lz) # [m s⁻¹]
Δx = minimum_xspacing(grid) # [m]
Δy = minimum_yspacing(grid) # [m]
Δh = sqrt(1/(1/Δx^2 + 1/Δy^2)) # [m]
Δt_barotropic = 0.7Δh/gravity_wave_speed
substeps = ceil(Int, 2Δt_max / Δt_barotropic)

# ## Model building
model = HydrostaticFreeSurfaceModel(; grid,
                                    free_surface = SplitExplicitFreeSurface(; gravitational_acceleration, substeps),
                                    momentum_advection = VectorInvariant(vorticity_scheme = WENO(), 
                                                                         divergence_scheme = WENO(), 
                                                                         vertical_scheme = WENO()),
                                    tracer_advection = WENO(),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = HydrostaticSphericalCoriolis(scheme = ActiveCellEnstrophyConservingScheme()),
                                    closure  = closures,
                                    tracers  = :b,
                                    boundary_conditions = (u = u_bcs, v = v_bcs, b = b_bcs))

# ## Initial conditions

bᵢ(λ, φ, z) = parameters.Δb * z / grid.Lz

set!(model, b = bᵢ)

# ## Simulation setup

simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

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

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(50))

run_simulation = true

# ## Output

if run_simulation

    u, v, w = model.velocities
    b = model.tracers.b

    speed = Field(u^2 + v^2)
    buoyancy_variance = Field(b^2)

    outputs = merge(model.velocities, model.tracers, (speed = speed, b² = buoyancy_variance))

    simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                          schedule = TimeInterval(7days),
                                                          filename = "double_gyre",
                                                          indices = (:, :, model.grid.Nz),
                                                          overwrite_existing = true)

    barotropic_u = Field(Average(model.velocities.u, dims = 3))
    barotropic_v = Field(Average(model.velocities.v, dims = 3))

    simulation.output_writers[:barotropic_velocities] =
        JLD2OutputWriter(model, (u = barotropic_u, v = barotropic_v), 
                         schedule = AveragedTimeInterval(30days, window = 10days), 
                         filename = "double_gyre_circulation",
                         overwrite_existing = true)

    run!(simulation)
    
end

# # A neat movie

# We open the JLD2 file, and extract the `grid` and the iterations we ended up saving at.

filename = "double_gyre.jld2"

u_timeseries = FieldTimeSeries(filename, "u"; architecture = CPU())
v_timeseries = FieldTimeSeries(filename, "v"; architecture = CPU())
s_timeseries = FieldTimeSeries(filename, "speed"; architecture = CPU())

times = u_timeseries.times

λᵤ, φᵤ, zᵤ = nodes(u_timeseries[1])
λᵥ, φᵥ, zᵥ = nodes(v_timeseries[1])
λₛ, φₛ, zₛ = nodes(s_timeseries[1])

# Finally, we're ready to animate.

@info "Making an animation from the saved data..."

n = Observable(1)

u = @lift interior(u_timeseries[$n], :, :)
v = @lift interior(v_timeseries[$n], :, :)
s = @lift interior(s_timeseries[$n], :, :)

extrema_reduction_factor = 0.8

ulims = extrema(u_timeseries.data) .* extrema_reduction_factor
vlims = extrema(v_timeseries.data) .* extrema_reduction_factor
slims = extrema(s_timeseries.data) .* extrema_reduction_factor

fig = Figure(resolution = (1650, 1250))

title_u = @lift "Zonal Velocity after " *string(round(times[$n]/day, digits = 1))*" days"
ax_u = Axis(fig[1:2,1]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)", xlabelsize = 22.5, 
            ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
            aspect = 1.0, title = title_u, titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_u = heatmap!(ax_u, λᵤ, φᵤ, u; colorrange = ulims, colormap = :balance)
Colorbar(fig[1:2,2], hm_u; label = "Zonal velocity (m s⁻¹)", labelsize = 22.5, labelpadding = 10.0, ticksize = 17.5)

title_v = @lift "Meridional Velocity after " *string(round(times[$n]/day, digits = 1))*" days"
ax_v = Axis(fig[3:4,1]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)", xlabelsize = 22.5, 
            ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
            aspect = 1.0, title = title_v, titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_v = heatmap!(ax_v, λᵥ, φᵥ, v; colorrange = vlims, colormap = :balance)
Colorbar(fig[3:4,2], hm_v; label = "Meridional velocity (m s⁻¹)", labelsize = 22.5, labelpadding = 10.0, 
         ticksize = 17.5)

title_s = @lift "Speed after " *string(round(times[$n]/day, digits = 1))*" days"
ax_s = Axis(fig[2:3,3]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)", xlabelsize = 22.5, 
            ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
            aspect = 1.0, title = title_s, titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_s = heatmap!(ax_s, λₛ, φₛ, s; colorrange = slims, colormap = :balance)
Colorbar(fig[2:3,4], hm_s; label = "Speed (m s⁻¹)", labelsize = 22.5, labelpadding = 10.0, ticksize = 17.5)

frames = 1:length(times)

CairoMakie.record(fig, filename * ".mp4", frames, framerate = 8) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end

nothing # hide

# Plot the barotropic circulation

filename_barotropic = "double_gyre_circulation.jld2"

U_timeseries = FieldTimeSeries(filename_barotropic, "u"; grid = grid, architecture = CPU())
V_timeseries = FieldTimeSeries(filename_barotropic, "v"; grid = grid, architecture = CPU())

# Average for the last `n_years`

n_years = 5

U = mean(interior(U_timeseries)[:, :, :, end:end], dims = 4)[:, :, 1, 1]
V = mean(interior(V_timeseries)[:, :, :, end:end], dims = 4)[:, :, 1, 1]

fig = Figure(resolution = (1650, 1250))

title_U = "Depth- and Time-Averaged Zonal Velocity"
ax_U = Axis(fig[1:2,1]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)", xlabelsize = 22.5, 
            ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
            aspect = 1.0, title = title_U, titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_U = heatmap!(ax_U, λᵤ, φᵤ, U; colorrange = ulims, colormap = :balance)
Colorbar(fig[1:2,2], hm_U, labelsize = 22.5, labelpadding = 10.0, ticksize = 17.5)

title_V = "Depth- and Time-Averaged Meridional Velocity"
ax_V = Axis(fig[3:4,1]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)", xlabelsize = 22.5, 
            ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
            aspect = 1.0, title = title_V, titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_V = heatmap!(ax_V, λᵥ, φᵥ, V; colorrange = vlims, colormap = :balance)
Colorbar(fig[3:4,2], hm_V, labelsize = 22.5, labelpadding = 10.0, ticksize = 17.5)

yspacings_CPU = yspacings(grid_CPU, Center(), Center())
Ψ = -cumsum(U, dims = 2) * yspacings_CPU * grid.Lz * 1e-6
Ψlims = extrema(Ψ) .* extrema_reduction_factor

title_Ψ = "Barotropic Streamfunction"
ax_Ψ = Axis(fig[2:3,3]; xlabel = "Longitude (Degree)", ylabel = "Latitude (Degree)", xlabelsize = 22.5, 
            ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, xlabelpadding = 10, ylabelpadding = 10, 
            aspect = 1.0, title = title_Ψ, titlesize = 27.5, titlegap = 15, titlefont = :bold)
hm_Ψ = heatmap!(ax_Ψ, λᵤ, φᵤ, Ψ; colorrange = Ψlims, colormap = :balance)
Colorbar(fig[2:3,4], hm_Ψ, labelsize = 22.5, labelpadding = 10.0, ticksize = 17.5)

save("double_gyre_circulation.pdf", fig)

# ![](assets/double_gyre_circulation.svg)