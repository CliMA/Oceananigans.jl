using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode, xnode
using Oceananigans.BoundaryConditions: FlatherBoundaryCondition, RadiationBoundaryCondition

Nx, Ny, Nz = 256, 64, 128
H, L = 2kilometers, 1000kilometers

z = MutableVerticalDiscretization((-H, 0))

underlying_grid = RectilinearGrid(CPU();
                                  size = (Nx, Ny, Nz), halo = (5, 5, 5),
                                  x = (-L, L), y = (-L/4, L/4), z = (-H, 0),
                                  topology = (Bounded, Bounded, Bounded))

h₀ = 250meters
width = 20kilometers
hill(x) = h₀ * exp(-x^2 / 2width^2)
slope(x) = -H + H * exp(-((x + L)^2) / (2 * (200kilometers)^2))
bottom(x, y) = (slope(x) + hill(x)) * ((-L/4+50kilometers)<y<(L/4-50kilometers))

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

coriolis = FPlane(latitude = -45)

T₂ = 12.421hours
ω₂ = 2π / T₂ # radians/sec
ϵ = 0.1 # excursion parameter
U₂ = ϵ * ω₂ * width
A₂ = U₂ * (ω₂^2 - coriolis.f^2) / ω₂

Nᵢ² = 1e-4  # [s⁻²] initial buoyancy frequency / stratification

@inline tidal_forcing(y, z, t, p) = p.U₂ * sin(p.ω₂ * t)

u_east_bc = RadiationBoundaryCondition(tidal_forcing; parameters=(; U₂, ω₂))
#u_west_bc = RadiationBoundaryCondition(tidal_forcing; parameters=(; U₂, ω₂))
u_bcs     = FieldBoundaryConditions(east = u_east_bc)# west = u_west_bc

@inline barotropic_tidal_forcing(j, k, grid, clock, fields, p) = (p.U₂ * sin(p.ω₂ * clock.time) * 2kilometers, 0.0)

#U_west_bc = FlatherBoundaryCondition(barotropic_tidal_forcing; discrete_form = true, parameters=(; U₂, ω₂))
U_east_bc = FlatherBoundaryCondition(barotropic_tidal_forcing; discrete_form = true, parameters=(; U₂, ω₂))
U_bcs     = FieldBoundaryConditions(grid, (Face(), Center(), nothing); east = U_east_bc) # west = U_west_bc

@inline b_background(y, z, t, p) = p.Nᵢ² * z
b_east_bc = RadiationBoundaryCondition(b_background; parameters = (; Nᵢ²))
#b_west_bc = RadiationBoundaryCondition(b_background; parameters = Nᵢ²)
b_bcs     = FieldBoundaryConditions(east = b_east_bc) # west = b_west_bc) 

@inline function sponge_mask(x, p)
    d_east = p.Lx - x
    d = d_east
    return ifelse(d < p.W, cospi(d / (2 * p.W))^2, zero(d))
end

@inline function u_sponge_forcing(i, j, k, grid, clock, fields, p)
    x = xnode(i, grid, Face())
    mask = sponge_mask(x, p)
    ut = tidal_forcing(1, 1, clock.time, p)
    return -p.rate * mask * (@inbounds fields.u[i, j, k] - ut)
end

@inline b_bg(z, p) = p.N² * z

@inline function b_sponge_forcing(i, j, k, grid, clock, fields, p)
    x = xnode(i, grid, Center())
    z = znode(k, grid, Center())
    mask = sponge_mask(x, p)
    bt = b_bg(z, p)
    return -p.rate * mask * (@inbounds fields.b[i, j, k] - bt)
end

sponge_params = (; Lx = L, W = 200kilometers, rate = 1 / 30minutes, U₂, ω₂, N² = Nᵢ²)

u_sponge = Forcing(u_sponge_forcing, discrete_form = true, parameters = sponge_params)
b_sponge = Forcing(b_sponge_forcing, discrete_form = true, parameters = sponge_params)

free_surface = SplitExplicitFreeSurface(grid; substeps = 30, extend_halos = false)

model = HydrostaticFreeSurfaceModel(grid; 
                                    coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    free_surface,
                                    timestepper = :SplitRungeKutta3,
                                    forcing = (; u = u_sponge, b = b_sponge),
                                    boundary_conditions = (; u = u_bcs, U = U_bcs, b = b_bcs))

# Initialize with the tidal flow and a linear stratification.
bᵢ(x, y, z) = Nᵢ² * z
gaussian(x, σ) = exp(-x^2 / 2σ^2)
ηᵢ(x, y, z) = gaussian(x, 100kilometers) # sin(π * x / L) # 
set!(model, b=bᵢ) #, η=ηᵢ)

# ## Simulation

Δt = 10minutes
stop_time = 10days
simulation = Simulation(model; Δt, stop_time)

# We add a callback to print a message about how the simulation is going.

using Printf

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

add_callback!(simulation, progress, name=:progress, IterationInterval(100))
nothing #hide

fields_to_check=(:u,:v,:w)
function nan_check(sim, parameters)        
    for field in parameters.fields_to_check
        if(isnan(sum(sim.model.velocities[field])))
            sim.running=false
            @info  @sprintf("Error: Simulation stopping. NaN found in %s...",field)
        end
    end
    return nothing
end
simulation.callbacks[:nan_checker] = Callback(nan_check, IterationInterval(1),
                                              parameters=(;fields_to_check=fields_to_check))

# ## Diagnostics/Output
#
# We save the deviation of ``u`` from its domain average, ``w``, buoyancy ``b``,
# the stratification ``N^2``, and the free surface ``\eta``.

b = model.tracers.b
u, v, w = model.velocities
U = Field(Average(u))
u′ = u - U
N² = ∂z(b)
η = model.free_surface.displacement
Ub = model.free_surface.barotropic_velocities.U

filename = "InternalTideTests/internal_tide_open_boundaries"

simulation.output_writers[:fields] = JLD2Writer(model, (; u, u′, w, b, N², η, Ub); filename,
                                                schedule = TimeInterval(1hour),
                                                overwrite_existing = true)

# We are ready -- let's run!

run!(simulation)

# ## Load output

saved_output_filename = filename * ".jld2"

 u_t = FieldTimeSeries(saved_output_filename, "u")
u′_t = FieldTimeSeries(saved_output_filename, "u′")
 w_t = FieldTimeSeries(saved_output_filename, "w")
N²_t = FieldTimeSeries(saved_output_filename, "N²")
 η_t = FieldTimeSeries(saved_output_filename, "η")
 U_t = FieldTimeSeries(saved_output_filename, "Ub")

umax = maximum(abs, u_t)
wmax = maximum(abs, w_t)

times = u′_t.times
nothing #hide

# ## Visualize
#
# We visualize the baroclinic velocity perturbation, vertical velocity,
# stratification, and free surface displacement. With open boundaries, the
# internal waves should radiate cleanly out of the domain without reflections.

using CairoMakie

n = Observable(1)

title = @lift @sprintf("t = %1.2f days = %1.2f T₂",
                       round(times[$n] / day, digits=2) , round(times[$n] / T₂, digits=2))

midIdx = Int(Ny/2)
uₙ = @lift interior(u_t[$n], :,midIdx , :)
uₙ_surface = @lift interior(u_t[$n], :, :, Nz)
 wₙ = @lift interior( w_t[$n], :, midIdx, :)
wₙ_surface = @lift interior( w_t[$n], :, :, Nz)
N²ₙ = @lift interior(N²_t[$n], :, midIdx, :)
 ηₙ = @lift interior( η_t[$n], :, midIdx, :)
ηₙ_surface = @lift interior( η_t[$n], :, :, 1)
 Uₙ = @lift interior( U_t[$n], :, midIdx, :)
u1ₙ = @lift interior( u_t[$n], 1, midIdx, :)
ueₙ = @lift interior( u_t[$n], grid.Nx+1, midIdx, :)

axis_kwargs = (; titlesize = 20)

fig = Figure(size = (900, 1100))

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

ax_u = Axis(fig[2, 1:2]; title = "Cross section u-velocity", axis_kwargs...)
hm_u = heatmap!(ax_u, uₙ; nan_color=:gray, colorrange=(-umax, umax), colormap=:balance)
Colorbar(fig[2, 3], hm_u, label = "m s⁻¹")

ax_u_s = Axis(fig[2, 4:5]; title = "Surface u-velocity", axis_kwargs...)
hm_u_s = heatmap!(ax_u_s, uₙ_surface; nan_color=:gray, colorrange=(-umax, umax), colormap=:balance)
Colorbar(fig[2, 6], hm_u_s, label = "m s⁻¹")

ax_w = Axis(fig[3, 1:2]; title = "Cross section w-velocity", axis_kwargs...)
hm_w = heatmap!(ax_w, wₙ; nan_color=:gray, colorrange=(-wmax, wmax), colormap=:balance)
Colorbar(fig[3, 3], hm_w, label = "m s⁻¹")

ax_w_s = Axis(fig[3, 4:5]; title = "Surface w-velocity", axis_kwargs...)
hm_w_s = heatmap!(ax_w_s, wₙ_surface; nan_color=:gray, colorrange=(-wmax, wmax), colormap=:balance)
Colorbar(fig[3, 6], hm_w_s, label = "m s⁻¹")

ax_N² = Axis(fig[4, 1:2]; title = "stratification N²", axis_kwargs...)
hm_N² = heatmap!(ax_N², N²ₙ; nan_color=:gray, colorrange=(0.9Nᵢ², 1.1Nᵢ²), colormap=:magma)
Colorbar(fig[4, 3], hm_N², label = "s⁻²")

ax_N²_s = Axis(fig[4, 4:5]; title = "Surface stratification", axis_kwargs...)
hm_N²_s = heatmap!(ax_N²_s, wₙ_surface; nan_color=:gray, colorrange=(0.9Nᵢ², 1.1Nᵢ²), colormap=:magma)
Colorbar(fig[4, 6], hm_N²_s, label = "s⁻²")

ax_η = Axis(fig[5, 1]; title = "cross section free surface η",
            xlabel = "x [km]",
            titlesize = 20,
            ylabel = "η [m]")
ηₙ_line = @lift interior(η_t[$n], :, 1, 1)
lines!(ax_η, ηₙ_line, color=:dodgerblue)
ylims!(ax_η, -4.0, 4.0)

ax_η_s = Axis(fig[5, 3:4]; title = "Free Surface", axis_kwargs...)
hm_η_s = heatmap!(ax_η_s, ηₙ_surface; nan_color=:gray, colorrange=(-4.0, 4.0), colormap=:delta)
Colorbar(fig[5, 5], hm_η_s, label = "m")

fig

# Finally, we can record a movie.

@info "Making an animation from saved data..."

frames = 1:length(times)

record(fig, filename * ".mp4", frames, framerate=16) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    n[] = i
end
nothing #hide

# ![](internal_tide_open_boundaries.mp4)
