using Statistics
using JLD2
using Printf
using GLMakie
using CUDA
using Oceananigans
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.Architectures: arch_array
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity
using Oceananigans.Units
using Oceananigans.Operators: Δzᵃᵃᶜ

#####
##### Grid
#####

latitude = (-84.375, 84.375)
Δφ = latitude[2] - latitude[1]

# 2.8125 degree resolution
Nx = 128
Ny = 60
Nz = 18

arch = GPU()
reference_density = 1035

bathymetry_path = "bathy_128x60var4.bin"
east_west_stress_path = "off_TAUXvar1.bin"
north_south_stress_path = "off_TAUY.bin"
sea_surface_temperature_path="sst25_128x60x12.bin"

bytes = sizeof(Float32) * Nx * Ny
bathymetry = reshape(bswap.(reinterpret(Float32, read(bathymetry_path, bytes))), (Nx, Ny))
τˣ = - reshape(bswap.(reinterpret(Float32, read(east_west_stress_path, bytes))), (Nx, Ny)) ./ reference_density
τʸ = - reshape(bswap.(reinterpret(Float32, read(north_south_stress_path, bytes))), (Nx, Ny)) ./ reference_density
target_sea_surface_temperature = reshape(bswap.(reinterpret(Float32, read(sea_surface_temperature_path, bytes))), (Nx, Ny))

bathymetry = arch_array(arch, bathymetry)
τˣ = arch_array(arch, τˣ)
τʸ = arch_array(arch, τʸ)
target_sea_surface_temperature = arch_array(arch, target_sea_surface_temperature)

H = 3600.0
bathymetry = - H .* (bathymetry .< -10)
# H = - minimum(bathymetry)

# A spherical domain
@show underlying_grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, Nz),
                                                     longitude = (-180, 180),
                                                     latitude = latitude,
                                                     halo = (3, 3, 3),
                                                     z = (-H, 0))

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

#####
##### Physics and model setup
#####

equatorial_Δx = grid.radius * deg2rad(grid.Δλ)
diffusive_time_scale = 120days

@show νh₂ = 1e-3 * equatorial_Δx^2 / diffusive_time_scale
@show νh₄ = 1e-5 * equatorial_Δx^4 / diffusive_time_scale
@show νz = 1e-3
@show κh = 1e+3
@show κz = 1e-4

background_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=νh₂, νz=νz, κh=κh, κz=κz)

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0)

#####
##### Boundary conditions / constant-in-time surface forcing
#####

Δz_top = CUDA.@allowscalar Δzᵃᵃᶜ(1, 1, grid.Nz, grid)
Δz_bottom = CUDA.@allowscalar Δzᵃᵃᶜ(1, 1, 1, grid)

@inline surface_temperature_relaxation(i, j, grid, clock, fields, p) = @inbounds p.λ * (fields.T[i, j, grid.Nz] - p.T★[i, j])

T_surface_relaxation_bc = FluxBoundaryCondition(surface_temperature_relaxation,
                                                discrete_form = true,
                                                parameters = (λ = Δz_top/3days, T★ = target_sea_surface_temperature))

u_wind_stress_bc = FluxBoundaryCondition(τˣ)
v_wind_stress_bc = FluxBoundaryCondition(τʸ)

@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

# Linear bottom drag:
μ = 1 / 10days * Δz_bottom

u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag, discrete_form = true, parameters = μ)
v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag, discrete_form = true, parameters = μ)

u_bcs = FieldBoundaryConditions(top = u_wind_stress_bc, bottom = u_bottom_drag_bc)
v_bcs = FieldBoundaryConditions(top = v_wind_stress_bc, bottom = v_bottom_drag_bc)
T_bcs = FieldBoundaryConditions(top = T_surface_relaxation_bc)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    architecture = arch,
                                    free_surface = ExplicitFreeSurface(),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs),
                                    buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4, β=0.0)),
                                    closure = nothing, # (background_diffusivity, convective_adjustment),
                                    tracers = (:T, :S))

#####
##### Initial condition:
#####

u, v, w = model.velocities
η = model.free_surface.η
T = model.tracers.T
S = model.tracers.S
T .= 5
S .= 30

#####
##### Simulation setup
#####

# Time-scale for gravity wave propagation across the smallest grid cell
g = model.free_surface.gravitational_acceleration
gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

minimum_Δx = abs(grid.radius * cosd(maximum(abs, grid.φᵃᶜᵃ)) * deg2rad(grid.Δλ))
minimum_Δy = abs(grid.radius * deg2rad(grid.Δφ))
wave_propagation_time_scale = min(minimum_Δx, minimum_Δy) / gravity_wave_speed
@show Δt = 0.2 * minimum_Δx / gravity_wave_speed

@info """
    Minimum wave propagation time scale: $(prettytime(wave_propagation_time_scale))
    Time step: $(prettytime(Δt))
"""

start_time = [time_ns()]

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    η = model.free_surface.η

    @info @sprintf("Time: %s, iteration: %d, max(|η|): %.2e m, wall time: %s",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   maximum(abs, η),
                   prettytime(wall_time))

    start_time[1] = time_ns()

    return nothing
end

simulation = Simulation(model, Δt = Δt, stop_iteration = 10000, iteration_interval = 100, progress = progress)

output_fields = merge(model.velocities, model.tracers, (; η=model.free_surface.η))
output_prefix = "global_lat_lon_$(grid.Nx)_$(grid.Ny)_$(grid.Nz)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(10day),
                                                      prefix = output_prefix,
                                                      force = true)

# Let's goo!
@printf "Running with Δt = %s" prettytime(simulation.Δt)
run!(simulation)

#####
##### Visualization
#####

η_cpu = Array(interior(model.free_surface.η))[:, :, 1]
T_cpu = Array(interior(model.tracers.T))[:, :, end]
h_cpu = Array(bathymetry)

max_η = maximum(abs, η_cpu)

max_T = maximum(T_cpu)
min_T = minimum(T_cpu)

fig = Figure(resolution = (600, 900))

ax_b = Axis(fig[1, 1], title="Bathymetry (m)")
hm_b = heatmap!(ax_b, h_cpu)
cb_b = Colorbar(fig[1, 2], hm_b)

ax_η = Axis(fig[2, 1], title="Free surface displacement (m)")
hm_η = heatmap!(ax_η, η_cpu, colorrange=(-max_η, max_η), colormap=:balance)
cb_η = Colorbar(fig[2, 2], hm_η)

ax_T = Axis(fig[3, 1], title="Sea surface temperature (ᵒC)")
hm_T = heatmap!(ax_T, T_cpu, colorrange=(min_T, max_T), colormap=:thermal)
cb_T = Colorbar(fig[3, 2], hm_T)

display(fig)
