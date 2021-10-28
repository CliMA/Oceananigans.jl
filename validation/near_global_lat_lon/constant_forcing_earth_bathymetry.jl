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

#####
##### Load forcing files roughly from CORE2 paper
##### (Probably https://data1.gfdl.noaa.gov/nomads/forms/core/COREv2.html)
#####

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
target_sea_surface_temperature = T★ = arch_array(arch, target_sea_surface_temperature)

H = 3600.0
# bathymetry = - H .* (bathymetry .< -10)
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

νh = 1e+5
νz = 1e-2
κh = 1e+3
κz = 1e-4

background_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=νh, νz=νz, κh=κh, κz=κz)
convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0)

#####
##### Boundary conditions / constant-in-time surface forcing
#####

Δz_top = CUDA.@allowscalar Δzᵃᵃᶜ(1, 1, grid.Nz, grid.grid)
Δz_bottom = CUDA.@allowscalar Δzᵃᵃᶜ(1, 1, 1, grid.grid)

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
                                    #free_surface = ExplicitFreeSurface(),
                                    free_surface = ImplicitFreeSurface(maximum_iterations=10),
                                    #free_surface = ImplicitFreeSurface(),
                                    momentum_advection = VectorInvariant(),
                                    tracer_advection = WENO5(),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs),
                                    buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4, β=0.0)),
                                    tracers = (:T, :S),
                                    closure = (background_diffusivity, convective_adjustment))

#=
#####
##### Visualize the setup
#####

setup_fig = Figure(resolution = (1200, 900))

free_surface = model.free_surface
∫Ax = free_surface.implicit_step_solver.vertically_integrated_lateral_areas.xᶠᶜᶜ
∫Ay = free_surface.implicit_step_solver.vertically_integrated_lateral_areas.yᶜᶠᶜ

h_cpu = Array(bathymetry)
τˣ_cpu = Array(τˣ)
τʸ_cpu = Array(τʸ)
T★_cpu = Array(T★)
∫Ax_cpu = Array(parent(∫Ax))[:, :, 1]
∫Ay_cpu = Array(parent(∫Ay))[:, :, 1]

ax_b = Axis(setup_fig[1, 1], title="Bathymetry (m)")
hm_b = heatmap!(ax_b, h_cpu)
cb_b = Colorbar(setup_fig[1, 2], hm_b)

ax_T★ = Axis(setup_fig[1, 3], title="Target sea surface temperature (ᵒC)")
hm_T★ = heatmap!(ax_T★, T★_cpu)
cb_T★ = Colorbar(setup_fig[1, 4], hm_T★)

ax_x = Axis(setup_fig[2, 1], title="Vertically-integrated x-face area (m³)")
hm_x = heatmap!(ax_x, ∫Ax_cpu)
cb_x = Colorbar(setup_fig[2, 2], hm_x)

ax_y = Axis(setup_fig[2, 3], title="Vertically-integrated y-face area (m³)")
hm_y = heatmap!(ax_y, ∫Ay_cpu)
cb_y = Colorbar(setup_fig[2, 4], hm_y)

ax_τˣ = Axis(setup_fig[3, 1], title="East-west wind stress (m² s⁻²)")
hm_τˣ = heatmap!(ax_τˣ, τˣ_cpu)
cb_τˣ = Colorbar(setup_fig[3, 2], hm_τˣ)

ax_τʸ = Axis(setup_fig[3, 3], title="North-south wind stress (m² s⁻²)")
hm_τʸ = heatmap!(ax_τʸ, τʸ_cpu)
cb_τʸ = Colorbar(setup_fig[3, 4], hm_τʸ)

display(setup_fig)
=#

#####
##### Initial condition:
#####

u, v, w = model.velocities
η = model.free_surface.η

T = model.tracers.T
T .= -1

S = model.tracers.S
S .= 30

#####
##### Simulation setup
#####

# Time-scale for gravity wave propagation across the smallest grid cell
g = model.free_surface.gravitational_acceleration
gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
    
minimum_Δx = abs(grid.radius * cosd(maximum(abs, grid.φᵃᶜᵃ[1:grid.Ny])) * deg2rad(grid.Δλ))
minimum_Δy = abs(grid.radius * deg2rad(grid.Δφ))
wave_propagation_time_scale = min(minimum_Δx, minimum_Δy) / gravity_wave_speed

if model.free_surface isa ExplicitFreeSurface
    Δt = 60seconds #0.2 * minimum_Δx / gravity_wave_speed
else
    Δt = 20minutes
end

start_time = [time_ns()]

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    η = model.free_surface.η

    if model.free_surface isa ExplicitFreeSurface
        @info @sprintf("Time: % 12s, iteration: %d, max(|η|): %.2e m, wall time: %s",
                       prettytime(sim.model.clock.time),
                       sim.model.clock.iteration,
                       maximum(abs, η),
                       prettytime(wall_time))
    else
        @info @sprintf("Time: % 12s, iteration: %d, free surface iterations: %d, max(|η|): %.2e m, wall time: %s",
                       prettytime(sim.model.clock.time),
                       sim.model.clock.iteration,
                       sim.model.free_surface.implicit_step_solver.preconditioned_conjugate_gradient_solver.iteration,
                       maximum(abs, η),
                       prettytime(wall_time))
    end

    start_time[1] = time_ns()

    return nothing
end

simulation = Simulation(model,
                        Δt = Δt,
                        stop_time = 60days,
                        iteration_interval = 10,
                        progress = progress)

output_fields = merge(model.velocities, model.tracers, (; η=model.free_surface.η))
output_prefix = "global_lat_lon_$(grid.Nx)_$(grid.Ny)_$(grid.Nz)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(10day),
                                                      prefix = output_prefix,
                                                      force = true)

# Let's goo!
@info "Running with Δt = $(prettytime(simulation.Δt))"

run!(simulation)

@info """

    Simulation took $(prettytime(simulation.run_time))

    Background diffusivity: $background_diffusivity
    Minimum wave propagation time scale: $(prettytime(wave_propagation_time_scale))
    Free surface: $(typeof(model.free_surface).name.wrapper)
    Time step: $(prettytime(Δt))

"""

#####
##### Visualize solution
#####

η_cpu = Array(interior(model.free_surface.η))[:, :, 1]
T_cpu = Array(interior(model.tracers.T))[:, :, end]
u_cpu = Array(interior(model.velocities.u))[:, :, end]
v_cpu = Array(interior(model.velocities.v))[:, :, end]

max_η = maximum(abs, η_cpu)
max_u = maximum(abs, u_cpu)
max_v = maximum(abs, v_cpu)

max_T = maximum(T_cpu)
min_T = minimum(T_cpu)

solution_fig = Figure(resolution = (1200, 600))

ax_η = Axis(solution_fig[1, 1], title="Free surface displacement (m)")
hm_η = heatmap!(ax_η, η_cpu, colorrange=(-max_η, max_η), colormap=:balance)
cb_η = Colorbar(solution_fig[1, 2], hm_η)

ax_T = Axis(solution_fig[2, 1], title="Sea surface temperature (ᵒC)")
hm_T = heatmap!(ax_T, T_cpu, colorrange=(min_T, max_T), colormap=:thermal)
cb_T = Colorbar(solution_fig[2, 2], hm_T)

ax_u = Axis(solution_fig[1, 3], title="East-West velocity (m s⁻¹)")
hm_u = heatmap!(ax_u, u_cpu, colorrange=(-max_u, max_u), colormap=:balance)
cb_u = Colorbar(solution_fig[1, 4], hm_u)

ax_v = Axis(solution_fig[2, 3], title="North-South velocity (m s⁻¹)")
hm_v = heatmap!(ax_v, v_cpu, colorrange=(-max_v, max_v), colormap=:balance)
cb_v = Colorbar(solution_fig[2, 4], hm_v)

display(solution_fig)
