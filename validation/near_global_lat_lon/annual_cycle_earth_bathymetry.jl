using Statistics
using JLD2
using Printf
using GLMakie
using Oceananigans
using Oceananigans.Units

using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.Architectures: arch_array
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity
using CUDA: @allowscalar
using Oceananigans.Operators: Δzᵃᵃᶜ

include("cyclic_interpolate_utils.jl")

#####
##### Grid
#####

latitude = (-84.375, 84.375)
Δφ = latitude[2] - latitude[1]

# 2.8125 degree resolution
Nx = 128
Ny = 60
Nz = 18

output_prefix = "annual_cycle_global_lat_lon_$(Nx)_$(Ny)_$Nz"

arch = CPU()
reference_density = 1035

#####
##### Load forcing files roughly from CORE2 paper
##### (Probably https://data1.gfdl.noaa.gov/nomads/forms/core/COREv2.html)
#####

bathymetry_path = "bathy_128x60var4.bin"
east_west_stress_path = "off_TAUXvar1.bin"
north_south_stress_path = "off_TAUY.bin"
sea_surface_temperature_path="sst25_128x60x12.bin"

Nmonths = 12
bytes = sizeof(Float32) * Nx * Ny

bathymetry = reshape(bswap.(reinterpret(Float32, read(bathymetry_path, bytes))), (Nx, Ny))
τˣ = - reshape(bswap.(reinterpret(Float32, read(east_west_stress_path, Nmonths * bytes))), (Nx, Ny, Nmonths)) ./ reference_density
τʸ = - reshape(bswap.(reinterpret(Float32, read(north_south_stress_path, Nmonths * bytes))), (Nx, Ny, Nmonths)) ./ reference_density
target_sea_surface_temperature = reshape(bswap.(reinterpret(Float32, read(sea_surface_temperature_path, Nmonths * bytes))), (Nx, Ny, Nmonths))

bathymetry = arch_array(arch, bathymetry)
τˣ = arch_array(arch, τˣ)
τʸ = arch_array(arch, τʸ)
target_sea_surface_temperature = T★ = arch_array(arch, target_sea_surface_temperature)

H = 3600.0
# H = - minimum(bathymetry)

# Uncomment for a flat bottom:
# bathymetry = - H .* (bathymetry .< -10)

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

Δz_top = @allowscalar Δzᵃᵃᶜ(1, 1, grid.Nz, grid.grid)
Δz_bottom = @allowscalar Δzᵃᵃᶜ(1, 1, 1, grid.grid)

@inline function surface_temperature_relaxation(i, j, grid, clock, fields, p)
    time = clock.time

    n₁ = current_time_index(time)
    n₂ = next_time_index(time)

    T★₁ = @inbounds p.T★[i, j, n₁]
    T★₂ = @inbounds p.T★[i, j, n₂]

    T★ = cyclic_interpolate(T★₁, T★₂, time)
                                
    T_surface = @inbounds fields.T[i, j, grid.Nz]

    return p.λ * (T_surface - T★)
end

T_surface_relaxation_bc = FluxBoundaryCondition(surface_temperature_relaxation,
                                                discrete_form = true,
                                                parameters = (λ = Δz_top/3days, T★ = target_sea_surface_temperature))

@inline function wind_stress(i, j, time, τ)
    n₁ = current_time_index(time)
    n₂ = next_time_index(time)
    τ₁ = @inbounds τ[i, j, n₁]
    τ₂ = @inbounds τ[i, j, n₂]
    return cyclic_interpolate(τ₁, τ₂, time)
end

@inline wind_stress_x(i, j, grid, clock, fields, τˣ) = wind_stress(i, j, clock.time, τˣ)
@inline wind_stress_y(i, j, grid, clock, fields, τʸ) = wind_stress(i, j, clock.time, τʸ)

u_wind_stress_bc = FluxBoundaryCondition(wind_stress_x, discrete_form = true, parameters = τˣ)
v_wind_stress_bc = FluxBoundaryCondition(wind_stress_y, discrete_form = true, parameters = τʸ)

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

#####
##### Initial condition:
#####

u, v, w = model.velocities
η = model.free_surface.η
T = model.tracers.T
T .= 5
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
    Δt = 60seconds
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

simulation = Simulation(model, Δt = Δt, stop_time = 60day, iteration_interval = 10, progress = progress)

u, v, w = model.velocities
T, S = model.tracers
η = model.free_surface.η

output_fields = (; u, v, T, S, η)

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(1day),
                                                      prefix = output_prefix,
                                                      field_slicer = FieldSlicer(k=grid.Nz),
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

file = jldopen(output_prefix * ".jld2")

iterations = parse.(Int, keys(file["timeseries/t"]))

iter = Node(0)

ηi(iter) = file["timeseries/η/" * string(iter)][:, :, 1]
ui(iter) = file["timeseries/u/" * string(iter)][:, :, 1]
vi(iter) = file["timeseries/v/" * string(iter)][:, :, 1]
Ti(iter) = file["timeseries/T/" * string(iter)][:, :, 1]
ti(iter) = string(file["timeseries/t/" * string(iter)] / day)

η = @lift ηi($iter) 
u = @lift ui($iter)
v = @lift vi($iter)
T = @lift Ti($iter)

max_η = 10
min_η = - max_η
max_u = 10
min_u = -max_u
max_T = 0
min_T = 20

#max_η = @lift + maximum(abs, ηi($iter))
#min_η = @lift - maximum(abs, ηi($iter))
#max_u = @lift + maximum(abs, ui($iter))
#min_u = @lift - maximum(abs, ui($iter))
#max_T = @lift maximum(Ti($iter))
#min_T = @lift minimum(Ti($iter))

fig = Figure(resolution = (1200, 600))

ax_η = Axis(fig[1, 1], title="Free surface displacement (m)")
hm_η = heatmap!(ax_η, η, colorrange=(min_η, max_η), colormap=:balance)
cb_η = Colorbar(fig[1, 2], hm_η)

ax_T = Axis(fig[2, 1], title="Sea surface temperature (ᵒC)")
hm_T = heatmap!(ax_T, T, colorrange=(min_T, max_T), colormap=:thermal)
cb_T = Colorbar(fig[2, 2], hm_T)

ax_u = Axis(fig[1, 3], title="East-west velocity (m s⁻¹)")
hm_u = heatmap!(ax_u, u, colorrange=(min_u, max_u), colormap=:balance)
cb_u = Colorbar(fig[1, 4], hm_u)

ax_v = Axis(fig[2, 3], title="East-west velocity (m s⁻¹)")
hm_v = heatmap!(ax_v, v, colorrange=(min_u, max_u), colormap=:balance)
cb_v = Colorbar(fig[2, 4], hm_v)

title_str = @lift "Earth day = " * ti($iter)
ax_t = fig[0, :] = Label(fig, title_str)

GLMakie.record(fig, output_prefix * ".mp4", iterations, framerate=8) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end

display(fig)

close(file)
