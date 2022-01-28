using Statistics
using JLD2
using Printf
using Oceananigans
using Oceananigans.Units

using Oceananigans.Fields: interpolate
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.Architectures: arch_array
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity, VerticallyImplicitTimeDiscretization
using CUDA: @allowscalar
using Oceananigans.Operators: Δzᵃᵃᶜ

include("cyclic_interpolate_utils.jl")

#####
##### Grid
#####

arch = GPU()
reference_density = 1035

latitude = (-84.375, 84.375)
Δφ = latitude[2] - latitude[1]

# 2.8125 degree resolution for the fluxes data
Nx_fluxes = 128
Ny_fluxes = 60

# 0.25 degree resolution
Nx = 1440
Ny = 675
Nz = 36

output_prefix = "annual_cycle_global_lat_lon_$(Nx)_$(Ny)_$(Nz)_fine"

#####
##### Load forcing files roughly from CORE2 paper
##### (Probably https://data1.gfdl.noaa.gov/nomads/forms/core/COREv2.html)
#####

using DataDeps

path = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/lat_lon_bathymetry_and_fluxes/"

dh = DataDep("near_global_lat_lon",
    "Forcing data for global latitude longitude simulation",
    [path * "bathymetry_lat_lon_128x60_FP32.bin",
     path * "sea_surface_temperature_25_128x60x12.jld2",
     path * "tau_x_128x60x12.jld2",
     path * "tau_y_128x60x12.jld2"]
)

DataDeps.register(dh)

datadep"near_global_lat_lon"

#####
##### Load forcing files roughly from CORE2 paper
##### (Probably https://data1.gfdl.noaa.gov/nomads/forms/core/COREv2.html)
#####

filename = [:sea_surface_temperature_25_128x60x12, :tau_x_128x60x12, :tau_y_128x60x12]

for name in filename
    datadep_path = @datadep_str "near_global_lat_lon/" * string(name) * ".jld2"
    file = Symbol(:file_, name)
    @eval $file = jldopen($datadep_path)
end

bathymetry_data = Array{Float32}(undef, Nx_fluxes*Ny_fluxes)
bathymetry_path = @datadep_str "near_global_lat_lon/bathymetry_lat_lon_128x60_FP32.bin"
read!(bathymetry_path, bathymetry_data)

bathymetry_data = bswap.(bathymetry_data) |> Array{Float64}
bathymetry_data = reshape(bathymetry_data, Nx_fluxes, Ny_fluxes)
bathymetry      = interpolate_fluxes(bathymetry_data, Nx_fluxes, Ny_fluxes, Nx, Ny)
    
τˣ_data = zeros(Nx_fluxes, Ny_fluxes, Nmonths)
τʸ_data = zeros(Nx_fluxes, Ny_fluxes, Nmonths)
T★_data = zeros(Nx_fluxes, Ny_fluxes, Nmonths)

τˣ = zeros(Nx, Ny, Nmonths)
τʸ = zeros(Nx, Ny, Nmonths)
T★ = zeros(Nx, Ny, Nmonths)

for month in 1:Nmonths
    τˣ_data[:, :, month] = file_tau_x_128x60x12["tau_x/$month"] ./ reference_density
    τʸ_data[:, :, month] = file_tau_y_128x60x12["tau_y/$month"] ./ reference_density
    T★_data[:, :, month] = file_sea_surface_temperature_25_128x60x12["sst25/$month"]
    
    τˣ[:, :, month] = interpolate_fluxes(τˣ_data[:, :, month], Nx_fluxes, Ny_fluxes, Nx, Ny)
    τʸ[:, :, month] = interpolate_fluxes(τʸ_data[:, :, month], Nx_fluxes, Ny_fluxes, Nx, Ny)
    T★[:, :, month] = interpolate_fluxes(T★_data[:, :, month], Nx_fluxes, Ny_fluxes, Nx, Ny)
end

bathymetry = arch_array(arch, bathymetry)
τˣ = arch_array(arch, - τˣ)
τʸ = arch_array(arch, - τʸ)
target_sea_surface_temperature = T★ = arch_array(arch, T★)

H = 3600.0
# H = - minimum(bathymetry)

# Uncomment for a flat bottom:
# bathymetry = - H .* (bathymetry .< -10)

# stretched grid 
k_center = collect(1:Nz)
Δz_center = @. 11.6 * 1.104^(Nz - k_center)

const Lz = sum(Δz_center)

z_faces = vcat([-Lz], -Lz .+ cumsum(Δz_center))
z_faces[Nz+1] = 0

# A spherical domain
@show underlying_grid = LatitudeLongitudeGrid(arch,
                                              size = (Nx, Ny, Nz),
                                              longitude = (-180, 180),
                                              latitude = latitude,
                                              halo = (3, 3, 3),
                                              z = z_faces,
                                              precompute_metrics = true)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

#####
##### Physics and model setup
#####

νh = 1e+4
νz = 1e+1
κh = 1e+3
κz = 1e-4

background_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=νh, νz=νz, κh=κh, κz=κz,
                                                                       time_discretization = VerticallyImplicitTimeDiscretization())

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0)

#####
##### Boundary conditions / constant-in-time surface forcing
#####

Δz_top    = @allowscalar Δzᵃᵃᶜ(1, 1, grid.Nz, grid.grid)
Δz_bottom = @allowscalar Δzᵃᵃᶜ(1, 1, 1, grid.grid)

@inline function surface_temperature_relaxation(i, j, grid, clock, fields, p)
    time = clock.time

    n₁ = current_time_index(time)
    n₂ = next_time_index(time)

    @inbounds begin
        T★₁ = p.T★[i, j, n₁]
        T★₂ = p.T★[i, j, n₂]
        T_surface = fields.T[i, j, grid.Nz]
    end

    T★ = cyclic_interpolate(T★₁, T★₂, time)
                                
    return p.λ * (T_surface - T★)
end

T_surface_relaxation_bc = FluxBoundaryCondition(surface_temperature_relaxation,
                                                discrete_form = true,
                                                parameters = (λ = Δz_top/3days, T★ = target_sea_surface_temperature))

@inline function wind_stress(i, j, grid, clock, fields, τ)
    time = clock.time
    n₁ = current_time_index(time)
    n₂ = next_time_index(time)

    @inbounds begin
        τ₁ = τ[i, j, n₁]
        τ₂ = τ[i, j, n₂]
    end

    return cyclic_interpolate(τ₁, τ₂, time)
end

u_wind_stress_bc = FluxBoundaryCondition(wind_stress, discrete_form = true, parameters = τˣ)
v_wind_stress_bc = FluxBoundaryCondition(wind_stress, discrete_form = true, parameters = τʸ)

@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

# Linear bottom drag:
μ = Δz_bottom / 10days

u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag, discrete_form = true, parameters = μ)
v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag, discrete_form = true, parameters = μ)

u_bcs = FieldBoundaryConditions(top = u_wind_stress_bc, bottom = u_bottom_drag_bc)
v_bcs = FieldBoundaryConditions(top = v_wind_stress_bc, bottom = v_bottom_drag_bc)
T_bcs = FieldBoundaryConditions(top = T_surface_relaxation_bc)

free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver)

buoyancy     = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4, β=0.0), constant_salinity = true)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    free_surface = free_surface,
                                    momentum_advection = VectorInvariant(),
                                    tracer_advection = WENO5(grid = underlying_grid),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs),
                                    buoyancy = buoyancy,
                                    tracers = (:T, ),
                                    closure = (background_diffusivity, convective_adjustment))


@show model.free_surface.implicit_step_solver.matrix_iterative_solver.preconditioner_method

#####
##### Initial condition:
#####

u, v, w = model.velocities
η = model.free_surface.η
T = model.tracers.T
T .= -1



#####
##### Simulation setup
#####

# Time-scale for gravity wave propagation across the smallest grid cell
g = model.free_surface.gravitational_acceleration
gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
    
minimum_Δx = abs(grid.radius * cosd(maximum(abs, grid.φᵃᶜᵃ[1:grid.Ny])) * deg2rad(grid.Δλᶜᵃᵃ))
minimum_Δy = abs(grid.radius * deg2rad(grid.Δφᵃᶜᵃ))
wave_propagation_time_scale = min(minimum_Δx, minimum_Δy) / gravity_wave_speed

if model.free_surface isa ExplicitFreeSurface
    Δt = 10seconds
else
    Δt = 3minutes
end

simulation = Simulation(model, Δt = Δt, stop_time = 30years)

start_time = [time_ns()]

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    η = model.free_surface.η
    u = model.velocities.u
    @info @sprintf("Time: % 12s, iteration: %d, max(|η|): %.2e m, max(|u|): %.2e ms⁻¹, wall time: %s",
                    prettytime(sim.model.clock.time),
                    sim.model.clock.iteration,
                    maximum(abs, η), maximum(abs, u),
                    prettytime(wall_time))

    start_time[1] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

u, v, w = model.velocities
T = model.tracers.T
η = model.free_surface.η

output_fields = (; u, v, T, η)
save_interval = 5days

simulation.output_writers[:surface_fields] = JLD2OutputWriter(model, (; u, v, T, η),
                                                              schedule = TimeInterval(save_interval),
                                                              prefix = output_prefix * "_surface",
                                                              field_slicer = FieldSlicer(k=grid.Nz),
                                                              force = true)

simulation.output_writers[:bottom_fields] = JLD2OutputWriter(model, (; u, v, T),
                                                             schedule = TimeInterval(save_interval),
                                                             prefix = output_prefix * "_bottom",
                                                             field_slicer = FieldSlicer(k=1),
                                                             force = true)

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(1year),
                                                        prefix = output_prefix * "_checkpoint",
                                                        cleanup = true,
                                                        force = true)

# Let's goo!
@info "Running with Δt = $(prettytime(simulation.Δt))"

run!(simulation)

@info """

    Simulation took $(prettytime(simulation.run_wall_time))
    Background diffusivity: $background_diffusivity
    Minimum wave propagation time scale: $(prettytime(wave_propagation_time_scale))
    Free surface: $(typeof(model.free_surface).name.wrapper)
    Time step: $(prettytime(Δt))
"""

####
#### Visualize solution
####

# using GLMakie

# surface_file = jldopen(output_prefix * "_surface.jld2")
# bottom_file = jldopen(output_prefix * "_bottom.jld2")

# iterations = parse.(Int, keys(surface_file["timeseries/t"]))
# iterations = iterations[15:end]

# iter = Node(0)

# ηi(iter) = surface_file["timeseries/η/" * string(iter)][:, :, 1]
# ui(iter) = surface_file["timeseries/u/" * string(iter)][:, :, 1]
# vi(iter) = surface_file["timeseries/v/" * string(iter)][:, :, 1]
# Ti(iter) = surface_file["timeseries/T/" * string(iter)][:, :, 1]
# ti(iter) = string(surface_file["timeseries/t/" * string(iter)] / day)

# ubi(iter) = bottom_file["timeseries/u/" * string(iter)][:, :, 1]
# vbi(iter) = bottom_file["timeseries/v/" * string(iter)][:, :, 1]

# η = @lift ηi($iter) 
# u = @lift ui($iter)
# v = @lift vi($iter)
# T = @lift Ti($iter)

# ub = @lift ubi($iter)
# vb = @lift vbi($iter)

# max_η = 4
# min_η = - max_η
# max_u = 0.2
# min_u = - max_u
# max_T = 32
# min_T = 0

# fig = Figure(resolution = (1200, 900))

# ax = Axis(fig[1, 1], title="Free surface displacement (m)")
# hm = GLMakie.heatmap!(ax, η, colorrange=(min_η, max_η), colormap=:balance)
# cb = Colorbar(fig[1, 2], hm)

# ax = Axis(fig[2, 1], title="Sea surface temperature (ᵒC)")
# hm = GLMakie.heatmap!(ax, T, colorrange=(min_T, max_T), colormap=:thermal)
# cb = Colorbar(fig[2, 2], hm)

# ax = Axis(fig[1, 3], title="East-west surface velocity (m s⁻¹)")
# hm = GLMakie.heatmap!(ax, u, colorrange=(min_u, max_u), colormap=:balance)
# cb = Colorbar(fig[1, 4], hm)

# ax = Axis(fig[2, 3], title="North-south surface velocity (m s⁻¹)")
# hm = GLMakie.heatmap!(ax, v, colorrange=(min_u, max_u), colormap=:balance)
# cb = Colorbar(fig[2, 4], hm)

# ax = Axis(fig[3, 1], title="East-west bottom velocity (m s⁻¹)")
# hm = GLMakie.heatmap!(ax, ub, colorrange=(min_u, max_u), colormap=:balance)
# cb = Colorbar(fig[3, 2], hm)

# ax = Axis(fig[3, 3], title="North-south bottom velocity (m s⁻¹)")
# hm = GLMakie.heatmap!(ax, vb, colorrange=(min_u, max_u), colormap=:balance)
# cb = Colorbar(fig[3, 4], hm)

# title_str = @lift "Earth day = " * ti($iter)
# ax_t = fig[0, :] = Label(fig, title_str)

# GLMakie.record(fig, output_prefix * ".mp4", iterations, framerate=8) do i
#     @info "Plotting iteration $i of $(iterations[end])..."
#     iter[] = i
# end

# display(fig)

# close(surface_file)
