using Statistics
using JLD2
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils

using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.Architectures: arch_array
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.TurbulenceClosures
using Oceananigans.Advection: VelocityStencil
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

output_prefix = "annual_cycle_global_lat_lon_$(Nx)_$(Ny)_$(Nz)_temp"

arch = CPU()
reference_density = 1035

#####
##### Load forcing files roughly from CORE2 paper
##### (Probably https://data1.gfdl.noaa.gov/nomads/forms/core/COREv2.html)
#####

using DataDeps

path = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/lat_lon_bathymetry_and_fluxes/"

dh = DataDep("near_global_lat_lon_3_degrees",
    "Forcing data for global latitude longitude simulation",
    [path * "bathymetry_lat_lon_128x60_FP32.bin",
     path * "sea_surface_temperature_25_128x60x12.jld2",
     path * "tau_x_128x60x12.jld2",
     path * "tau_y_128x60x12.jld2"]
)

DataDeps.register(dh)

datadep"near_global_lat_lon_3_degrees"

#####
##### Load forcing files roughly from CORE2 paper
##### (Probably https://data1.gfdl.noaa.gov/nomads/forms/core/COREv2.html)
#####

filename = [:sea_surface_temperature_25_128x60x12, :tau_x_128x60x12, :tau_y_128x60x12]

for name in filename
    datadep_path = @datadep_str "near_global_lat_lon_3_degrees/" * string(name) * ".jld2"
    file = Symbol(:file_, name)
    @eval $file = jldopen($datadep_path)
end

bathymetry_data = Array{Float32}(undef, Nx*Ny)
bathymetry_path = @datadep_str "near_global_lat_lon_3_degrees/bathymetry_lat_lon_128x60_FP32.bin"
read!(bathymetry_path, bathymetry_data)

bathymetry_data = bswap.(bathymetry_data) |> Array{Float64}
bathymetry_data = reshape(bathymetry_data, Nx, Ny)

τˣ = zeros(Nx, Ny, Nmonths)
τʸ = zeros(Nx, Ny, Nmonths)
T★ = zeros(Nx, Ny, Nmonths)

for month in 1:Nmonths
    τˣ[:, :, month] = file_tau_x_128x60x12["tau_x/$month"] ./ reference_density
    τʸ[:, :, month] = file_tau_y_128x60x12["tau_y/$month"] ./ reference_density
    T★[:, :, month] = file_sea_surface_temperature_25_128x60x12["sst25/$month"]
end

bathymetry = arch_array(arch, bathymetry_data)

H = 3600.0
# H = - minimum(bathymetry)

# Uncomment for a flat bottom:
# bathymetry = - H .* (bathymetry .< -10)

# A spherical domain
@show underlying_grid = LatitudeLongitudeGrid(arch,
                                              size = (Nx, Ny, Nz),
                                              longitude = (-180, 180),
                                              latitude = latitude,
                                              halo = (5, 5, 5),
                                              z = (-H, 0),
                                              precompute_metrics = true)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

τˣ = arch_array(arch, - τˣ)
τʸ = arch_array(arch, - τʸ)

target_sea_surface_temperature = T★ = arch_array(arch, T★)

#####
##### Physics and model setup
#####

νh = 1e+5
νz = 1e+1
κh = 1e+3
κz = 1e-4

vertical_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν = νz, κ = κz)

horizontal_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)
                                       
convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0)

gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = 1000.0,
                                                                κ_symmetric = 900.0,
                                                                slope_limiter = FluxTapering(1e-2))

#####
##### Boundary conditions / constant-in-time surface forcing
#####

Δz_top    = @allowscalar Δzᵃᵃᶜ(1, 1, grid.Nz, grid.underlying_grid)
Δz_bottom = @allowscalar Δzᵃᵃᶜ(1, 1, 1, grid.underlying_grid)

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

equation_of_state=LinearEquationOfState(thermal_expansion=2e-4)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    free_surface = free_surface,
                                    momentum_advection = VectorInvariant(scheme=WENO(), stencil=VelocityStencil()),
                                    tracer_advection = WENO(),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs),
                                    buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=30),
                                    tracers = :T,
                                    closure = (vertical_closure, convective_adjustment, gent_mcwilliams_diffusivity)) 

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

Δt = 20minutes

simulation = Simulation(model, Δt = Δt, stop_time = 5years)

start_time = [time_ns()]

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    η = model.free_surface.η
    u = model.velocities.u
    @info @sprintf("Time: % 12s, iteration: %d, max(|u|): %.2e ms⁻¹, max(|w|): %.2e ms⁻¹, wall time: %s",
                    prettytime(sim.model.clock.time),
                    sim.model.clock.iteration,
                    maximum(abs, u), maximum(abs, w),
                    prettytime(wall_time))

    start_time[1] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

u, v, w = model.velocities

T = model.tracers.T
η = model.free_surface.η

save_interval = 5days

simulation.output_writers[:surface_fields] = JLD2OutputWriter(model, (; u, v, T, η),
                                                              schedule = TimeInterval(save_interval),
                                                              filename = output_prefix * "_surface",
                                                              indices = (:, :, grid.Nz),
                                                              overwrite_existing = true)

simulation.output_writers[:atlantic] = JLD2OutputWriter(model, (; u, v, T, η),
                                                        schedule = TimeInterval(save_interval),
                                                        filename = output_prefix * "_atlantic",
                                                        indices = (60, :, :),
                                                        overwrite_existing = true)

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(1year),
                                                        prefix = output_prefix * "_checkpoint",
                                                        overwrite_existing = true)

# Let's goo!
@info "Running with Δt = $(prettytime(simulation.Δt))"

run!(simulation)

@info """

    Simulation took $(prettytime(simulation.run_wall_time))
    Free surface: $(typeof(model.free_surface).name.wrapper)
    Time step: $(prettytime(Δt))
"""

####
#### Visualize solution
####

using GLMakie, JLD2

output_prefix = "annual_cycle_global_lat_lon_128_60_18_temp"

surface_file = jldopen(output_prefix * "_surface.jld2")

iterations = parse.(Int, keys(surface_file["timeseries/t"]))

iter = Observable(0)

ηi(iter) = surface_file["timeseries/η/" * string(iter)][:, :, 1]
ui(iter) = surface_file["timeseries/u/" * string(iter)][:, :, 1]
vi(iter) = surface_file["timeseries/v/" * string(iter)][:, :, 1]
Ti(iter) = surface_file["timeseries/T/" * string(iter)][:, :, 1]
ti(iter) = string(surface_file["timeseries/t/" * string(iter)] / day)

η = @lift ηi($iter) 
u = @lift ui($iter)
v = @lift vi($iter)
T = @lift Ti($iter)

max_η = 2
min_η = - max_η
max_u = 0.2
min_u = - max_u
max_T = 32
min_T = 0

fig = Figure(resolution = (1200, 900))

ax = Axis(fig[1, 1], title="Free surface displacement (m)")
hm = GLMakie.heatmap!(ax, η, colorrange=(min_η, max_η), colormap=:balance)
cb = Colorbar(fig[1, 2], hm)

ax = Axis(fig[2, 1], title="Sea surface temperature (ᵒC)")
hm = GLMakie.heatmap!(ax, T, colorrange=(min_T, max_T), colormap=:thermal)
cb = Colorbar(fig[2, 2], hm)

ax = Axis(fig[1, 3], title="East-west surface velocity (m s⁻¹)")
hm = GLMakie.heatmap!(ax, u, colorrange=(min_u, max_u), colormap=:balance)
cb = Colorbar(fig[1, 4], hm)

ax = Axis(fig[2, 3], title="North-south surface velocity (m s⁻¹)")
hm = GLMakie.heatmap!(ax, v, colorrange=(min_u, max_u), colormap=:balance)
cb = Colorbar(fig[2, 4], hm)

title_str = @lift "Earth day = " * ti($iter)
ax_t = fig[0, :] = Label(fig, title_str)

GLMakie.record(fig, output_prefix * ".mp4", iterations, framerate=8) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end

display(fig)

close(surface_file)
