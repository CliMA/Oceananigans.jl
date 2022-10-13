using Statistics
using JLD2
using Printf
using Plots
using Oceananigans
using Oceananigans.Units

using Oceananigans.Fields: interpolate, Field
using Oceananigans.Architectures: arch_array
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.BoundaryConditions
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, inactive_node, peripheral_node
using CUDA: @allowscalar, device!
using Oceananigans.Operators
using Oceananigans.Operators: Δzᵃᵃᶜ
using Oceananigans: prognostic_fields

using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField

@inline function visualize(field, lev, dims)
    (dims == 1) && (idx = (lev, :, :))
    (dims == 2) && (idx = (:, lev, :))
    (dims == 3) && (idx = (:, :, lev))

    r = deepcopy(Array(interior(field)))[idx...]
    r[ r.==0 ] .= NaN
    return r
end

#####
##### Grid
#####

arch = GPU()
reference_density = 1029

latitude = (-75, 75)

# 0.25 degree resolution
Nx = 1440
Ny = 600
Nz = 1

const Nyears  = 1
const Nmonths = 12
const thirty_days = 30days

output_prefix = "near_global_lat_lon_$(Nx)_$(Ny)_$(Nz)_fine"
pickup_file   = false

#####
##### Load forcing files and inital conditions from ECCO version 4
##### https://ecco.jpl.nasa.gov/drive/files
##### Bathymetry is interpolated from ETOPO1 https://www.ngdc.noaa.gov/mgg/global/
#####

using DataDeps

path = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/ss/new_hydrostatic_data_after_cleared_bugs/quarter_degree_near_global_input_data/"

datanames = ["z_faces-50-levels",
             "bathymetry-1440x600",
             "temp-1440x600-latitude-75",
             "salt-1440x600-latitude-75",
             "tau_x-1440x600-latitude-75",
             "tau_y-1440x600-latitude-75",
             "initial_conditions"]

dh = DataDep("quarter_degree_near_global_lat_lon",
    "Forcing data for global latitude longitude simulation",
    [path * data * ".jld2" for data in datanames]
)

DataDeps.register(dh)

datadep"quarter_degree_near_global_lat_lon"

files = [:file_z_faces, :file_bathymetry, :file_temp, :file_salt, :file_tau_x, :file_tau_y, :file_init]
for (data, file) in zip(datanames, files)
    datadep_path = @datadep_str "quarter_degree_near_global_lat_lon/" * data * ".jld2"
    @eval $file = jldopen($datadep_path)
end

bathymetry = file_bathymetry["bathymetry"]
bathymetry[bathymetry .< 0] .= -10e3

τˣ = zeros(Nx, Ny, Nmonths)
τʸ = zeros(Nx, Ny, Nmonths)

# Files contain 1 year (1992) of 12 monthly averages
τˣ = file_tau_x["field"] ./ reference_density
τʸ = file_tau_y["field"] ./ reference_density

# Remember the convention!! On the surface a negative flux increases a positive decreases
bathymetry = arch_array(arch, bathymetry)

# A spherical domain
@show underlying_grid = LatitudeLongitudeGrid(arch,
                                              size = (Nx, Ny, Nz),
                                              longitude = (-180, 180),
                                              latitude = latitude,
                                              halo = (3, 3, 3),
                                              z = (-10e3, 0),
                                              #z = z_faces,
                                              precompute_metrics = true)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

τˣ = arch_array(arch, - τˣ)
τʸ = arch_array(arch, - τʸ)

#####
##### Boundary conditions / time-dependent fluxes
#####

@inline current_time_index(time, tot_months)     = mod(unsafe_trunc(Int32, time / thirty_days), tot_months) + 1
@inline next_time_index(time, tot_months)        = mod(unsafe_trunc(Int32, time / thirty_days) + 1, tot_months) + 1
@inline cyclic_interpolate(u₁::Number, u₂, time) = u₁ + mod(time / thirty_days, 1) * (u₂ - u₁)

Δz_top    = @allowscalar Δzᵃᵃᶜ(1, 1, grid.Nz, grid.underlying_grid)
Δz_bottom = @allowscalar Δzᵃᵃᶜ(1, 1, 1, grid.underlying_grid)

@inline function surface_wind_stress(i, j, grid, clock, fields, τ)
    time = clock.time
    n₁ = current_time_index(time, Nmonths)
    n₂ = next_time_index(time, Nmonths)

    @inbounds begin
        τ₁ = τ[i, j, n₁]
        τ₂ = τ[i, j, n₂]
    end

    return cyclic_interpolate(τ₁, τ₂, time)
end

u_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress, discrete_form = true, parameters = τˣ)
v_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress, discrete_form = true, parameters = τʸ)

@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

# Linear bottom drag:
μ = 0.001 # ms⁻¹

u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag, discrete_form = true, parameters = μ)
v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag, discrete_form = true, parameters = μ)

u_bcs = FieldBoundaryConditions(top = u_wind_stress_bc, bottom = u_bottom_drag_bc)
v_bcs = FieldBoundaryConditions(top = v_wind_stress_bc, bottom = v_bottom_drag_bc)

free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver)

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState())

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    free_surface = free_surface,
                                    momentum_advection = VectorInvariant(),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    buoyancy = nothing,
                                    boundary_conditions = (u=u_bcs, v=v_bcs))

#####
##### Initial condition:
#####

u, v, w = model.velocities
η = model.free_surface.η

@info "model initialized"

#####
##### Simulation setup
#####

ζ = VerticalVorticityField(model)
compute!(ζ)

Δt = 6minutes  # for initialization, then we can go up to 6 minutes?
simulation = Simulation(model, Δt = Δt, stop_time = Nyears*years)
start_time = [time_ns()]

using Oceananigans.Utils 

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    u = sim.model.velocities.u
    η = sim.model.free_surface.η

    @info @sprintf("Time: % 12s, iteration: %d, max(|u|): %.2e ms⁻¹, wall time: %s", 
                    prettytime(sim.model.clock.time),
                    sim.model.clock.iteration, maximum(abs, u),
                    prettytime(wall_time))

    start_time[1] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

u, v, w = model.velocities
η = model.free_surface.η

output_fields = merge(model.velocities, (η=model.free_surface.η, ζ=ζ))
save_interval = 1days

simulation.output_writers[:surface_fields] = JLD2OutputWriter(model, output_fields, #(; u, v, T, S, η),
                                                              schedule = TimeInterval(save_interval),
                                                              filename = output_prefix * "_surface",
                                                              indices = (:, :, grid.Nz),
                                                              overwrite_existing = true)

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(1day),
                                                        prefix = output_prefix * "_checkpoint",
                                                        overwrite_existing = true)

@info "Running with Δt = $(prettytime(simulation.Δt))"

run!(simulation, pickup = pickup_file)

@info """

    Simulation took $(prettytime(simulation.run_wall_time))
    Free surface: $(typeof(model.free_surface).name.wrapper)
    Time step: $(prettytime(Δt))
"""
