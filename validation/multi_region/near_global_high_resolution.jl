import Pkg
Pkg.activate("/home/ssilvest/Oceananigans.jl")

using Statistics
using JLD2
using Printf
using Oceananigans
using Oceananigans.Units

using Oceananigans.Operators
using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: multi_region_object_from_array
using Oceananigans.Fields: interpolate, Field
using Oceananigans.Architectures: arch_array
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.BoundaryConditions
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, PartialCellBottom, inactive_node, peripheral_node
using CUDA: @allowscalar, device!
using Oceananigans.Operators
using Oceananigans.Operators: Δzᵃᵃᶜ
using Oceananigans: prognostic_fields
using SeawaterPolynomials

include("horizontal_visc.jl")

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

arch = GPU();
reference_density = 1029;

latitude = (-75, 75);

# 1/12 degree resolution
Nx = 4320;
Ny = 1800;
Nz = 48;

const Nyears  = 1;
const Nmonths = 12;
const thirty_days = 30days;

output_prefix = "near_global_lat_lon_$(Nx)_$(Ny)_$(Nz)_fine"
pickup_file   = false #"near_global_lat_lon_4320_1800_48_fine_checkpoint_iteration5000.jld2" 

#####
##### Load forcing files and inital conditions from ECCO version 4
##### https://ecco.jpl.nasa.gov/drive/files
##### Bathymetry is interpolated from ETOPO1 https://www.ngdc.noaa.gov/mgg/global/
#####

using DataDeps

path = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/ss/new_hydrostatic_data_after_cleared_bugs/quarter_degree_near_global_input_data/"

dh = DataDep("quarter_degree_near_global_lat_lon",
    "Forcing data for global latitude longitude simulation",
    path * "z_faces-50-levels.jld2"
)

DataDeps.register(dh)

datadep"quarter_degree_near_global_lat_lon"

datadep_path = @datadep_str "quarter_degree_near_global_lat_lon/z_faces-50-levels.jld2"
file_z_faces = jldopen(datadep_path)

bathy_path = "../data/bathymetry-ad-hoc.jld2" # "smooth-bathymetry.jld2" #
bathymetry = jldopen(bathy_path)["bathymetry"]

τˣ = zeros(Nx, Ny, Nmonths)
τʸ = zeros(Nx, Ny, Nmonths)
T★ = zeros(Nx, Ny, Nmonths)
S★ = zeros(Nx, Ny, Nmonths)

path_bc = "../data/boundary_conditions_twelth_degree.jld2"

# Files contain 1 year (1992) of 12 monthly averages
τˣ = jldopen(path_bc)["τˣ"] ./ reference_density;
τʸ = jldopen(path_bc)["τʸ"] ./ reference_density;
T★ = jldopen(path_bc)["Tₛ"];
S★ = jldopen(path_bc)["Sₛ"];

T_bounds = extrema(T★)
S_bounds = extrema(S★)

# Remember the convention!! On the surface a negative flux increases a positive decreases
bathymetry = arch_array(arch, bathymetry);

# Stretched faces taken from ECCO Version 4 (50 levels in the vertical)
z_faces = file_z_faces["z_faces"][3:end];

# A spherical domain
@show underlying_grid = LatitudeLongitudeGrid(arch,
                                              size = (Nx, Ny, Nz),
                                              longitude = (-180, 180),
                                              latitude = latitude,
                                              halo = (5, 5, 5),
                                              z = z_faces,
                                              precompute_metrics = true)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry));

underlying_mrg = MultiRegionGrid(underlying_grid, partition = XPartition(3), devices = 3);
mrg            = MultiRegionGrid(grid,            partition = XPartition(3), devices = 3);

τˣ = multi_region_object_from_array(- τˣ, mrg);
τʸ = multi_region_object_from_array(- τʸ, mrg);

target_sea_surface_temperature = T★ = multi_region_object_from_array(T★, mrg);
target_sea_surface_salinity    = S★ = multi_region_object_from_array(S★, mrg);

#####
##### Physics and model setup
#####

using Oceananigans.Operators: Δx, Δy
using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: HorizontalDivergenceFormulation, HorizontalFormulation

include("leith_viscosity.jl")
include("horizontal_visc.jl")
convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=0.2, background_κz=1e-5, background_νz=5e-4) 
# biharmonic_viscosity  = LeithBiharmonicViscosity(C_vort = 2.75, C_div = 3.75)
biharmonic_viscosity  = leith_viscosity(HorizontalDivergenceFormulation(), grid; C_vort = 2.75, C_div = 3.75)
closures = (biharmonic_viscosity, convective_adjustment)

#####
##### Boundary conditions / time-dependent fluxes 
#####

@inline current_time_index(time, tot_months)     = mod(unsafe_trunc(Int32, time / thirty_days), tot_months) + 1
@inline next_time_index(time, tot_months)        = mod(unsafe_trunc(Int32, time / thirty_days) + 1, tot_months) + 1
@inline cyclic_interpolate(u₁::Number, u₂, time) = u₁ + mod(time / thirty_days, 1) * (u₂ - u₁)

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

Δz_top = @allowscalar grid.Δzᵃᵃᶜ[Nz]

using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ

# Linear bottom drag:
μ = 0.003 # Non dimensional

@inline speedᶠᶜᶜ(i, j, k, grid, fields) = (fields.u[i, j, k]^2 + ℑxyᶠᶜᵃ(i, j, k, grid, fields.v)^2)^0.5
@inline speedᶜᶠᶜ(i, j, k, grid, fields) = (fields.v[i, j, k]^2 + ℑxyᶜᶠᵃ(i, j, k, grid, fields.u)^2)^0.5

@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1] * speedᶠᶜᶜ(i, j, 1, grid, fields)
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1] * speedᶜᶠᶜ(i, j, 1, grid, fields)

@inline u_immersed_bottom_drag(i, j, k, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, k] * speedᶠᶜᶜ(i, j, k, grid, fields) 
@inline v_immersed_bottom_drag(i, j, k, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, k] * speedᶜᶠᶜ(i, j, k, grid, fields) 

drag_u = FluxBoundaryCondition(u_immersed_bottom_drag, discrete_form=true, parameters = μ)
drag_v = FluxBoundaryCondition(v_immersed_bottom_drag, discrete_form=true, parameters = μ)

u_immersed_bc = ImmersedBoundaryCondition(bottom = drag_u)
v_immersed_bc = ImmersedBoundaryCondition(bottom = drag_v)

u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag, discrete_form = true, parameters = μ)
v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag, discrete_form = true, parameters = μ)

u_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress, discrete_form = true, parameters = τˣ);
v_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress, discrete_form = true, parameters = τʸ);

@inline function surface_temperature_relaxation(i, j, grid, clock, fields, p)
    time = clock.time

    n₁ = current_time_index(time, Nmonths)
    n₂ = next_time_index(time, Nmonths)

    @inbounds begin
        T★₁ = p.T★[i, j, n₁]
        T★₂ = p.T★[i, j, n₂]
        T_surface = fields.T[i, j, grid.Nz]
    end

    T★ = cyclic_interpolate(T★₁, T★₂, time)
                                
    return p.λ * (T_surface - T★)
end

@inline function surface_salinity_relaxation(i, j, grid, clock, fields, p)
    time = clock.time

    n₁ = current_time_index(time, Nmonths)
    n₂ = next_time_index(time, Nmonths)

    @inbounds begin
        S★₁ = p.S★[i, j, n₁]
        S★₂ = p.S★[i, j, n₂]
        S_surface = fields.S[i, j, grid.Nz]
    end

    S★ = cyclic_interpolate(S★₁, S★₂, time)
                                
    return p.λ * (S_surface - S★)
end

T_surface_relaxation_bc = FluxBoundaryCondition(surface_temperature_relaxation,
                                                discrete_form = true,
                                                parameters = (λ = Δz_top/7days, T★ = target_sea_surface_temperature));

S_surface_relaxation_bc = FluxBoundaryCondition(surface_salinity_relaxation,
                                                discrete_form = true,
                                                parameters = (λ = Δz_top/7days, S★ = target_sea_surface_salinity));

u_bcs = FieldBoundaryConditions(top = u_wind_stress_bc, bottom = u_bottom_drag_bc, immersed = u_immersed_bc);
v_bcs = FieldBoundaryConditions(top = v_wind_stress_bc, bottom = v_bottom_drag_bc, immersed = v_immersed_bc);
T_bcs = FieldBoundaryConditions(top = T_surface_relaxation_bc);
S_bcs = FieldBoundaryConditions(top = S_surface_relaxation_bc);

free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver);
equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(; reference_density)

buoyancy = SeawaterBuoyancy(; equation_of_state)

using Oceananigans.Advection: VelocityStencil, EnstrophyConservingScheme

model = HydrostaticFreeSurfaceModel(grid = mrg,
                                    free_surface = free_surface,
                                    momentum_advection = WENO(vector_invariant = VelocityStencil()),
                                    coriolis = HydrostaticSphericalCoriolis(scheme = EnstrophyConservingScheme()),
                                    buoyancy = buoyancy,
                                    tracers = (:T, :S),
                                    closure = closures,
                                    boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs),
                                    tracer_advection = WENO(underlying_mrg))

#####
##### Initial condition:
#####

u, v, w = model.velocities
η = model.free_surface.η
T = model.tracers.T
S = model.tracers.S

@info "Reading initial conditions"
file_init = jldopen("../data/evolved-initial-conditions-70days.jld2")

model.clock.time = 70 * days 

@info "initializing model"
T_init = multi_region_object_from_array(file_init["T"], mrg);
S_init = multi_region_object_from_array(file_init["S"], mrg);
u_init = multi_region_object_from_array(file_init["u"], mrg);
v_init = multi_region_object_from_array(file_init["v"], mrg);
η_init = multi_region_object_from_array(file_init["η"], mrg);
set!(model, T=T_init, S=S_init, u=u_init, v=v_init, η=η_init)

@info "model initialized"

#####
##### Simulation setup
#####

Δt = 120  # for initialization, then we can go up to 6 minutes?

simulation = Simulation(model, Δt = Δt, stop_iteration = 14400)

start_time = [time_ns()]

using Oceananigans.Utils 
using Oceananigans.MultiRegion: reconstruct_global_field

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    u = reconstruct_global_field(sim.model.velocities.u)
    w = reconstruct_global_field(sim.model.velocities.w)

    intw  = Array(interior(w))
    max_w = findmax(intw)

    mw = max_w[1]
    iw = max_w[2]

    @info @sprintf("Time: % 12s, iteration: %d, max(|u|): %.2e ms⁻¹, wmax: %.2e , loc: (%d, %d, %d), wall time: %s", 
                    prettytime(sim.model.clock.time),
                    sim.model.clock.iteration, maximum(abs, u), mw, iw[1], iw[2], iw[3], 
                    prettytime(wall_time))

    start_time[1] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = IterationInterval(3600),
                                                        prefix = output_prefix * "_checkpoint",
                                                        overwrite_existing = true)

@info "Running with Δt = $(prettytime(simulation.Δt))"

run!(simulation, pickup = pickup_file)

@info """

    Simulation took $(prettytime(simulation.run_wall_time))
    Free surface: $(typeof(model.free_surface).name.wrapper)
    Time step: $(prettytime(Δt))
"""


