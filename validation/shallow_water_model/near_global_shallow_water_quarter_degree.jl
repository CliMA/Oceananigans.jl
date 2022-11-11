using Statistics
using JLD2
using Printf
using Oceananigans
using Oceananigans.Units

using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: multi_region_object_from_array
using Oceananigans.Fields: interpolate, Field
using Oceananigans.Architectures: arch_array
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.BoundaryConditions
using Oceananigans.Grids: boundary_node, inactive_node, peripheral_node
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary 
using CUDA: @allowscalar, device!
using Oceananigans.Operators
using Oceananigans.Operators: Δzᵃᵃᶜ
using Oceananigans: prognostic_fields

@inline function visualize(field, lev, dims)
    (dims == 1) && (idx = (lev, :, :))
    (dims == 2) && (idx = (:, lev, :))
    (dims == 3) && (idx = (:, :, lev))

    r = deepcopy(Array(interior(field)))[idx...]
    r[ r.==0 ] .= NaN
    return r
end

device!(2)

#####
##### Grid
#####

arch = GPU()
reference_density = 1029

latitude = (-75, 75)

# 0.25 degree resolution
Nx = 1440
Ny = 600

const Nyears  = 2
const Nmonths = 12
const thirty_days = 30days

output_prefix = "near_global_shallow_water_$(Nx)_$(Ny)"
pickup_file   = false

#####
##### Load forcing files and inital conditions from ECCO version 4
##### https://ecco.jpl.nasa.gov/drive/files
##### Bathymetry is interpolated from ETOPO1 https://www.ngdc.noaa.gov/mgg/global/
#####

using DataDeps

path = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/ss/new_hydrostatic_data_after_cleared_bugs/quarter_degree_near_global_input_data/"

datanames = ["tau_x-1440x600-latitude-75",
             "tau_y-1440x600-latitude-75",
             "bathymetry-1440x600"]

dh = DataDep("quarter_degree_near_global_lat_lon",
    "Forcing data for global latitude longitude simulation",
    [path * data * ".jld2" for data in datanames]
)

DataDeps.register(dh)

datadep"quarter_degree_near_global_lat_lon"

files = [:file_tau_x, :file_tau_y, :file_bathymetry]
for (data, file) in zip(datanames, files)
    datadep_path = @datadep_str "quarter_degree_near_global_lat_lon/" * data * ".jld2"
    @eval $file = jldopen($datadep_path)
end

τˣ = zeros(Nx, Ny, Nmonths)
τʸ = zeros(Nx, Ny, Nmonths)

# Files contain 1 year (1992) of 12 monthly averages
τˣ = file_tau_x["field"] ./ reference_density
τʸ = file_tau_y["field"] ./ reference_density
τˣ = arch_array(arch, τˣ)
τʸ = arch_array(arch, τʸ)

bat = file_bathymetry["bathymetry"]
boundary = Int.(bat .> 0)
bat[ bat .> 0 ] .= 0 
bat = -bat

# A spherical domain
@show underlying_grid = LatitudeLongitudeGrid(arch,
                                              size = (Nx, Ny),
                                              longitude = (-180, 180),
                                              latitude = latitude,
                                              halo = (4, 4),
                                              topology = (Periodic, Bounded, Flat),
                                              precompute_metrics = true)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(boundary))

#####
##### Boundary conditions / time-dependent fluxes
#####

@inline current_time_index(time, tot_months)     = mod(unsafe_trunc(Int32, time / thirty_days), tot_months) + 1
@inline next_time_index(time, tot_months)        = mod(unsafe_trunc(Int32, time / thirty_days) + 1, tot_months) + 1
@inline cyclic_interpolate(u₁::Number, u₂, time) = u₁ + mod(time / thirty_days, 1) * (u₂ - u₁)

using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ

@inline function boundary_stress_u(i, j, k, grid, clock, fields, p)
    time = clock.time
    n₁ = current_time_index(time, Nmonths)
    n₂ = next_time_index(time, Nmonths)

    @inbounds begin
        τ₁ = p.τ[i, j, n₁]
        τ₂ = p.τ[i, j, n₂]
    end

    h_int = ℑxᶠᵃᵃ(i, j, k, grid, fields.h)
    if h_int > 0
        return (cyclic_interpolate(τ₁, τ₂, time) - p.μ * fields.u[i, j, k]) / h_int
    else
        return 0.0
    end
end

@inline function boundary_stress_v(i, j, k, grid, clock, fields, p)
    time = clock.time
    n₁ = current_time_index(time, Nmonths)
    n₂ = next_time_index(time, Nmonths)

    @inbounds begin
        τ₁ = p.τ[i, j, n₁]
        τ₂ = p.τ[i, j, n₂]
    end

    h_int =  ℑyᵃᶠᵃ(i, j, k, grid, fields.h)
    if h_int > 0
        return (cyclic_interpolate(τ₁, τ₂, time) - p.μ * fields.v[i, j, k]) / h_int
    else
        return 0.0
    end
end

# Linear bottom drag:
μ = 0.001 # ms⁻¹

Fu = Forcing(boundary_stress_u, discrete_form = true, parameters = (μ = μ, τ = τˣ))
Fv = Forcing(boundary_stress_v, discrete_form = true, parameters = (μ = μ, τ = τʸ))

using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Oceananigans.TurbulenceClosures: HorizontalDivergenceFormulation

νh = 0e+1

using Oceananigans.Operators: Δx, Δy
using Oceananigans.TurbulenceClosures

@inline νhb(i, j, k, grid, lx, ly, lz) = (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2 ))^2 / 5days

horizontal_diffusivity = HorizontalScalarDiffusivity(ν=νh)
biharmonic_viscosity   = HorizontalScalarBiharmonicDiffusivity(ν=νhb, discrete_form=true)

model = ShallowWaterModel(grid = grid,
			              gravitational_acceleration = 9.8055,
                          momentum_advection = VectorInvariant(scheme=WENO(), stencil=VorticityStencil()),
                          mass_advection = WENO(),
                          bathymetry = bat,
                          coriolis = HydrostaticSphericalCoriolis(),
                          forcing = (u=Fu, v=Fv),
			              formulation = VectorInvariantFormulation())

#####
##### Initial condition:
#####

h_init = deepcopy(1e1 .+ maximum(bat) .- bat) 
set!(model, h=h_init)
fill_halo_regions!(model.solution.h)

@info "model initialized"

#####
##### Simulation setup
#####

Δt = 20seconds 

simulation = Simulation(model, Δt = Δt, stop_time = Nyears*years)

start_time = [time_ns()]

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    u = sim.model.solution.u
    h = sim.model.solution.h

    @info @sprintf("Time: % 12s, iteration: %d, max(|u|): %.2e ms⁻¹, min(h): %.2e ms⁻¹, wall time: %s",
                    prettytime(sim.model.clock.time),
                    sim.model.clock.iteration, maximum(abs, u), minimum(h),
                    prettytime(wall_time))

    start_time[1] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

u, v, h = model.solution

ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid; computed_dependencies=(u, v)); 
ζ = Field(ζ_op)
compute!(ζ)

save_interval = 1days

simulation.output_writers[:surface_fields] = JLD2OutputWriter(model, (; u, v, h, ζ),
                                                            schedule = TimeInterval(save_interval),
                                                            filename = output_prefix * "_surface",
                                                            overwrite_existing = true)

# Let's go!
@info "Running with Δt = $(prettytime(simulation.Δt))"
run!(simulation)
