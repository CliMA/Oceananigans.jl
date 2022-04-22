using Oceananigans
using SeawaterPolynomials.TEOS10

arch = CPU()
reference_density = 1029
latitude = (-75, 75)

# 1 degree resolution
Nx = 360
Ny = 150
Nz = 48

const Nyears = 60.0
const Nmonths = 12
const thirty_days = 30days

output_prefix = "near_global_lat_lon_$(Nx)_$(Ny)_$(Nz)"

bathymetry = jldopen("bathymetry-360x150-latitude-75.0.jld2")["bathymetry"]

τˣ = zeros(Nx, Ny, Nmonths)
τʸ = zeros(Nx, Ny, Nmonths)
T★ = zeros(Nx, Ny, Nmonths)
S★ = zeros(Nx, Ny, Nmonths)

# Files contain 1 year (1992) of 12 monthly averages
τˣ = jldopen("boundary_conditions-1degree.jld2")["τˣ"] ./ reference_density
τʸ = jldopen("boundary_conditions-1degree.jld2")["τˣ"] ./ reference_density
T★ = jldopen("boundary_conditions-1degree.jld2")["Tˢ"]
S★ = jldopen("boundary_conditions-1degree.jld2")["Sˢ"]

# Remember the convention!! On the surface a negative flux increases a positive decreases
bathymetry = arch_array(arch, bathymetry)
τˣ = arch_array(arch, -τˣ)
τʸ = arch_array(arch, -τʸ)

target_sea_surface_temperature = T★ = arch_array(arch, T★)
target_sea_surface_salinity = S★ = arch_array(arch, S★)

# Stretched faces taken from ECCO Version 4 (50 levels in the vertical)
z = jldopen("zgrid.jld2")["z"][5:end-4]

# A spherical domain
@show underlying_grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nz), halo = (4, 4, 4),
                                              latitude, 
                                              longitude = (-180, 180),
                                              z,
                                              precompute_metrics = true)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

#####
##### Physics and model setup
#####

νh = 1e+1
νz = 5e-3
κh = 1e+1
κz = 1e-4

κ_skew = 1000.0 * 1e-3       # [m² s⁻¹] skew diffusivity
κ_symmetric = 900.0 * 1e-3   # [m² s⁻¹] symmetric diffusivity

using Oceananigans.Operators: Δx, Δy
using Oceananigans.TurbulenceClosures

@inline νhb(i, j, k, grid, lx, ly, lz) = (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))^2 / 5days

horizontal_diffusivity = HorizontalScalarDiffusivity(ν=νh, κ=κh)
vertical_diffusivity = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=νz, κ=κz)
convective_adjustment = RiBasedVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), κ₀=1.0)
biharmonic_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=νhb, discrete_form=true)

gerdes_koberle_willebrand_tapering = FluxTapering(1e-2)
gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = κ_skew,
                                                                κ_symmetric = κ_symmetric,
                                                                slope_limiter = gerdes_koberle_willebrand_tapering)

#####
##### Boundary conditions / time-dependent fluxes 
#####

@inline current_time_index(time, tot_months) = mod(unsafe_trunc(Int32, time / thirty_days), tot_months) + 1
@inline next_time_index(time, tot_months) = mod(unsafe_trunc(Int32, time / thirty_days) + 1, tot_months) + 1
@inline cyclic_interpolate(u₁::Number, u₂, time) = u₁ + mod(time / thirty_days, 1) * (u₂ - u₁)

Δz_top = @allowscalar Δzᵃᵃᶜ(1, 1, grid.Nz, grid.grid)
Δz_bottom = @allowscalar Δzᵃᵃᶜ(1, 1, 1, grid.grid)

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

u_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress, discrete_form=true, parameters=τˣ)
v_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress, discrete_form=true, parameters=τʸ)

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

T_surface_relaxation_bc = FluxBoundaryCondition(surface_temperature_relaxation,
                                                discrete_form = true,
                                                parameters = (λ=Δz_top / 7days, T★=target_sea_surface_temperature))

u_bcs = FieldBoundaryConditions(top=u_wind_stress_bc)
v_bcs = FieldBoundaryConditions(top=v_wind_stress_bc)
T_bcs = FieldBoundaryConditions(top=T_surface_relaxation_bc)

free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver, preconditioner_method=:SparseInverse, preconditioner_settings=(ε=0.01, nzrel=10))

equation_of_state = LinearEquationOfState()
buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=35.0)

closures = (horizontal_diffusivity, vertical_diffusivity, convective_adjustment, biharmonic_viscosity, gent_mcwilliams_diffusivity)

model = HydrostaticFreeSurfaceModel(; grid, free_surface, buoyancy,
                                    momentum_advection = VectorInvariant(),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    tracers = :T,
                                    closure = closures,
                                    boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs),
                                    tracer_advection = WENO5(grid=underlying_grid))

simulation = Simulation(model; Δt=20minutes, stop_iteration=10)
run!(simulation)

#=
#####
##### Initial condition:
#####

u, v, w = model.velocities
η = model.free_surface.η
T = model.tracers.T
S = model.tracers.S

file_init = jldopen("initial_conditions-1degree.jld2")

@info "Reading initial conditions"
T_init = file_init["T"]
S_init = file_init["S"]

set!(model, T=T_init, S=S_init)

#####
##### Simulation setup
#####


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

#=
u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
η = model.free_surface.η

output_fields = (; u, v, T, S, η)
save_interval = 5days

u2 = Field(u * u)
v2 = Field(v * v)
w2 = Field(w * w)
η2 = Field(η * η)
T2 = Field(T * T)

outputs = (; u, v, T, S, η)
average_outputs = (; u, v, T, S, η, u2, v2, T2, η2)

simulation.output_writers[:surface_fields] = JLD2OutputWriter(model, (; u, v, T, S, η),
                                                              schedule = TimeInterval(save_interval),
                                                              prefix = output_prefix * "_surface",
                                                              indices = (:, :, grid.Nz), 
                                                              force = true)

simulation.output_writers[:averages] = JLD2OutputWriter(model, average_outputs,
                                                              schedule = AveragedTimeInterval(4*30days, window=4*30days),
                                                              prefix = output_prefix * "_averages",
                                                              force = true)


simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(6*30days),
                                                        prefix = output_prefix * "_checkpoint",
                                                        force = true)
=#
# Let's goo!
@info "Running with Δt = $(prettytime(simulation.Δt))"

if load_initial_condition
    @info "load in initial condition from " * ic_filepath
    jlfile = jldopen(ic_filepath)
    interior(simulation.model.velocities.u) .= CuArray(jlfile["velocities"]["u"])
    interior(simulation.model.velocities.v) .= CuArray(jlfile["velocities"]["v"])
    interior(simulation.model.velocities.w) .= CuArray(jlfile["velocities"]["w"])
    interior(simulation.model.tracers.T) .= CuArray(jlfile["tracers"]["T"])
    interior(simulation.model.tracers.S) .= CuArray(jlfile["tracers"]["S"])
    interior(simulation.model.free_surface.η) .= CuArray(jlfile["free_surface"]["eta"])
    close(jlfile)
end

run!(simulation)

@info """

    Simulation took $(prettytime(simulation.run_wall_time))
    Free surface: $(typeof(model.free_surface).name.wrapper)
    Time step: $(prettytime(Δt))
"""



rm(qs_filepath, force=true)
jlfile = jldopen(qs_filepath, "a+")
JLD2.Group(jlfile, "velocities")
JLD2.Group(jlfile, "tracers")
JLD2.Group(jlfile, "free_surface") # don't forget free surface

jlfile["velocities"]["u"] = Array(interior(simulation.model.velocities.u))
jlfile["velocities"]["v"] = Array(interior(simulation.model.velocities.v))
jlfile["velocities"]["w"] = Array(interior(simulation.model.velocities.w))
jlfile["tracers"]["T"] = Array(interior(simulation.model.tracers.T))
jlfile["tracers"]["S"] = Array(interior(simulation.model.tracers.S))
jlfile["free_surface"]["eta"] = Array(interior(simulation.model.free_surface.η))
close(jlfile)
