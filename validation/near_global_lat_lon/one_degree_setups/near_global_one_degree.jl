using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using JLD2
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

include("one_degree_artifacts.jl")
bathymetry_path = download_bathymetry()
bathymetry = jldopen(bathymetry_path)["bathymetry"]

include("one_degree_interface_heights.jl")
z = one_degree_interface_heights()

# A spherical domain
@show underlying_grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nz), halo = (4, 4, 4),
                                              latitude, z,
                                              longitude = (-180, 180),
                                              precompute_metrics = true)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

using Oceananigans.Operators: Δx, Δy
using Oceananigans.TurbulenceClosures

@inline ν₄(i, j, k, grid, lx, ly, lz) = (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2))^2 / 5days

horizontal_diffusivity = HorizontalScalarDiffusivity(ν=1e1, κ=1e1)
background_vertical_diffusivity = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1e-2, κ=1e-4)
dynamic_vertical_diffusivity = RiBasedVerticalDiffusivity()
biharmonic_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=ν₄, discrete_form=true)

κ_skew = 1000.0      # [m² s⁻¹] skew diffusivity
κ_symmetric = 1000.0 # [m² s⁻¹] symmetric diffusivity
gerdes_koberle_willebrand_tapering = FluxTapering(1e-2)
gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = κ_skew,
                                                                κ_symmetric = κ_symmetric,
                                                                slope_limiter = gerdes_koberle_willebrand_tapering)

free_surface = ImplicitFreeSurface(solver_method = :HeptadiagonalIterativeSolver)
                                   # preconditioner_method = :SparseInverse,
                                   # preconditioner_settings = (ε=0.01, nzrel=10))

equation_of_state = LinearEquationOfState()
buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=35.0)

closures = (horizontal_diffusivity, background_vertical_diffusivity, dynamic_vertical_diffusivity, gent_mcwilliams_diffusivity)
# closures = (horizontal_diffusivity, vertical_diffusivity, convective_adjustment, biharmonic_viscosity, gent_mcwilliams_diffusivity)

#=
@inline reference_temperature(φ) = 30.0 * cos(2π * φ / 180)
@inline temperature_relaxation(λ, φ, t, T, λ) = λ * (T - reference_temperature(φ))
T_top_bc = FluxBoundaryCondition(temperature_relaxation, field_dependencies=:T, parameters=30days)
T_bcs = FieldBoundaryConditions(top=T_top_bc)

@inline surface_stress_x(λ, φ, t, p) = p.τ₀ (1 + exp(-φ^2 / 200) - (p.τ₀ + p.τˢ) * exp(-(φ + 50)^2 / 200) -
                                                                   (p.τ₀ + p.τᴺ) * exp(-(φ - 50)^2 / 200)

u_top_bc = FluxBoundaryCondition(surface_stress, parameters=(τ₀=6e-5, τˢ=2e-4, τᴺ=5e-5))
=#

model = HydrostaticFreeSurfaceModel(; grid, free_surface, buoyancy,
                                    momentum_advection = VectorInvariant(),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    tracers = :T,
                                    closure = closures,
                                    tracer_advection = WENO5(grid=underlying_grid))

simulation = Simulation(model; Δt=20minutes, stop_iteration=10)

wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    @info string("Iteration: ", iteration(sim),
                 ", time: ", prettytime(sim),
                 ", wall time: ", prettytime(elapsed))

    wall_clock[] = time_ns()

    return nothing
end

simulation.callbacks[:p] = Callback(progress, IterationInterval(1))

run!(simulation)

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
