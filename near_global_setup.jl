using Statistics
using JLD2
using Printf
using Oceananigans
using Oceananigans.Units

using Oceananigans.Grids
using Oceananigans.Operators: Δzᵃᵃᶜ
using Oceananigans.ImmersedBoundaries: peripheral_node, inactive_node
using Oceananigans.Architectures: on_architecture
using Oceananigans.OutputReaders
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization
using Oceananigans.BoundaryConditions
using CUDA: @allowscalar, device!

device!(3)

#####
##### Grid and boundary data
#####

arch = GPU()

bathymetry_link = "https://www.dropbox.com/scl/fi/9djbqt3xbrxxi8lj9ajqb/quarter_degree_bathymetry.jld2?rlkey=tvl4f6wh72r9qtqu66swjrg77&st=uh9zaygu&dl=1"
data_link = "https://www.dropbox.com/scl/fi/zj84wzr97m49h7hkr0xvr/quarter_degree_data.jld2?rlkey=1g7fat1f91l7kef6p3u1v721w&st=puqfbwde&dl=1"
# init_link = "https://www.dropbox.com/scl/fi/9ayda5axbq92wy234uhfn/quarter_initial_conditions.jld2?rlkey=etk4j29zsdg71d5y6obpyh5vk&st=u0qcwxrx&dl=1"

isfile("quarter_degree_bathymetry.jld2")  || download(bathymetry_link, "./quarter_degree_bathymetry.jld2")
isfile("quarter_degree_data.jld2")        || download(data_link, "./quarter_degree_data.jld2")
# isfile("quarter_initial_conditions.jld2") || download(init_link, "./quarter_initial_conditions.jld2")

τx = FieldTimeSeries("quarter_degree_data.jld2", "τx"; architecture = arch, time_indexing = Cyclical())
# τy = FieldTimeSeries("quarter_degree_data.jld2", "τy"; architecture = arch, time_indexing = Cyclical())
# T★ = FieldTimeSeries("quarter_degree_data.jld2", "Tr"; architecture = arch, time_indexing = Cyclical())
# S★ = FieldTimeSeries("quarter_degree_data.jld2", "Sr"; architecture = arch, time_indexing = Cyclical())

underlying_grid = τx.grid
bottom_height = jldopen("quarter_degree_bathymetry.jld2")["bathymetry"]
bottom_height = on_architecture(arch, bottom_height)

r_faces = Grids.cpu_face_constructor_z(underlying_grid)
z_faces = r_faces 

underlying_grid = LatitudeLongitudeGrid(arch, size = size(underlying_grid), halo = (7, 7, 5), latitude = (-75, 75), 
                                        longitude  = (0, 360), z = z_faces)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map = true)

#####
##### Physics and model setup
#####

const Lz = grid.Lz
const h  = 100meters 

@inline exponential_profile(z) = (exp(z / h) - exp( - Lz / h)) / (1 - exp( - Lz / h))

@inline νz(x, y, z, t) = (5e-3 - 5e-4) * exponential_profile(z) + 5e-4
κz = 1e-5

vertical_diffusivity  = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=νz, κ=κz)
convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), convective_κz = 1.0)

#####
##### Boundary conditions / time-dependent fluxes
#####

#=
# remember the sign convention!
@inline fts_flux_boundary(i, j, grid, clock, fields, fts) = @inbounds - fts[i, j, 1, Time(clock.time)] / 1000.0
    
u_wind_stress_bc = FluxBoundaryCondition(fts_flux_boundary; discrete_form = true, parameters = τx)
v_wind_stress_bc = FluxBoundaryCondition(fts_flux_boundary; discrete_form = true, parameters = τy)

# Linear bottom drag:
μ = 0.001 # ms⁻¹

@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

# Keep a constant linear drag parameter independent on vertical level
@inline u_immersed_bottom_drag(i, j, k, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, k]
@inline v_immersed_bottom_drag(i, j, k, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, k]

u_immersed_bc = ImmersedBoundaryCondition(bottom = FluxBoundaryCondition(u_immersed_bottom_drag, discrete_form = true, parameters = μ))
v_immersed_bc = ImmersedBoundaryCondition(bottom = FluxBoundaryCondition(v_immersed_bottom_drag, discrete_form = true, parameters = μ))

u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag, discrete_form = true, parameters = μ)
v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag, discrete_form = true, parameters = μ)

struct Temperature end
struct Salinity end

Base.getindex(nt::NamedTuple, ::Temperature, idx...) = nt.T[idx...]
Base.getindex(nt::NamedTuple, ::Salinity,    idx...) = nt.S[idx...]

@inline function surface_relaxation(i, j, grid, clock, fields, p)
    time = clock.time

    @inbounds begin
        S = fields[p.variable, i, j, grid.Nz]
        R = p[p.variable, i, j, 1, Time(time)]
    end

    return p.λ * (S - R)
end

Δr_top = @allowscalar Oceananigans.Operators.Δzᶜᶜᶜ(1, 1, grid.Nz, grid)

T_surface_relaxation_bc = FluxBoundaryCondition(surface_relaxation,
                                                discrete_form = true,
                                                parameters = (λ = Δr_top/7days, T = T★, variable = Temperature()))

S_surface_relaxation_bc = FluxBoundaryCondition(surface_relaxation,
                                                discrete_form = true,
                                                parameters = (λ = Δr_top/7days, S = S★, variable = Salinity()))

u_bcs = FieldBoundaryConditions(top = u_wind_stress_bc, bottom = u_bottom_drag_bc, immersed = u_immersed_bc)
v_bcs = FieldBoundaryConditions(top = v_wind_stress_bc, bottom = v_bottom_drag_bc, immersed = v_immersed_bc)
T_bcs = FieldBoundaryConditions(top = T_surface_relaxation_bc)
S_bcs = FieldBoundaryConditions(top = S_surface_relaxation_bc)
=#

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState())
free_surface = SplitExplicitFreeSurface(grid; cfl = 0.75)
timestepper = Symbol(get(ENV, "TIMESTEPPER", "QuasiAdamsBashforth2"))

model = HydrostaticFreeSurfaceModel(; grid = grid,
                                      free_surface = free_surface,
                                      momentum_advection = WENOVectorInvariant(),
                                      coriolis = HydrostaticSphericalCoriolis(),
                                      buoyancy = buoyancy,
                                      timestepper,
                                      tracers = (:T, :S),
                                      closure = (vertical_diffusivity, convective_adjustment),
                                      # boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs),
                                      tracer_advection = WENO(grid; order = 7))

#####
##### Initial condition:
#####
#=
u, v, w = model.velocities
η = model.free_surface.η
T = model.tracers.T
S = model.tracers.S

@info "Reading initial conditions"
T_init = jldopen("quarter_initial_conditions.jld2")["T"]
S_init = jldopen("quarter_initial_conditions.jld2")["S"]

set!(model, T=T_init, S=S_init)
fill_halo_regions!(T)
fill_halo_regions!(S)

@info "model initialized"
=#
#####
##### Simulation setup
#####

Δt = timestepper == :QuasiAdamsBashforth2 ? 6minutes : 18minutes

simulation = Simulation(model, Δt = Δt, stop_iteration = 200)

start_time = [time_ns()]

using Oceananigans.Utils

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    u = sim.model.velocities.u
    w = interior(sim.model.velocities.w, :, :, grid.Nz+1)
    η = sim.model.free_surface.η

    @info @sprintf("Time: % 12s, Δt: %s, iteration: %d, max(|u|): %.2e ms⁻¹, max(|w|): %.2e ms⁻¹, wall time: %s",
                   prettytime(sim.model.clock.time), prettytime(sim.Δt),
                    sim.model.clock.iteration, maximum(abs, u), maximum(abs, w),
                    prettytime(wall_time))

    start_time[1] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

#=
wizard = TimeStepWizard(cfl = 0.75, max_change = 1.5, max_Δt = 60minutes)
simulation.callbacks[:wizard]   = Callback(wizard,   IterationInterval(10))

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
η = model.free_surface.η

output_fields = (; u, v, T, S)

simulation.output_writers[:surface] = JLD2OutputWriter(model, output_fields,
                                                       schedule = TimeInterval(1days),
                                                       indices  = (:, :, grid.Nz),
                                                       filename = "near_global_surface_rk3.jld2")

simulation.output_writers[:free_surface] = JLD2OutputWriter(model, (; η),
                                                            schedule = TimeInterval(1days),
                                                            filename = "near_global_free_surface_rk3.jld2")

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(10days),
                                                        prefix = "near_global_checkpoint_rk3",
                                                        overwrite_existing = true)

# Let's goo!
@info "Running with Δt = $(prettytime(simulation.Δt))"
=#

run!(simulation)

@info """
    Simulation took $(prettytime(simulation.run_wall_time))
    Free surface: $(typeof(model.free_surface).name.wrapper)
    Time step: $(prettytime(Δt))
"""
