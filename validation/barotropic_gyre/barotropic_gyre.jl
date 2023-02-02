# # Barotropic gyre

using Oceananigans
using Oceananigans.Grids

using Oceananigans.Coriolis: HydrostaticSphericalCoriolis

using Oceananigans.Advection:
    EnergyConservingScheme,
    EnstrophyConservingScheme

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    VectorInvariant,
    ExplicitFreeSurface,
    ImplicitFreeSurface


using Oceananigans.Utils: prettytime, hours, day, days, years
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, GridFittedBottom

using Statistics
using JLD2
using Printf

using Oceananigans.Distributed

Nx = 120
Ny = 120

using MPI
MPI.Init()

comm   = MPI.COMM_WORLD
rank   = MPI.Comm_rank(comm)
Nranks = MPI.Comm_size(comm)

topo = (Periodic, Bounded, Bounded)
arch = MultiArch(CPU(); topology = topo, ranks=(Nranks, 1, 1))

# A spherical domain
underlying_grid = LatitudeLongitudeGrid(arch, size = (Nx, Ny, 1),
                                        longitude = (-30, 30),
                                        latitude = (15, 75),
                                        z = (-4000, 0),
                                        halo = (2, 2, 2),
                                        topology = topo)


nx, ny, nz = size(underlying_grid)
bathymetry = zeros(nx, ny) .- 4000
view(bathymetry, 10:15, 43:47) .= 0

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry) )

using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Grids: min_Δx, min_Δy

Δt         = 1200 
CFL        = 0.7
wave_speed = sqrt(g_Earth * grid.Lz)
Δg         = 1 / sqrt(1 / min_Δx(grid)^2 + 1 / min_Δy(grid)^2)
@show substeps = Int(ceil(2 * Δt / (CFL / wave_speed * Δg)))

using Oceananigans.Models.HydrostaticFreeSurfaceModels: AdamsBashforth3Scheme

free_surface = SplitExplicitFreeSurface(; substeps)
# free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1)

coriolis = HydrostaticSphericalCoriolis(scheme = EnstrophyConservingScheme())

@show surface_wind_stress_parameters = (τ₀ = 1e-4,
                                        Lφ = grid.Ly,
                                        φ₀ = 15)

@inline surface_wind_stress(λ, φ, t, p) = p.τ₀ * cos(2π * (φ - p.φ₀) / p.Lφ)

surface_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress,
                                               parameters = surface_wind_stress_parameters)

μ = 1 / 60days

@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag,
                                         discrete_form = true,
                                         parameters = μ)

v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag,
                                         discrete_form = true,
                                         parameters = μ)

u_bcs = FieldBoundaryConditions(top = surface_wind_stress_bc,
                                bottom = u_bottom_drag_bc)

v_bcs = FieldBoundaryConditions(bottom = v_bottom_drag_bc)

@show const νh₀ = 5e3 * (60 / grid.Nx)^2

@inline νh(λ, φ, z, t) = νh₀ * cos(π * φ / 180)

variable_horizontal_diffusivity = HorizontalScalarDiffusivity(ν = νh)
constant_horizontal_diffusivity = HorizontalScalarDiffusivity(ν = νh₀)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    momentum_advection = VectorInvariant(),
                                    free_surface = free_surface,
                                    coriolis = coriolis,
                                    boundary_conditions = (u=u_bcs, v=v_bcs),
                                    closure = constant_horizontal_diffusivity,
                                    tracers = nothing,
                                    buoyancy = nothing)

simulation = Simulation(model; Δt, stop_iteration = 10000)
            
mutable struct Progress
    interval_start_time :: Float64
end

function (p::Progress)(sim)
    wall_time = (time_ns() - p.interval_start_time) * 1e-9

    @info @sprintf("Time: %s, iteration: %d, max(u): %.2e m s⁻¹, wall time: %s",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   maximum(sim.model.velocities.u),
                   prettytime(wall_time))

    p.interval_start_time = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(Progress(time_ns()), IterationInterval(20))

indices = (:, :, 1)

ηarr = Vector{Field}(undef, Int(simulation.stop_iteration))
varr = Vector{Field}(undef, Int(simulation.stop_iteration))
uarr = Vector{Field}(undef, Int(simulation.stop_iteration))

save_η(sim) = sim.model.clock.iteration > 0 ? ηarr[sim.model.clock.iteration] = deepcopy(sim.model.free_surface.η) : nothing
save_v(sim) = sim.model.clock.iteration > 0 ? varr[sim.model.clock.iteration] = deepcopy(sim.model.velocities.v)   : nothing
save_u(sim) = sim.model.clock.iteration > 0 ? uarr[sim.model.clock.iteration] = deepcopy(sim.model.velocities.u)   : nothing

simulation.callbacks[:save_η]   = Callback(save_η, IterationInterval(1))
simulation.callbacks[:save_v]   = Callback(save_v, IterationInterval(1))
simulation.callbacks[:save_u]   = Callback(save_u, IterationInterval(1))

run!(simulation)

jldsave("variables_rank$(rank).jld2", varr = varr, ηarr = ηarr, uarr = uarr)

#####
##### Animation!
#####

# include("visualize_barotropic_gyre.jl")

# visualize_barotropic_gyre(simulation.output_writers[:fields])
