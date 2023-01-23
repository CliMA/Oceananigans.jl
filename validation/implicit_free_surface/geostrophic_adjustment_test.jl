using Revise
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using Oceananigans.MultiRegion
using Statistics
using Printf
using LinearAlgebra, SparseArrays
using Oceananigans.Solvers: constructors, unpack_constructors

function geostrophic_adjustment_simulation(free_surface, topology, multi_region; arch = Oceananigans.CPU())

    Lh = 100kilometers
    Lz = 400meters

    grid = RectilinearGrid(arch,
        size = (80, 3, 1),
        x = (0, Lh), y = (0, Lh), z = (-Lz, 0), halo = (4, 4, 4),
        topology = topology)

    bottom(x, y) = x > 80kilometers && x < 90kilometers ? 0.0 : -500meters

    # grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

    if multi_region
        if arch isa GPU
            devices = (0, 1)
        else
            devices = nothing
        end
        mrg = MultiRegionGrid(grid, partition = XPartition(2), devices = devices)
    else
        mrg = grid
    end

    @show mrg

    coriolis = FPlane(f = 1e-4)

    model = HydrostaticFreeSurfaceModel(grid = mrg,
                                        coriolis = coriolis,
                                        free_surface = free_surface, 
                                        tracer_advection = WENO(), 
                                        momentum_advection = WENO())

    gaussian(x, L) = exp(-x^2 / 2L^2)

    U = 0.1 # geostrophic velocity
    L = grid.Lx / 40 # gaussian width
    x₀ = grid.Lx / 4 # gaussian center

    vᵍ(x, y, z) = -U * (x - x₀) / L * gaussian(x - x₀, L)

    g = model.free_surface.gravitational_acceleration
    η = model.free_surface.η

    η₀ = coriolis.f * U * L / g # geostrophic free surface amplitude

    ηᵍ(x) = η₀ * gaussian(x - x₀, L)

    ηⁱ(x, y, z) = 2 * ηᵍ(x)
    
    set!(model, v = vᵍ)
    set!(model.free_surface.η, ηⁱ)

    gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
    wave_propagation_time_scale = model.grid.Δxᶜᵃᵃ / gravity_wave_speed
    simulation = Simulation(model, Δt = 10wave_propagation_time_scale, stop_iteration = 600)

    return simulation
end

using Oceananigans.MultiRegion: reconstruct_global_field

function run_and_analyze(simulation)
    η       = simulation.model.free_surface.η
    u, v, w = simulation.model.velocities
    Δt      = simulation.Δt

    f = simulation.model.free_surface

    if f isa SplitExplicitFreeSurface
        solver_method = "SplitExplicitFreeSurfaceSolver"
    elseif f isa ExplicitFreeSurface
        solver_method = "ExplicitFreeSurfaceSolver"
    else
        solver_method = string(simulation.model.free_surface.solver_method)
    end
    
    ηarr = Vector{Field}(undef, Int(simulation.stop_iteration))
    varr = Vector{Field}(undef, Int(simulation.stop_iteration))
    uarr = Vector{Field}(undef, Int(simulation.stop_iteration))

    save_η(sim) = sim.model.clock.iteration > 0 ? ηarr[sim.model.clock.iteration] = deepcopy(sim.model.free_surface.η) : nothing
    save_v(sim) = sim.model.clock.iteration > 0 ? varr[sim.model.clock.iteration] = deepcopy(sim.model.velocities.v)   : nothing
    save_u(sim) = sim.model.clock.iteration > 0 ? uarr[sim.model.clock.iteration] = deepcopy(sim.model.velocities.u)   : nothing

    progress_message(sim) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
        100 * sim.model.clock.time / sim.stop_time, sim.model.clock.iteration,
        sim.model.clock.time, maximum(abs, sim.model.velocities.u))

    simulation.callbacks[:save_η]   = Callback(save_η, IterationInterval(1))
    simulation.callbacks[:save_v]   = Callback(save_v, IterationInterval(1))
    simulation.callbacks[:save_u]   = Callback(save_u, IterationInterval(1))
    simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

    run!(simulation)

    return (ηarr, varr, uarr)
end

# fft_based_free_surface = ImplicitFreeSurface()
pcg_free_surface           = ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient);
matrix_free_surface        = ImplicitFreeSurface(solver_method = :HeptadiagonalIterativeSolver);
splitexplicit_free_surface = SplitExplicitFreeSurface(substeps = 20)
explicit_free_surface      = ExplicitFreeSurface()

topology_types = [(Bounded, Periodic, Bounded), (Periodic, Periodic, Bounded)]
topology_types = [topology_types[1]]

archs = [Oceananigans.CPU()] #, Oceananigans.GPU()]
archs = [archs[1]]

free_surfaces = [splitexplicit_free_surface, matrix_free_surface] #, explicit_free_surface] #, matrix_free_surface];

simulations = [geostrophic_adjustment_simulation(free_surface, topology_type, multi_region, arch = arch) 
              for (free_surface, multi_region) in zip(free_surfaces, (false, false, false)), 
              topology_type in topology_types, 
              arch in archs];

data = [run_and_analyze(sim) for sim in simulations];

using GLMakie
using JLD2

topology = topology_types[1]
arch = CPU()
Lh = 100kilometers
Lz = 400meters

grid = RectilinearGrid(arch,
        size = (80, 3, 1),
        x = (0, Lh), y = (0, Lh), z = (-Lz, 0),
        topology = topology)

x  = grid.xᶜᵃᵃ[1:grid.Nx]
xf = grid.xᶠᵃᵃ[1:grid.Nx+1]
y  = grid.yᵃᶜᵃ[1:grid.Ny]

iter = Observable(1) # Node or Observable depending on Makie version
mid = Int(floor(grid.Ny / 2))

η0 = interior(data[1][1][1], :, mid, 1)
η1 = @lift(interior(data[1][1][$iter], :, mid, 1))
η2 = @lift(interior(data[2][1][$iter], :, mid, 1))
# η3 = @lift(interior(data[3][1][$iter], :, mid, 1))

v01 = interior(data[1][2][1], :, mid, 1)
v02 = interior(data[2][2][1], :, mid, 1)
v1 = @lift(interior(data[1][2][$iter], :, mid, 1))
v2 = @lift(interior(data[2][2][$iter], :, mid, 1))
# v3 = @lift(interior(data[3][2][$iter], :, mid, 1))

u01 = interior(data[1][3][1], :, mid, 1)
u02 = interior(data[2][3][1], :, mid, 1)
u1 = @lift(interior(data[1][3][$iter], :, mid, 1))
u2 = @lift(interior(data[2][3][$iter], :, mid, 1))
# u3 = @lift(interior(data[3][3][$iter], :, mid, 1))

fig = Figure(resolution = (1400, 1000))
options = (; ylabelsize = 22,
    xlabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
    xticksize = 10, ytickalign = 1, yticksize = 10, xlabel = "y [m]")
ax1 = Axis(fig[1, 1]; options..., ylabel = "η [m]")

ηlines0 = scatter!(ax1, x, η0, color = :black)
ηlines1 = lines!(ax1, x, η1, color = :red)
ηlines2 = lines!(ax1, x, η2, color = :blue)
# ηlines3 = lines!(ax1, x, η3, color = :orange)
axislegend(ax1,
    [ηlines0, ηlines1, ηlines2], #, ηlines3],
    ["Initial Condition", "Split-Explicit", "Matrix"]) #, "Explicit"])
ylims!(ax1, (-5e-4, 5e-3))
xlims!(ax1, extrema(x))

ax2 = Axis(fig[1, 2]; options..., ylabel = "u [m/s]")
vlines01 = scatter!(ax2, x, v01, color = :black)
vlines02 = scatter!(ax2, x, v02, color = :grey)
vlines1 = lines!(ax2, x, v1, color = :red)
vlines2 = lines!(ax2, x, v2, color = :blue)
# vlines3 = lines!(ax2, x, v2, color = :orange)
axislegend(ax2,
    [vlines01, vlines02, vlines1, vlines2], #, vlines3],
    ["Initial Condition 1", "Initial Condition 2", "Split-Explicit", "Matrix"]) #, "Explicit"])
ylims!(ax2, (-1e-1, 1e-1))

ax2 = Axis(fig[1, 3]; options..., ylabel = "u [m/s]")
xf  = length(u01) == length(xf) ? xf : x
ulines01 = scatter!(ax2, xf, u01, color = :black)
ulines02 = scatter!(ax2, xf, u02, color = :grey)
ulines1 = lines!(ax2, xf, u1, color = :red)
ulines2 = lines!(ax2, xf, u2, color = :blue)
# ulines3 = lines!(ax2, xf, u3, color = :blue)
axislegend(ax2,
    [ulines01, ulines02, ulines1, ulines2], #, ulines3],
    ["Initial Condition 1", "Initial Condition 2", "Split-Explicit", "Matrix"]) #, "Explicit"])
ylims!(ax2, (-2e-4, 2e-4))

GLMakie.record(fig, "free_surface_bounded.mp4", 1:600, framerate = 12) do i
    @info "Plotting iteration $i of 600..."
    iter[] = i
end




