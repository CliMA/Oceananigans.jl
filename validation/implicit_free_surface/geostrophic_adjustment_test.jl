using Revise
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using Statistics
using Printf
using LinearAlgebra, SparseArrays
using Oceananigans.Solvers: constructors, unpack_constructors

function geostrophic_adjustment_simulation(free_surface, topology; arch = Oceananigans.CPU())

    Lh = 100kilometers
    Lz = 400meters

    grid = RectilinearGrid(arch,
        size = (64, 3, 1),
        x = (0, Lh), y = (0, Lh), z = (-Lz, 0),
        topology = topology)

    coriolis = FPlane(f = 1e-4)

    model = HydrostaticFreeSurfaceModel(grid = grid,
        coriolis = coriolis,
        free_surface = free_surface)

    gaussian(x, L) = exp(-x^2 / 2L^2)

    U = 0.1 # geostrophic velocity
    L = grid.Lx / 40 # gaussian width
    x₀ = grid.Lx / 4 # gaussian center

    vᵍ(x, y, z) = -U * (x - x₀) / L * gaussian(x - x₀, L)

    g = model.free_surface.gravitational_acceleration
    η = model.free_surface.η


    η₀ = coriolis.f * U * L / g # geostrohpic free surface amplitude

    ηᵍ(x) = η₀ * gaussian(x - x₀, L)

    ηⁱ(x, y) = 2 * ηᵍ(x)

    set!(model, v = vᵍ)
    set!(η, ηⁱ)

    gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
    wave_propagation_time_scale = model.grid.Δxᶜᵃᵃ / gravity_wave_speed
    simulation = Simulation(model, Δt = 2wave_propagation_time_scale, stop_iteration = 300)

    return simulation
end

function run_and_analyze(simulation)
    η = simulation.model.free_surface.η
    u, v, w = simulation.model.velocities
    Δt = simulation.Δt

    ηx = Field(∂x(η))
    compute!(ηx)

    f = simulation.model.free_surface
    @views u₀ = interior(u)[:, 1, 1]
    @views v₀ = interior(v)[:, 1, 1]
    @views η₀ = interior(η)[:, 1, 1]
    @views ηx₀ = interior(ηx)[:, 1, 1]

    if f isa SplitExplicitFreeSurface
        solver_method = "SplitExplicitFreeSurface"
    else
        solver_method = string(simulation.model.free_surface.solver_method)
    end

    simulation.output_writers[:fields] = JLD2OutputWriter(simulation.model, (η, ηx, u, v, w),
        schedule = TimeInterval(Δt),
        prefix = "solution_$(solver_method)",
        force = true)

    progress_message(sim) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
        100 * sim.model.clock.time / sim.stop_time, sim.model.clock.iteration,
        sim.model.clock.time, maximum(abs, sim.model.velocities.u))


    simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

    run!(simulation)

    compute!(ηx)

    @views u₁ = interior(u)[:, 1, 1]
    @views v₁ = interior(v)[:, 1, 1]
    @views η₁ = interior(η)[:, 1, 1]
    @views ηx₁ = interior(ηx)[:, 1, 1]

    @show mean(Array(η₀))
    @show mean(Array(η₁))

    Δη = Array(η₁) - Array(η₀)

    return (; η₀, η₁, Δη, ηx₀, ηx₁, u₀, u₁, v₀, v₁)
end

# fft_based_free_surface = ImplicitFreeSurface()
pcg_free_surface = ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient);
matrix_free_surface = ImplicitFreeSurface(solver_method = :HeptadiagonalIterativeSolver);
splitexplicit_free_surface = SplitExplicitFreeSurface()

topology_types = [(Bounded, Periodic, Bounded), (Periodic, Periodic, Bounded)]
topology_types = [topology_types[1]]

archs = [Oceananigans.CPU(), Oceananigans.GPU()]
archs = [archs[1]]

free_surfaces = [pcg_free_surface, matrix_free_surface, splitexplicit_free_surface];
simulations = [geostrophic_adjustment_simulation(free_surface, topology_type, arch = arch) for free_surface in free_surfaces, topology_type in topology_types, arch in archs];
data = [run_and_analyze(sim) for sim in simulations];
# run_and_analyze(simulations[3])


using GLMakie
using JLD2

file1 = jldopen("solution_PreconditionedConjugateGradient.jld2")
file2 = jldopen("solution_MatrixIterativeSolver.jld2")
file3 = jldopen("solution_SplitExplicitFreeSurface.jld2")

grid = file1["serialized/grid"]

x = grid.xᶜᵃᵃ[1:grid.Nx]
xf = grid.xᶠᵃᵃ[1:grid.Nx+1]
y = grid.yᵃᶜᵃ[1:grid.Ny]


iterations = parse.(Int, keys(file1["timeseries/t"]))
iterations = iterations[1:200]

iter = Node(0) # Node or Observable depending on Makie version
mid = Int(floor(grid.Ny / 2))
η0 = file1["timeseries/1/0"][:, mid, 1]
η1 = @lift(Array(file1["timeseries/1/"*string($iter)])[:, mid, 1])
η2 = @lift(Array(file2["timeseries/1/"*string($iter)])[:, mid, 1])
η3 = @lift(Array(file3["timeseries/1/"*string($iter)])[:, mid, 1])
u1 = @lift(Array(file1["timeseries/3/"*string($iter)])[:, mid, 1])
u2 = @lift(Array(file2["timeseries/3/"*string($iter)])[:, mid, 1])
u3 = @lift(Array(file3["timeseries/3/"*string($iter)])[:, mid, 1])
fig = Figure(resolution = (1400, 1000))
options = (; ylabelsize = 22,
    xlabelsize = 22, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
    xticksize = 10, ytickalign = 1, yticksize = 10, xlabel = "y [m]", xlims = extrema(x))
ax1 = Axis(fig[1, 1]; options..., ylabel = "η [m]", ylims = (-5e-4, 5e-4))
ax2 = Axis(fig[1, 2]; options..., ylabel = "u [m/s]", ylims = (-5e-5, 5e-5))

ηlines0 = lines!(ax1, x, η0, color = :black)
ηlines1 = lines!(ax1, x, η1, color = :red)
ηlines2 = lines!(ax1, x, η2, color = :blue)
ηlines3 = lines!(ax1, x, η3, color = :orange)
axislegend(ax1,
    [ηlines0, ηlines1, ηlines2, ηlines3],
    ["Initial Condition", "PCG", "Matrix", "Split-Explicit"])
ylims!(ax1, (-5e-4, 5e-3))
u0 = Array(file3["timeseries/3/"*string(0)])[:, mid, 1]
xf = length(u0) == length(xf) ? xf : x
ulines1 = lines!(ax2, xf, u1, color = :red)
ulines2 = lines!(ax2, xf, u2, color = :blue)
ulines3 = lines!(ax2, xf, u3, color = :orange)
axislegend(ax2,
    [ulines1, ulines2, ulines3],
    ["PCG", "Matrix", "Split-Explicit"])
ylims!(ax2, (-2e-4, 2e-4))
GLMakie.record(fig, "free_surface_bounded.mp4", iterations, framerate = 12) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end




