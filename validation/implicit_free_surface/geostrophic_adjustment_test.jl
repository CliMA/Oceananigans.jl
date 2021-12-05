using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using Statistics
using IterativeSolvers
using LinearAlgebra, SparseArrays
using Oceananigans.Solvers: constructors, unpack_constructors

function geostrophic_adjustment_simulation(free_surface, topology)

    Lh = 100kilometers
    Lz = 400meters

    grid = RectilinearGrid(size = (64, 3, 1),
                                  x = (0, Lh), y = (0, Lh), z = (-Lz, 0),
                                  topology = topology)

    coriolis = FPlane(f=1e-4)

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        coriolis = coriolis,
                                        free_surface = free_surface)

    gaussian(x, L) = exp(-x^2 / 2L^2)
    
    U = 0.1 # geostrophic velocity
    L = grid.Lx / 40 # gaussian width
    x₀ = grid.Lx / 4 # gaussian center
    
    vᵍ(x, y, z) = - U * (x - x₀) / L * gaussian(x - x₀, L)
    
    g = model.free_surface.gravitational_acceleration
    
    η₀ = coriolis.f * U * L / g # geostrohpic free surface amplitude
    
    ηᵍ(x) = η₀ * gaussian(x - x₀, L)

    ηⁱ(x, y) = 2 * ηᵍ(x)

    set!(model, v=vᵍ, η=ηⁱ)
    gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
    wave_propagation_time_scale = model.grid.Δxᶜᵃᵃ / gravity_wave_speed
    simulation = Simulation(model, Δt=2wave_propagation_time_scale, stop_iteration=10000)

    return simulation
end

function run_and_analyze(simulation, sim)
    η = simulation.model.free_surface.η
    u, v, w = simulation.model.velocities
    Δt = simulation.Δt

    ηx = ComputedField(∂x(η))
    compute!(ηx)

    u₀  = interior(u)[:, 1, 1]
    v₀  = interior(v)[:, 1, 1]
    η₀  = interior(η)[:, 1, 1]
    ηx₀ = interior(ηx)[:, 1, 1]

    simulation.output_writers[:fields] = JLD2OutputWriter(simulation.model, (η, ηx, u, v, w),
                                                          schedule = TimeInterval(Δt),
                                                          prefix = "solution.$(sim)")


    run!(simulation)
    
    compute!(ηx)

    u₁ = interior(u)[:, 1, 1]
    v₁ = interior(v)[:, 1, 1]
    η₁ = interior(η)[:, 1, 1]
    ηx₁ = interior(ηx)[:, 1, 1]

    @show mean(η₀)
    @show mean(η₁)

    Δη = η₁ .- η₀

    return (; η₀, η₁, Δη, ηx₀, ηx₁, u₀, u₁, v₀, v₁)
end

# fft_based_free_surface = ImplicitFreeSurface()
pcg_free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient);
matrix_free_surface = ImplicitFreeSurface(solver_method=:MatrixIterativeSolver);

topology_types = [(Bounded, Periodic, Bounded), (Periodic, Periodic, Bounded)]

free_surfaces = [pcg_free_surface, matrix_free_surface];
simulations = [geostrophic_adjustment_simulation(free_surface, topology_type) for free_surface in free_surfaces, topology_type in topology_types];
data = [run_and_analyze(sim, i) for (sim, i) in zip(simulations, 1:length(simulations))];

# mod1 = simulations[1].model
# mod2 = simulations[2].model

# func = mod1.free_surface.implicit_step_solver.preconditioned_conjugate_gradient_solver.linear_operation!
# matr = mod2.free_surface.implicit_step_solver.matrix_iterative_solver.matrix

# η = similar(mod1.free_surface.η)
# β = similar(mod1.free_surface.η)

# parent(η) .= 1

# Ax = mod1.free_surface.implicit_step_solver.vertically_integrated_lateral_areas.xᶠᶜᶜ
# Ay = mod1.free_surface.implicit_step_solver.vertically_integrated_lateral_areas.yᶜᶠᶜ

# Nx, Ny = (mod1.grid.Nx, mod1.grid.Ny)

# Δt = simulations[1].Δt
# g = mod1.free_surface.gravitational_acceleration
# grid = mod1.grid

# parent(η) .= 0
# η[1,1] = 1
# func(β, η, Ax, Ay, g, Δt)
# x1 = interior(β)

# x2 = matr * interior(η)[:]
# x2 = reshape(x2, Nx, Ny)


# cy = grid.Lz * grid.Δxᶜᵃᵃ / grid.Δyᵃᶜᵃ   
# cx = grid.Lz * grid.Δyᵃᶜᵃ / grid.Δxᶜᵃᵃ   

# cL = grid.Δyᵃᶜᵃ * grid.Lz / 4 / grid.Δxᶜᵃᵃ

using GLMakie
using JLD2 

file1 = jldopen("solution.1.jld2")
file2 = jldopen("solution.2.jld2")
file3 = jldopen("solution.3.jld2")
file4 = jldopen("solution.4.jld2")

grid = file1["serialized/grid"]

x  = grid.xᶜᵃᵃ[1:grid.Nx]
xf = grid.xᶠᵃᵃ[1:grid.Nx+1]
y  = grid.yᵃᶜᵃ[1:grid.Ny]


iterations = parse.(Int, keys(file1["timeseries/t"]))
iterations = iterations[1:200]

iter = Node(0)

mid = Int(floor(grid.Ny/2))
η0 = file1["timeseries/1/0"][:, mid, 1]
η1 = @lift(Array(file1["timeseries/1/" * string($iter)])[:, mid, 1])
η2 = @lift(Array(file2["timeseries/1/" * string($iter)])[:, mid, 1])
u1 = @lift(Array(file1["timeseries/3/" * string($iter)])[:, mid, 1])
u2 = @lift(Array(file2["timeseries/3/" * string($iter)])[:, mid, 1])

fig = Figure(resolution=(1000, 500))
plot(fig[1,1] , x, η0, color = :green)
plot!(fig[1,1], x, η1, color = :red)
plot!(fig[1,1], x, η2, color = :blue)
plot(fig[1,2], xf, u1, color = :red)
plot!(fig[1,2],xf, u2, color = :blue)
ylims!(-5e-5, 5e-5)
GLMakie.record(fig, "free_surface_bounded.mp4", iterations, framerate=12) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end

mid = Int(floor(grid.Ny/2))
η3 = @lift(Array(file3["timeseries/1/" * string($iter)])[:, mid, 1])
η4 = @lift(Array(file4["timeseries/1/" * string($iter)])[:, mid, 1])
u3 = @lift(Array(file3["timeseries/3/" * string($iter)])[:, mid, 1])
u4 = @lift(Array(file4["timeseries/3/" * string($iter)])[:, mid, 1])


fig = Figure(resolution=(1000, 500))
plot(fig[1,1] , x, η0, color = :green)
plot!(fig[1,1], x, η3, color = :red)
plot!(fig[1,1], x, η4, color = :blue)
plot(fig[1,2],  x, u3, color = :red )
plot!(fig[1,2], x, u4, color = :blue)
ylims!(-5e-5, 5e-5)
GLMakie.record(fig, "free_surface_periodic.mp4", iterations, framerate=12) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end
