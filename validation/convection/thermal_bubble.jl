using Printf
using Oceananigans

#####
##### Initial (= final) conditions
#####

@inline ϕ_Gaussian(x, y; L, A, σˣ, σʸ) = A * exp(-(x-L/2)^2/(2σˣ^2) -(y-L/2)^2/(2σʸ^2))
@inline ϕ_Square(x, y; L, A, σˣ, σʸ)   = A * (-σˣ <= x-L/2 <= σˣ) * (-σʸ <= y-L/2 <= σʸ)

ic_name(::typeof(ϕ_Gaussian)) = "Gaussian"
ic_name(::typeof(ϕ_Square))   = "Square"

#####
##### Experiment functions
#####

function setup_simulation(N, advection)
    L = 2000
    topology = (Periodic, Flat, Bounded)
    grid = RectilinearGrid(; topology, size=(N, N), halo=(5, 5), extent=(L, L))
    model = NonhydrostaticModel(; grid, advection, tracers=:b, buoyancy=BuoyancyTracer())

    x₀, z₀ = L/2, -L/2
    b₀(x, z) = 0.01 * exp(-100 * ((x - x₀)^2 + (z - z₀)^2) / (L^2 + L^2))
    set!(model, b=b₀)

    simulation = Simulation(model, Δt=10, stop_iteration=5000)
    add_callback!(simulation, print_progress, IterationInterval(10))
    conjure_time_step_wizard!(simulation, cfl=0.7)

    filename = @sprintf("thermal_bubble_%s_N%d.jld2", typeof(advection).name.wrapper, N)
    fields = (u = model.velocities.u, w = model.velocities.w, b = model.tracers.b)
    simulation.output_writers[:fields] = JLD2Writer(model, fields; filename, schedule=TimeInterval(100))

    return simulation
end 

function print_progress(simulation)
    model = simulation.model

    progress = 100 * (model.clock.iteration / simulation.stop_iteration)

    u_max = maximum(abs, model.velocities.u)
    w_max = maximum(abs, model.velocities.w)
    b_min, b_max = extrema(model.tracers.b)

    i, t = model.clock.iteration, model.clock.time

    @info @sprintf("[%06.2f%%] i: %d, t: %s, U_max: (%.2e, %.2e), b: (min=%.5f, max=%.5f)",
                   progress, i, prettytime(t), u_max, w_max, b_min, b_max)
end

schemes = (WENO(), Centered(order=4))
Ns = (32, 128)

for scheme in schemes, N in Ns
    @info @sprintf("Running thermal bubble advection [%s, N=%d]...", typeof(scheme), N)
    simulation = setup_simulation(N, scheme)
    run!(simulation)
end

