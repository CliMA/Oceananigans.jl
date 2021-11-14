using Printf
using Logging
using Plots

using Oceananigans
using Oceananigans.Advection
using Oceananigans.OutputWriters
using Oceananigans.Utils

ENV["GKSwstype"] = "100"
pyplot()

Logging.global_logger(OceananigansLogger())

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

function setup_simulation(N, advection_scheme)
    L = 2000
    topology = (Periodic, Flat, Bounded)
    grid = RectilinearGrid(topology=topology, size=(N, 1, N), halo=(5, 5, 5), extent=(L, L, L))

    model = NonhydrostaticModel(
               grid = grid,
        timestepper = :RungeKutta3,
          advection = advection_scheme,
            closure = IsotropicDiffusivity(ν=0, κ=0)
    )

    x₀, z₀ = L/2, -L/2
    T₀(x, y, z) = 20 + 0.01 * exp(-100 * ((x - x₀)^2 + (z - z₀)^2) / (L^2 + L^2))
    set!(model, T=T₀)

    simulation = Simulation(model, Δt=10, stop_iteration=5000, progress=print_progress, iteration_interval=10)

    filename = @sprintf("thermal_bubble_%s_N%d.nc", typeof(advection_scheme), N)
    fields = Dict("u" => model.velocities.u, "w" => model.velocities.w, "T" => model.tracers.T)
    global_attributes = Dict("N" => N, "advection_scheme" => string(typeof(advection_scheme)))

    simulation.output_writers[:fields] =
        NetCDFOutputWriter(model, fields, filename=filename, schedule=TimeInterval(100), global_attributes=global_attributes)

    return simulation
end 

function print_progress(simulation)
    model = simulation.model

    progress = 100 * (model.clock.iteration / simulation.stop_iteration)

    u_max = maximum(abs, interior(model.velocities.u))
    w_max = maximum(abs, interior(model.velocities.w))
    T_min, T_max = extrema(interior(model.tracers.T))
    CFL = max(u_max, w_max) * simulation.Δt / min(model.grid.Δxᶜᵃᵃ, model.grid.Δzᵃᵃᶜ)

    i, t = model.clock.iteration, model.clock.time
    @info @sprintf("[%06.2f%%] i: %d, t: %.4f, U_max: (%.2e, %.2e), T: (min=%.5f, max=%.5f), CFL: %.4f",
                   progress, i, t, u_max, w_max, T_min, T_max, CFL)
end

schemes = (WENO5(), CenteredFourthOrder())
Ns = (32, 128)

for scheme in schemes, N in Ns
    @info @sprintf("Running thermal bubble advection [%s, N=%d]...", typeof(scheme), N)
    simulation = setup_simulation(N, scheme)
    run!(simulation)
end
