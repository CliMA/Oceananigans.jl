module GaussianAdvectionDiffusion

using Printf, Statistics

using Oceananigans, Oceananigans.OutputWriters

include("analysis.jl")

# Functions that define the forced flow problem
σ(t, κ, t₀) = 4 * κ * (t + t₀)
c(x, y, z, t, U, κ, t₀) = 1 / √(4π*κ*(t+t₀)) * exp(-(x - U * t)^2 / σ(t, κ, t₀))

function run_advection_diffusion_test(; Nx, Δt, stop_time, U = 1, κ = 1e-4, width = 0.05,
                                      architecture = CPU(), topo = (Periodic, Periodic, Bounded))
                                      
    t₀ = width^2 / 4κ

    grid = RegularCartesianGrid(size=(Nx, 1, 1), x=(-1, 1.5), y=(0, 1), z=(0, 1), topology=topo)

    model = IncompressibleModel(architecture = architecture,
                                        grid = grid,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = ConstantIsotropicDiffusivity(ν=κ, κ=κ))

    set!(model, u = U, 
                v = (x, y, z) -> c(x, y, z, 0, U, κ, t₀),
                c = (x, y, z) -> c(x, y, z, 0, U, κ, t₀))

    simulation = Simulation(model, Δt=Δt, stop_time=0.25, progress_frequency=1)

    println("Running Gaussian advection diffusion test with Nx = $Nx and Δt = $Δt...")
    run!(simulation)

    c_simulation = model.tracers.c
    c_simulation = interior(c_simulation)[:, 1, 1]

    v_simulation = model.velocities.v
    v_simulation = interior(v_simulation)[:, 1, 1]

    c_analytical = c.(grid.xC, 0, 0, model.clock.time, U, κ, t₀)

    c_errors = compute_error(c_simulation, c_analytical)
    v_errors = compute_error(v_simulation, c_analytical)

    return (

            c = (simulation = c_simulation,
                 analytical = c_analytical,
                         L₁ = c_errors.L₁,
                         L∞ = c_errors.L∞),

            v = (simulation = v_simulation,
                 analytical = c_analytical, # same solution as c.
                         L₁ = v_errors.L₁,
                         L∞ = v_errors.L∞),

            grid = grid

            )
end

end # module
