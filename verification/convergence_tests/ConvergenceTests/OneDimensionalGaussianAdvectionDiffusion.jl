module OneDimensionalGaussianAdvectionDiffusion

using Printf
using Statistics

using Oceananigans
using Oceananigans.Grids

include("analysis.jl")

# Advection and diffusion of a Gaussian.
σ(t, κ, t₀) = 4 * κ * (t + t₀)
c(x, y, z, t, U, κ, t₀) = 1 / √(4π * κ * (t + t₀)) * exp(-(x - U * t)^2 / σ(t, κ, t₀))

function run_test(; Nx, Δt, stop_iteration, U = 1, κ = 1e-4, width = 0.05,
                  architecture = CPU(), topo = (Periodic, Periodic, Bounded), advection = CenteredSecondOrder())

    t₀ = width^2 / 4κ

    #####
    ##### Test cx and v-advection
    #####

    domain = (x=(-1, 1.5), y=(0, 1), z=(0, 1))
    grid = RegularCartesianGrid(topology=topo, size=(Nx, 1, 1), halo=(3, 3, 3); domain...)

    model = IncompressibleModel(architecture = architecture,
                                 timestepper = :RungeKutta3,
                                        grid = grid,
                                   advection = advection,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = IsotropicDiffusivity(ν=κ, κ=κ))

    set!(model, u = U,
                v = (x, y, z) -> c(x, y, z, 0, U, κ, t₀),
                c = (x, y, z) -> c(x, y, z, 0, U, κ, t₀))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running Gaussian advection diffusion test for v and cx with Nx = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    x = xnodes(model.tracers.c)
    c_analytical = c.(x, 0, 0, model.clock.time, U, κ, t₀)

    # Calculate errors
    cx_simulation = model.tracers.c
    cx_simulation = interior(cx_simulation)[:, 1, 1]
    cx_errors = compute_error(cx_simulation, c_analytical)

    v_simulation = model.velocities.v
    v_simulation = interior(v_simulation)[:, 1, 1]
    v_errors = compute_error(v_simulation, c_analytical)

    #####
    ##### Test cy and u-advection
    #####

    ydomain = (x=(0, 1), y=(-1, 1.5), z=(0, 1))
    ygrid = RegularCartesianGrid(topology=topo, size=(1, Nx, 1), halo=(3, 3, 3); ydomain...)

    model = IncompressibleModel(architecture = architecture,
                                 timestepper = :RungeKutta3,
                                        grid = ygrid,
                                   advection = advection,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = IsotropicDiffusivity(ν=κ, κ=κ))

    set!(model, v = U,
                c = (x, y, z) -> c(y, x, z, 0, U, κ, t₀),
                u = (x, y, z) -> c(y, x, z, 0, U, κ, t₀))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running Gaussian advection diffusion test for u and cy with Nx = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    # Calculate errors
    u_simulation = model.velocities.u
    u_simulation = interior(u_simulation)[1, :, 1]
    u_errors = compute_error(u_simulation, c_analytical)

    cy_simulation = model.tracers.c
    cy_simulation = interior(cy_simulation)[1, :, 1]
    cy_errors = compute_error(cy_simulation, c_analytical)

    return (

            cx = (simulation = cx_simulation,
                  analytical = c_analytical,
                          L₁ = cx_errors.L₁,
                          L∞ = cx_errors.L∞),

            cy = (simulation = cy_simulation,
                  analytical = c_analytical,
                          L₁ = cy_errors.L₁,
                          L∞ = cy_errors.L∞),

            u = (simulation = u_simulation,
                 analytical = c_analytical, # same solution as c.
                         L₁ = u_errors.L₁,
                         L∞ = u_errors.L∞),

            v = (simulation = v_simulation,
                 analytical = c_analytical, # same solution as c.
                         L₁ = v_errors.L₁,
                         L∞ = v_errors.L∞),

            grid = grid

            )
end

end # module
