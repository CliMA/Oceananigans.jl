module OneDimensionalCosineAdvectionDiffusion

using Printf
using Statistics

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection

include("analysis.jl")

# Advection and diffusion of a cosine.
c(x, y, z, t, U, κ) = exp(-κ * t) * cos(x - U * t)

function run_test(; Nx, Δt, stop_iteration, U = 1, κ = 1e-4,
                  architecture = CPU(), topo = (Periodic, Periodic, Bounded), advection = CenteredSecondOrder())

    #####
    ##### Test cx and v-advection
    #####

    domain = (x=(0, 2π), y=(0, 1), z=(0, 1))
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
                v = (x, y, z) -> c(x, y, z, 0, U, κ),
                c = (x, y, z) -> c(x, y, z, 0, U, κ))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running 1D in x cosine advection diffusion test for v and cx with Nx = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    x = xnodes(model.tracers.c)
    c_analytical = c.(x, 0, 0, model.clock.time, U, κ)

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

    ydomain = (x=(0, 1), y=(0, 2π), z=(0, 1))
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
                u = (x, y, z) -> c(y, x, z, 0, U, κ),
                c = (x, y, z) -> c(y, x, z, 0, U, κ))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, schedule=IterationInterval(stop_iteration))

    @info "Running 1D in y cosine advection diffusion test for u and cy with Ny = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    # Calculate errors
    u_simulation = model.velocities.u
    u_simulation = interior(u_simulation)[1, :, 1]
    u_errors = compute_error(u_simulation, c_analytical)

    # Calculate errors
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

            v = (simulation = v_simulation,
                 analytical = c_analytical, # same solution as c.
                         L₁ = v_errors.L₁,
                         L∞ = v_errors.L∞),

            u = (simulation = u_simulation,
                 analytical = c_analytical, # same solution as c.
                         L₁ = u_errors.L₁,
                         L∞ = u_errors.L∞),

            grid = grid

            )
end

end # module
