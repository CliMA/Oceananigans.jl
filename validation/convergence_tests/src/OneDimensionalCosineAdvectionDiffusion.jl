module OneDimensionalCosineAdvectionDiffusion

using Printf
using Statistics

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.Fields: interior

using ConvergenceTests: compute_error

# Advection and diffusion of a cosine.
c(x, y, z, t, U, κ) = exp(-κ * t) * cos(x - U * t)

function run_test(; Nx, Δt, stop_iteration, U = 1, κ = 1e-4,
                  architecture = CPU(), topo = (Periodic, Periodic, Periodic), advection = CenteredSecondOrder())

    #####
    ##### Test advection-diffusion in the x-direction
    #####

    domain = (x=(0, 2π), y=(0, 1), z=(0, 1))
    grid = RectilinearGrid(architecture, topology=topo, size=(Nx, 1, 1), halo=(3, 3, 3); domain...)

    model = NonhydrostaticModel(
                                 timestepper = :RungeKutta3,
                                        grid = grid,
                                   advection = advection,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = ScalarDiffusivity(ν=κ, κ=κ))

    set!(model, u = U,
                v = (x, y, z) -> c(x, y, z, 0, U, κ),
                w = (x, y, z) -> c(x, y, z, 0, U, κ),
                c = (x, y, z) -> c(x, y, z, 0, U, κ))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running cosine advection diffusion test for vx, wx, and cx with Nz = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    x = xnodes(model.tracers.c)
    c_analytical = c.(x, 0, 0, model.clock.time, U, κ)

    # Calculate errors
    vx_simulation = interior(model.velocities.v)[:, 1, 1] |> Array
    vx_errors = compute_error(vx_simulation, c_analytical)

    wx_simulation = interior(model.velocities.w)[:, 1, 1] |> Array
    wx_errors = compute_error(wx_simulation, c_analytical)

    cx_simulation = interior(model.tracers.c)[:, 1, 1] |> Array
    cx_errors = compute_error(cx_simulation, c_analytical)

    #####
    ##### Test advection-diffusion in the y-direction
    #####

    ydomain = (x=(0, 1), y=(0, 2π), z=(0, 1))
    ygrid = RectilinearGrid(topology=topo, size=(1, Nx, 1), halo=(3, 3, 3); ydomain...)

    model = NonhydrostaticModel(timestepper = :RungeKutta3,
                                       grid = ygrid,
                                  advection = advection,
                                   coriolis = nothing,
                                   buoyancy = nothing,
                                    tracers = :c,
                                    closure = ScalarDiffusivity(ν=κ, κ=κ))

    set!(model, v = U,
                u = (x, y, z) -> c(y, x, z, 0, U, κ),
                w = (x, y, z) -> c(y, x, z, 0, U, κ),
                c = (x, y, z) -> c(y, x, z, 0, U, κ))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running cosine advection diffusion test for uy, wy, and cy with Ny = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    # Calculate errors
    uy_simulation = interior(model.velocities.u)[1, :, 1] |> Array
    uy_errors = compute_error(uy_simulation, c_analytical)

    wy_simulation = interior(model.velocities.w)[1, :, 1] |> Array
    wy_errors = compute_error(wy_simulation, c_analytical)

    cy_simulation = interior(model.tracers.c)[1, :, 1] |> Array
    cy_errors = compute_error(cy_simulation, c_analytical)

    #####
    ##### Test advection-diffusion in the z-direction
    #####

    zdomain = (x=(0, 1), y=(0, 1), z=(0, 2π))
    zgrid = RectilinearGrid(topology=topo, size=(1, 1, Nx), halo=(3, 3, 3); zdomain...)

    model = NonhydrostaticModel(timestepper = :RungeKutta3,
                                       grid = zgrid,
                                  advection = advection,
                                   coriolis = nothing,
                                   buoyancy = nothing,
                                    tracers = :c,
                                    closure = ScalarDiffusivity(ν=κ, κ=κ))

    set!(model, w = U,
                u = (x, y, z) -> c(z, y, x, 0, U, κ),
                v = (x, y, z) -> c(z, y, x, 0, U, κ),
                c = (x, y, z) -> c(z, y, x, 0, U, κ))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running cosine advection diffusion test for uz, vz, and cz with Nz = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    # Calculate errors
    uz_simulation = interior(model.velocities.u)[1, 1, :] |> Array
    uz_errors = compute_error(uz_simulation, c_analytical)

    vz_simulation = interior(model.velocities.v)[1, 1, :] |> Array
    vz_errors = compute_error(vz_simulation, c_analytical)

    cz_simulation = interior(model.tracers.c)[1, 1, :] |> Array
    cz_errors = compute_error(cz_simulation, c_analytical)

    return (

            cx = (simulation = cx_simulation,
                  analytical = c_analytical,
                          L₁ = cx_errors.L₁,
                          L∞ = cx_errors.L∞),

            cy = (simulation = cy_simulation,
                  analytical = c_analytical,
                          L₁ = cy_errors.L₁,
                          L∞ = cy_errors.L∞),

            cz = (simulation = cz_simulation,
                  analytical = c_analytical,
                          L₁ = cz_errors.L₁,
                          L∞ = cz_errors.L∞),

            uy = (simulation = uy_simulation,
                  analytical = c_analytical, # same solution as c.
                          L₁ = uy_errors.L₁,
                          L∞ = uy_errors.L∞),

            uz = (simulation = uz_simulation,
                  analytical = c_analytical, # same solution as c.
                          L₁ = uz_errors.L₁,
                          L∞ = uz_errors.L∞),

            vx = (simulation = vx_simulation,
                  analytical = c_analytical, # same solution as c.
                          L₁ = vx_errors.L₁,
                          L∞ = vx_errors.L∞),

            vz = (simulation = vz_simulation,
                  analytical = c_analytical, # same solution as c.
                          L₁ = vz_errors.L₁,
                          L∞ = vz_errors.L∞),

            wx = (simulation = wx_simulation,
                  analytical = c_analytical, # same solution as c.
                          L₁ = wx_errors.L₁,
                          L∞ = wx_errors.L∞),

            wy = (simulation = wy_simulation,
                  analytical = c_analytical, # same solution as c.
                          L₁ = wy_errors.L₁,
                          L∞ = wy_errors.L∞),

            grid = grid

            )
end

end # module
