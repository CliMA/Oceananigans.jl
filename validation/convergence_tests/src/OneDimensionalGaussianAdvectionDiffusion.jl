module OneDimensionalGaussianAdvectionDiffusion

using Printf
using Statistics

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.Fields: interior

using ConvergenceTests: compute_error

# Advection and diffusion of a Gaussian.
σ(t, κ, t₀) = 4 * κ * (t + t₀)
c(x, y, z, t, U, κ, t₀) = 1 / √(4π * κ * (t + t₀)) * exp(-(x - U * t)^2 / σ(t, κ, t₀))

function run_test(; Nx, Δt, stop_iteration, U = 1, κ = 1e-4, width = 0.05,
                  architecture = CPU(), topo = (Periodic, Periodic, Periodic), advection = CenteredSecondOrder())

    t₀ = width^2 / 4κ

    #####
    ##### Test advection-diffusion in the x-direction
    #####

    domain = (x=(-1, 1.5), y=(0, 1), z=(0, 1))
    grid = RegularRectilinearGrid(topology=topo, size=(Nx, 1, 1), halo=(3, 3, 3); domain...)

    model = NonhydrostaticModel(architecture = architecture,
                                 timestepper = :RungeKutta3,
                                        grid = grid,
                                   advection = advection,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = IsotropicDiffusivity(ν=κ, κ=κ))

    set!(model, u = U,
                v = (x, y, z) -> c(x, y, z, 0, U, κ, t₀),
                w = (x, y, z) -> c(x, y, z, 0, U, κ, t₀),
                c = (x, y, z) -> c(x, y, z, 0, U, κ, t₀))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running Gaussian advection diffusion test for vx, wx, and cx with Nx = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    x = xnodes(model.tracers.c)
    c_analytical = c.(x, 0, 0, model.clock.time, U, κ, t₀)

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

    ydomain = (x=(0, 1), y=(-1, 1.5), z=(0, 1))
    ygrid = RegularRectilinearGrid(topology=topo, size=(1, Nx, 1), halo=(3, 3, 3); ydomain...)

    model = NonhydrostaticModel(architecture = architecture,
                                 timestepper = :RungeKutta3,
                                        grid = ygrid,
                                   advection = advection,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = IsotropicDiffusivity(ν=κ, κ=κ))

    set!(model, v = U,
                u = (x, y, z) -> c(y, x, z, 0, U, κ, t₀),
                w = (x, y, z) -> c(y, x, z, 0, U, κ, t₀),
                c = (x, y, z) -> c(y, x, z, 0, U, κ, t₀))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running Gaussian advection diffusion test for uy, wy, and cy with Ny = $Nx and Δt = $Δt ($(typeof(advection)))..."
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

    zdomain = (x=(0, 1), y=(0, 1), z=(-1, 1.5))
    zgrid = RegularRectilinearGrid(topology=topo, size=(1, 1, Nx), halo=(3, 3, 3); zdomain...)

    model = NonhydrostaticModel(architecture = architecture,
                                 timestepper = :RungeKutta3,
                                        grid = zgrid,
                                   advection = advection,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = IsotropicDiffusivity(ν=κ, κ=κ))

    set!(model, w = U,
                u = (x, y, z) -> c(z, x, y, 0, U, κ, t₀),
                v = (x, y, z) -> c(z, x, y, 0, U, κ, t₀),
                c = (x, y, z) -> c(z, x, y, 0, U, κ, t₀))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running Gaussian advection diffusion test for uz, vz, and cz with Nz = $Nx and Δt = $Δt ($(typeof(advection)))..."
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

            cz = (simulation = cy_simulation,
                  analytical = c_analytical,
                          L₁ = cy_errors.L₁,
                          L∞ = cy_errors.L∞),

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
