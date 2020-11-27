module OneDimensionalGaussianAdvectionDiffusion

using Printf
using Statistics

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection

include("analysis.jl")

# Advection and diffusion of a Gaussian.
σ(t, κ, t₀) = 4 * κ * (t + t₀)
c(x, y, z, t, U, κ, t₀) = 1 / √(4π * κ * (t + t₀)) * exp(-(x - U * t)^2 / σ(t, κ, t₀))

function run_test(; Nx, Δt, stop_iteration, U = 1, κ = 1e-4, width = 0.05,
                  architecture = CPU(), topo = (Periodic, Periodic, Periodic), advection = CenteredSecondOrder())

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
                c = (x, y, z) -> c(x, y, z, 0, U, κ, t₀),
                v = (x, y, z) -> c(x, y, z, 0, U, κ, t₀),
                w = (x, y, z) -> c(x, y, z, 0, U, κ, t₀))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running Gaussian advection diffusion test for v and cx with Nx = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    x = xnodes(model.tracers.c)
    c_analytical = c.(x, 0, 0, model.clock.time, U, κ, t₀)

    # Calculate errors
    cx_simulation = model.tracers.c
    cx_simulation = interior(cx_simulation)[:, 1, 1]
    cx_errors = compute_error(cx_simulation, c_analytical)

    vx_simulation = model.velocities.v
    vx_simulation = interior(vx_simulation)[:, 1, 1]
    vx_errors = compute_error(vx_simulation, c_analytical)

    wx_simulation = model.velocities.w
    wx_simulation = interior(wx_simulation)[:, 1, 1]
    wx_errors = compute_error(wx_simulation, c_analytical)

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
                u = (x, y, z) -> c(y, x, z, 0, U, κ, t₀),
                w = (x, y, z) -> c(y, x, z, 0, U, κ, t₀))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running Gaussian advection diffusion test for u and cy with Nx = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    # Calculate errors
    cy_simulation = model.tracers.c
    cy_simulation = interior(cy_simulation)[1, :, 1]
    cy_errors = compute_error(cy_simulation, c_analytical)

    uy_simulation = model.velocities.u
    uy_simulation = interior(uy_simulation)[1, :, 1]
    uy_errors = compute_error(uy_simulation, c_analytical)

    wy_simulation = model.velocities.w
    wy_simulation = interior(wy_simulation)[1, :, 1]
    wy_errors = compute_error(wy_simulation, c_analytical)

    #####
    ##### Test cz and w-advection
    #####

    zdomain = (x=(0, 1), y=(0, 1), z=(-1, 1.5))
    zgrid = RegularCartesianGrid(topology=topo, size=(1, 1, Nx), halo=(3, 3, 3); zdomain...)

    model = IncompressibleModel(architecture = architecture,
                                 timestepper = :RungeKutta3,
                                        grid = zgrid,
                                   advection = advection,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = IsotropicDiffusivity(ν=κ, κ=κ))

    set!(model, w = U,
                c = (x, y, z) -> c(z, x, y, 0, U, κ, t₀),
                u = (x, y, z) -> c(z, x, y, 0, U, κ, t₀),
                v = (x, y, z) -> c(z, x, y, 0, U, κ, t₀))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running Gaussian advection diffusion test for u and cy with Nx = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    # Calculate errors
    cz_simulation = model.tracers.c
    cz_simulation = interior(cz_simulation)[1, 1, :]
    cz_errors = compute_error(cz_simulation, c_analytical)

    uz_simulation = model.velocities.u
    uz_simulation = interior(uz_simulation)[1, 1, :]
    uz_errors = compute_error(uz_simulation, c_analytical)

    vz_simulation = model.velocities.v
    vz_simulation = interior(vz_simulation)[1, 1, :]
    vz_errors = compute_error(vz_simulation, c_analytical)

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
