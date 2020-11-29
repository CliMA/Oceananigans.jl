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
                  architecture = CPU(), topo = (Periodic, Periodic, Periodic), advection = CenteredSecondOrder())

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
                w = (x, y, z) -> c(x, y, z, 0, U, κ),
                c = (x, y, z) -> c(x, y, z, 0, U, κ))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running cosine advection diffusion test for vx, wx, and cx with Nz = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    x = xnodes(model.tracers.c)
    c_analytical = c.(x, 0, 0, model.clock.time, U, κ)

    # Calculate errors
    vx_simulation = model.velocities.v
    vx_simulation = Array(interior(vx_simulation))[:, 1, 1]
    vx_errors = compute_error(vx_simulation, c_analytical)

    wx_simulation = model.velocities.w
    wx_simulation = Array(interior(wx_simulation))[:, 1, 1]
    wx_errors = compute_error(wx_simulation, c_analytical)

    cx_simulation = model.tracers.c
    cx_simulation = Array(interior(cx_simulation))[:, 1, 1]
    cx_errors = compute_error(cx_simulation, c_analytical)

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
                w = (x, y, z) -> c(y, x, z, 0, U, κ),
                c = (x, y, z) -> c(y, x, z, 0, U, κ))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running cosine advection diffusion test for uy, wy, and cy with Ny = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    # Calculate errors
    uy_simulation = model.velocities.u
    uy_simulation = Array(interior(uy_simulation))[1, :, 1]
    uy_errors = compute_error(uy_simulation, c_analytical)

    wy_simulation = model.velocities.w
    wy_simulation = Array(interior(wy_simulation))[1, :, 1]
    wy_errors = compute_error(wy_simulation, c_analytical)

    cy_simulation = model.tracers.c
    cy_simulation = Array(interior(cy_simulation))[1, :, 1]
    cy_errors = compute_error(cy_simulation, c_analytical)

    #####
    ##### Test cz and w-advection
    #####

    zdomain = (x=(0, 1), y=(0, 1), z=(0, 2π))
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
                u = (x, y, z) -> c(z, y, x, 0, U, κ),
                v = (x, y, z) -> c(z, y, x, 0, U, κ),
                c = (x, y, z) -> c(z, y, x, 0, U, κ))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration, iteration_interval=stop_iteration)

    @info "Running cosine advection diffusion test for uz, vz, and cz with Nz = $Nx and Δt = $Δt ($(typeof(advection)))..."
    run!(simulation)

    # Calculate errors
    uz_simulation = model.velocities.u
    uz_simulation = Array(interior(uz_simulation))[1, 1, :]
    uz_errors = compute_error(uz_simulation, c_analytical)

    vz_simulation = model.velocities.v
    vz_simulation = Array(interior(vz_simulation))[1, 1, :]
    vz_errors = compute_error(vz_simulation, c_analytical)

    cz_simulation = model.tracers.c
    cz_simulation = Array(interior(cz_simulation))[1, 1, :]
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
