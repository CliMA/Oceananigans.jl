module TwoDimensionalGaussianAdvectionDiffusion

using Printf
using Statistics

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.Advection: boundary_buffer
using Oceananigans.Fields: interior

using ConvergenceTests: compute_error

# Advection and diffusion of a Gaussian.
σ(t, κ, t₀) = 4 * κ * (t + t₀)
c(x, y, z, t, U, κ, t₀) = 1 / √(4π * κ * (t + t₀)) * exp(-((x - U * t)^2 + (y - U * t)^2) / σ(t, κ, t₀))

function run_test(; Nx, Δt, stop_iteration, U = 1, κ = 1e-4, width = 0.05,
                  architecture = CPU(), topo = (Periodic, Periodic, Periodic), advection = CenteredSecondOrder())

    t₀ = width^2 / 4κ

    #####
    ##### Test advection-diffusion in the xy-direction
    #####

    domain = (x=(-1, 1.5), y=(-1, 1.5), z=(0, 1))
    grid = RectilinearGrid(architecture, topology=topo, size=(Nx, Nx, 1), halo=(6, 6, 6); domain...)

    model = NonhydrostaticModel( timestepper = :RungeKutta3,
                                        grid = grid,
                                   advection = advection,
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = ScalarDiffusivity(ν=κ, κ=κ))

    set!(model, u = U,
                v = U,
                w = (x, y, z) -> c(x, y, z, 0, U, κ, t₀),
                c = (x, y, z) -> c(x, y, z, 0, U, κ, t₀))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration)

    @info "Running Gaussian advection diffusion test for wxy, and cxy with Nx = $Nx and Δt = $Δt ($(typeof(advection).name.wrapper) buffer $(boundary_buffer(advection)))..."
    run!(simulation)

    x = xnodes(model.tracers.c)
    y = ynodes(model.tracers.c)
    c_analytical = zeros(Nx, Nx)
    for i in 1:Nx, j in 1:Nx
        c_analytical[i, j] = c.(x[i], y[j], 0, model.clock.time, U, κ, t₀)
    end

    # Calculate errors
    wxy_simulation = interior(model.velocities.w)[:, :, 1] |> Array
    wxy_errors = compute_error(wxy_simulation, c_analytical)

    cxy_simulation = interior(model.tracers.c)[:, :, 1] |> Array
    cxy_errors = compute_error(cxy_simulation, c_analytical)

    #####
    ##### Test advection-diffusion in the yz-direction
    #####

    ydomain = (x=(0, 1), y=(-1, 1.5), z=(-1, 1.5))
    ygrid = RectilinearGrid(topology=topo, size=(1, Nx, Nx), halo=(6, 6, 6); ydomain...)

    model = NonhydrostaticModel(timestepper = :RungeKutta3,
                                       grid = ygrid,
                                  advection = advection,
                                   coriolis = nothing,
                                   buoyancy = nothing,
                                    tracers = :c,
                                    closure = ScalarDiffusivity(ν=κ, κ=κ))

    set!(model, v = U,
                w = U,
                u = (x, y, z) -> c(y, z, x, 0, U, κ, t₀),
                c = (x, y, z) -> c(y, z, x, 0, U, κ, t₀))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration)

    @info "Running Gaussian advection diffusion test for uyz and cyz with Ny = $Nx and Δt = $Δt ($(typeof(advection).name.wrapper) buffer $(boundary_buffer(advection)))..."
    run!(simulation)

    # Calculate errors
    uyz_simulation = interior(model.velocities.u)[1, :, :] |> Array
    uyz_errors = compute_error(uyz_simulation, c_analytical)

    cyz_simulation = interior(model.tracers.c)[1, :, :] |> Array
    cyz_errors = compute_error(cyz_simulation, c_analytical)

    #####
    ##### Test advection-diffusion in the xz-direction
    #####

    zdomain = (x=(-1, 1.5), y=(0, 1), z=(-1, 1.5))
    zgrid = RectilinearGrid(topology=topo, size=(Nx, 1, Nx), halo=(6, 6, 6); zdomain...)

    model = NonhydrostaticModel(timestepper = :RungeKutta3,
                                       grid = zgrid,
                                  advection = advection,
                                   coriolis = nothing,
                                   buoyancy = nothing,
                                    tracers = :c,
                                    closure = ScalarDiffusivity(ν=κ, κ=κ))

    set!(model, w = U,
                u = U,
                v = (x, y, z) -> c(x, z, y, 0, U, κ, t₀),
                c = (x, y, z) -> c(x, z, y, 0, U, κ, t₀))

    simulation = Simulation(model, Δt=Δt, stop_iteration=stop_iteration)

    @info "Running Gaussian advection diffusion test for vxz and cxz with Nz = $Nx and Δt = $Δt ($(typeof(advection).name.wrapper) buffer $(boundary_buffer(advection)))..."
    run!(simulation)

    # Calculate errors
    vxz_simulation = interior(model.velocities.v)[:, 1, :] |> Array
    vxz_errors = compute_error(vxz_simulation, c_analytical)

    cxz_simulation = interior(model.tracers.c)[:, 1, :] |> Array
    cxz_errors = compute_error(cxz_simulation, c_analytical)

    return (

            cxy = (simulation = cxy_simulation,
                   analytical = c_analytical,
                           L₁ = cxy_errors.L₁,
                           L∞ = cxy_errors.L∞),

            cyz = (simulation = cyz_simulation,
                   analytical = c_analytical,
                           L₁ = cyz_errors.L₁,
                           L∞ = cyz_errors.L∞),

            cxz = (simulation = cxz_simulation,
                   analytical = c_analytical,
                           L₁ = cxz_errors.L₁,
                           L∞ = cxz_errors.L∞),

            uyz = (simulation = uyz_simulation,
                   analytical = c_analytical, # same solution as c.
                           L₁ = uyz_errors.L₁,
                           L∞ = uyz_errors.L∞),

            vxz = (simulation = vxz_simulation,
                   analytical = c_analytical, # same solution as c.
                           L₁ = vxz_errors.L₁,
                           L∞ = vxz_errors.L∞),

            wxy = (simulation = wxy_simulation,
                   analytical = c_analytical, # same solution as c.
                           L₁ = wxy_errors.L₁,
                           L∞ = wxy_errors.L∞),

            grid = grid

            )
end

end # module
