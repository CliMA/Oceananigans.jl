using Revise
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface, ZStar
using Oceananigans.MultiRegion
using Statistics
using Printf
using LinearAlgebra, SparseArrays
using Oceananigans.Solvers: constructors, unpack_constructors
using Oceananigans.Grids
using GLMakie
using JLD2

function geostrophic_adjustment_simulation(free_surface, grid, timestepper=:QuasiAdamsBashforth2)

    model = HydrostaticFreeSurfaceModel(; grid,
                                          coriolis=FPlane(f = 1e-4),
                                          timestepper,
                                          free_surface,
                                          vertical_coordinate=ZStar())

    gaussian(x, L) = exp(-x^2 / 2L^2)

    U = 0.1 # geostrophic velocity
    L = grid.Lx / 40 # gaussian width
    x₀ = grid.Lx / 4 # gaussian center

    vᴳ(x, y, z) = -U * (x - x₀) / L * gaussian(x - x₀, L)

    g  = model.free_surface.gravitational_acceleration
    η₀ = model.coriolis.f * U * L / g # geostrophic free surface amplitude

    ηᴳ(x, y, z) = 2 * η₀ * gaussian(x - x₀, L)

    set!(model, v=vᴳ, η=ηᴳ)

    stop_iteration=1000

    gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
    wave_propagation_time_scale = model.grid.Δxᶜᵃᵃ / gravity_wave_speed
    simulation = Simulation(model; Δt = 20wave_propagation_time_scale, stop_iteration)

    ηarr = Vector{Field}(undef, stop_iteration+1)
    varr = Vector{Field}(undef, stop_iteration+1)
    uarr = Vector{Field}(undef, stop_iteration+1)

    save_η(sim) = ηarr[sim.model.clock.iteration+1] = deepcopy(sim.model.free_surface.η)
    save_v(sim) = varr[sim.model.clock.iteration+1] = deepcopy(sim.model.velocities.v)
    save_u(sim) = uarr[sim.model.clock.iteration+1] = deepcopy(sim.model.velocities.u)

    progress_message(sim) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
        100 * sim.model.clock.time / sim.stop_time, sim.model.clock.iteration,
        sim.model.clock.time, maximum(abs, sim.model.velocities.u))

    simulation.callbacks[:save_η]   = Callback(save_η, IterationInterval(1))
    simulation.callbacks[:save_v]   = Callback(save_v, IterationInterval(1))
    simulation.callbacks[:save_u]   = Callback(save_u, IterationInterval(1))
    simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

    run!(simulation)

    return (η=ηarr, v=varr, u=uarr)
end

Lh = 100kilometers
Lz = 400meters

grid = RectilinearGrid(size = (80, 3, 1),
                       halo = (2, 2, 2),
                       x = (0, Lh), y = (0, Lh), z = MutableVerticalDiscretization((-Lz, 0)),
                       topology = (Periodic, Periodic, Bounded))

bottom(x, y) = x > 80kilometers && x < 90kilometers ? 0.0 : -500meters

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))


explicit_free_surface = ExplicitFreeSurface()
splitexplicit_free_surface = SplitExplicitFreeSurface(grid, substeps=10)

seab2 = geostrophic_adjustment_simulation(splitexplicit_free_surface, grid)
serk3 = geostrophic_adjustment_simulation(splitexplicit_free_surface, grid, :SplitRungeKutta3)
efab2 = geostrophic_adjustment_simulation(explicit_free_surface, grid)

function plot_variable(sims, var; filename="test.mp4")
    fig = Figure()
    ax  = Axis(fig[1, 1])

    iter = Observable(1)
    for (is, sim) in enumerate(sims)
        vi = @lift(interior(sim[var][$iter], :, 1, 1))
        lines!(ax, vi, label="sim: $is")
    end

    axislegend(ax; position=:rt)

    Nt = length(sims[1][var])

    record(fig, filename, 1:Nt, framerate=15) do i
        @info "Frame $i of $Nt"
        iter[] = i
    end
end