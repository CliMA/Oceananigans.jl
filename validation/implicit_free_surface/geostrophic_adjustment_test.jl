using Revise
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface, ZStarCoordinate
using Oceananigans.MultiRegion
using Statistics
using Printf
using LinearAlgebra, SparseArrays
using Oceananigans.Solvers: constructors, unpack_constructors
using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind
using Oceananigans.Grids
using GLMakie
using JLD2

function geostrophic_adjustment_simulation(free_surface, grid, timestepper=:QuasiAdamsBashforth2)

    model = HydrostaticFreeSurfaceModel(; grid,
                                          momentum_advection = nothing,
                                          coriolis=FPlane(f = 1e-4),
                                          tracers = :c,
                                          buoyancy = nothing,
                                          timestepper,
                                          free_surface)

    gaussian(x, L) = exp(-x^2 / 2L^2)

    U = 0.1 # geostrophic velocity
    L = grid.Lx / 40 # gaussian width
    x₀ = grid.Lx / 4 # gaussian center

    vᴳ(x, y, z) = -U * (x - x₀) / L * gaussian(x - x₀, L)
    Vᴳ(x, y) = grid.Lz * vᴳ(x, y, 1) 

    g  = model.free_surface.gravitational_acceleration
    η₀ = model.coriolis.f * U * L / g # geostrophic free surface amplitude

    ηᴳ(x, y, z) = 2 * η₀ * gaussian(x - x₀, L)

    set!(model, v=vᴳ, η=ηᴳ, c=1)

    if free_surface isa SplitExplicitFreeSurface
        set!(model.free_surface.barotropic_velocities.V, Vᴳ)
    end

    if grid isa MutableGridOfSomeKind
        # Initialize the vertical grid scaling
        z = model.grid.z
        Oceananigans.BoundaryConditions.fill_halo_regions!(model.free_surface.η)
        parent(z.ηⁿ)   .=  parent(model.free_surface.η)
        for i in 0:grid.Nx+1, j in 0:grid.Ny+1
            Oceananigans.Models.HydrostaticFreeSurfaceModels.update_grid_scaling!(z.σᶜᶜⁿ, z.σᶠᶜⁿ, z.σᶜᶠⁿ, z.σᶠᶠⁿ, z.σᶜᶜ⁻, i, j, grid, z.ηⁿ)
        end
    end
        
    stop_iteration=1000

    gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
    wave_propagation_time_scale = model.grid.Δxᶜᵃᵃ / gravity_wave_speed
    simulation = Simulation(model; Δt = 0.2 * wave_propagation_time_scale, stop_iteration)

    ηarr = Vector{Field}(undef, stop_iteration+1)
    varr = Vector{Field}(undef, stop_iteration+1)
    uarr = Vector{Field}(undef, stop_iteration+1)
    carr = Vector{Field}(undef, stop_iteration+1)
    warr = [zeros(grid.Nx) for _ in 1:stop_iteration+1]
    garr = [zeros(grid.Nx) for _ in 1:stop_iteration+1]

    save_η(sim) = ηarr[sim.model.clock.iteration+1] = deepcopy(sim.model.free_surface.η)
    save_v(sim) = varr[sim.model.clock.iteration+1] = deepcopy(sim.model.velocities.v)
    save_u(sim) = uarr[sim.model.clock.iteration+1] = deepcopy(sim.model.velocities.u)
    save_c(sim) = carr[sim.model.clock.iteration+1] = deepcopy(sim.model.tracers.c)
    save_w(sim) = warr[sim.model.clock.iteration+1] .= sim.model.velocities.w[1:sim.model.grid.Nx, 2, 2]
    
    if grid isa MutableGridOfSomeKind
        save_g(sim) = garr[sim.model.clock.iteration+1] .= sim.model.grid.z.ηⁿ[1:sim.model.grid.Nx, 2, 1]
        simulation.callbacks[:save_g] = Callback(save_g, IterationInterval(1))
    end

    function progress_message(sim) 
        H = sum(sim.model.free_surface.η)
        msg = @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e, sim(η): %e",
                        100 * sim.model.clock.time / sim.stop_time, sim.model.clock.iteration,
                        sim.model.clock.time, maximum(abs, sim.model.velocities.u), H)

        if grid isa MutableGridOfSomeKind
                msg2 = @sprintf(", max(Δη): %.2e", maximum(sim.model.grid.z.ηⁿ[1:sim.model.grid.Nx, 2, 1] .- interior(sim.model.free_surface.η)))
                msg  = msg * msg2
        end
        
        @info msg
    end

    simulation.callbacks[:save_η]   = Callback(save_η, IterationInterval(1))
    simulation.callbacks[:save_v]   = Callback(save_v, IterationInterval(1))
    simulation.callbacks[:save_u]   = Callback(save_u, IterationInterval(1))
    simulation.callbacks[:save_c]   = Callback(save_c, IterationInterval(1))
    simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

    run!(simulation)

    return (η=ηarr, v=varr, u=uarr, c=carr, w=warr, g=garr), model
end

Lh = 100kilometers
Lz = 400meters

grid = RectilinearGrid(size = (80, 3, 1),
                       halo = (2, 2, 2),
                       x = (0, Lh), y = (0, Lh), 
                       z = (-Lz, 0), #MutableVerticalDiscretization((-Lz, 0)),
                       topology = (Periodic, Periodic, Bounded))

explicit_free_surface = ExplicitFreeSurface()
implicit_free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver)
splitexplicit_free_surface = SplitExplicitFreeSurface(deepcopy(grid), substeps=120)

seab2, sim2 = geostrophic_adjustment_simulation(splitexplicit_free_surface, deepcopy(grid))
serk3, sim3 = geostrophic_adjustment_simulation(splitexplicit_free_surface, deepcopy(grid), :SplitRungeKutta3)
efab2, sim4 = geostrophic_adjustment_simulation(explicit_free_surface, deepcopy(grid))
efrk3, sim5 = geostrophic_adjustment_simulation(explicit_free_surface, deepcopy(grid), :SplitRungeKutta3)
imab2, sim6 = geostrophic_adjustment_simulation(implicit_free_surface, deepcopy(grid))
imrk3, sim7 = geostrophic_adjustment_simulation(implicit_free_surface, deepcopy(grid), :SplitRungeKutta3)

import Oceananigans.Fields: interior
interior(a::Array, idx...) = a

function plot_variable(sims, var; 
                       filename="test.mp4",
                       labels=nothing,
                       Nt=length(sims[1][var]))
    fig = Figure()
    ax  = Axis(fig[1, 1])


    iter = Observable(1)
    for (is, sim) in enumerate(sims)
        vi = @lift(interior(sim[var][$iter], :, 1, 1))
        if labels === nothing
            label = "sim $is"
        else
            label = labels[is]
        end
        lines!(ax, vi; label)
    end

    axislegend(ax; position=:rt)

    record(fig, filename, 1:Nt, framerate=15) do i
        @info "Frame $i of $Nt"
        iter[] = i
    end
end

function plot_variable2(sims, var1, var2; 
                        filename="test.mp4",
                        labels=nothing,
                        Nt = length(sims[1][var1]))

    fig = Figure()
    ax  = Axis(fig[1, 1])

    iter = Observable(1)
    for (is, sim) in enumerate(sims)
        vi = @lift(interior(sim[var1][$iter], :, 1, 1))
        v2 = @lift(interior(sim[var2][$iter], :, 1, 1))
        if labels === nothing
            label = "sim $is"
        else
            label = labels[is]
        end
        lines!(ax, vi; label)
        lines!(ax, v2; label, linestyle = :dash)
    end

    axislegend(ax; position=:rt)

    record(fig, filename, 1:Nt, framerate=15) do i
        @info "Frame $i of $Nt"
        iter[] = i
    end
end