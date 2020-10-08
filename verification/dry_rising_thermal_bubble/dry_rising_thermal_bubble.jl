"""
This example sets up a dry, warm thermal bubble perturbation in a uniform
lateral mean flow which buoyantly rises.
"""

using Printf
using Plots
using VideoIO
using FileIO
using JULES
using Oceananigans

using Oceananigans.Fields: interiorparent
interiorxz(field) = dropdims(interiorparent(field), dims=2)

const km = 1000.0
const hPa = 100.0

function simulate_dry_rising_thermal_bubble(; thermodynamic_variable, end_time=1000.0, make_plots=true)
    tvar = thermodynamic_variable

    Lx = 20km
    Lz = 10km
    Δ  = 0.2km  # grid spacing [m]

    Nx = Int(Lx/Δ)
    Ny = 1
    Nz = Int(Lz/Δ)

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), halo=(2, 2, 2),
                                x=(-Lx/2, Lx/2), y=(-Lx/2, Lx/2), z=(0, Lz))

    model = CompressibleModel(
                          grid = grid,
                         gases = DryEarth(),
        thermodynamic_variable = tvar,
                       closure = IsotropicDiffusivity(ν=75.0, κ=75.0)
    )

    #####
    ##### Dry thermal bubble perturbation
    #####

    gas = model.gases.ρ
    R, cₚ, cᵥ = gas.R, gas.cₚ, gas.cᵥ
    g  = model.gravity
    pₛ = 1000hPa
    Tₛ = 300.0

    # Define an approximately hydrostatic background state
    θ₀(x, y, z) = Tₛ
    p₀(x, y, z) = pₛ * (1 - g*z / (cₚ*Tₛ))^(cₚ/R)
    T₀(x, y, z) = Tₛ * (p₀(x, y, z)/pₛ)^(R/cₚ)
    ρ₀(x, y, z) = p₀(x, y, z) / (R*T₀(x, y, z))

    # Define both energy and entropy
    uᵣ, Tᵣ, ρᵣ, sᵣ = gas.u₀, gas.T₀, gas.ρ₀, gas.s₀  # Reference values
    ρe₀(x, y, z) = ρ₀(x, y, z) * (uᵣ + cᵥ * (T₀(x, y, z) - Tᵣ) + g*z)
    ρs₀(x, y, z) = ρ₀(x, y, z) * (sᵣ + cᵥ * log(T₀(x, y, z)/Tᵣ) - R * log(ρ₀(x, y, z)/ρᵣ))

    # Define the initial density perturbation
    xᶜ, zᶜ = 0km, 2km
    xʳ, zʳ = 2km, 2km

    L(x, y, z) = sqrt(((x - xᶜ)/xʳ)^2 + ((z - zᶜ)/zʳ)^2)

    function ρ′(x, y, z; θᶜ′ = 2.0)
        l = L(x, y, z)
        θ′ = (l <= 1) * θᶜ′ * cos(π/2 * L(x, y, z))^2
        return -ρ₀(x, y, z) * θ′ / θ₀(x, y, z)
    end

    # Define initial state
    ρᵢ(x, y, z) = ρ₀(x, y, z) + ρ′(x, y, z)
    pᵢ(x, y, z) = p₀(x, y, z)
    Tᵢ(x, y, z) = pᵢ(x, y, z) / (R * ρᵢ(x, y, z))

    ρeᵢ(x, y, z) = ρᵢ(x, y, z) * (uᵣ + cᵥ * (Tᵢ(x, y, z) - Tᵣ) + g*z)
    ρsᵢ(x, y, z) = ρᵢ(x, y, z) * (sᵣ + cᵥ * log(Tᵢ(x, y, z)/Tᵣ) - R * log(ρᵢ(x, y, z)/ρᵣ))

    # Set hydrostatic background state
    set!(model.tracers.ρ, ρ₀)
    tvar isa Energy  && set!(model.tracers.ρe, ρe₀)
    tvar isa Entropy && set!(model.tracers.ρs, ρs₀)
    update_total_density!(model)

    # Save hydrostatic base state
    ρʰᵈ = interiorxz(model.total_density)
    tvar isa Energy  && (eʰᵈ = interiorxz(model.lazy_tracers.e))
    tvar isa Entropy && (sʰᵈ = interiorxz(model.lazy_tracers.s))

    # Set initial state (which includes the thermal perturbation)
    set!(model.tracers.ρ, ρᵢ)
    tvar isa Energy  && set!(model.tracers.ρe, ρeᵢ)
    tvar isa Entropy && set!(model.tracers.ρs, ρsᵢ)
    update_total_density!(model)

    if make_plots
        ρ_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km,
                         rotr90(interiorxz(model.total_density) .- ρʰᵈ),
                         fill=true, levels=10, xlims=(-5, 5),
                         clims=(-0.008, 0.008), color=:balance, dpi=200)
        savefig(ρ_plot, "rho_prime_initial_condition_with_$(typeof(tvar)).png")

        if tvar isa Energy
            e_slice = rotr90(interiorxz(model.lazy_tracers.e))
            e_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km, e_slice,
                             fill=true, levels=10, xlims=(-5, 5), color=:thermal, dpi=200)
            savefig(e_plot, "energy_initial_condition.png")
        elseif tvar isa Entropy
            s_slice = rotr90(interiorxz(model.lazy_tracers.s))
            s_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km, s_slice,
                             fill=true, levels=10, xlims=(-5, 5), color=:thermal, dpi=200)
            savefig(s_plot, "entropy_initial_condition.png")
        end
    end

    #####
    ##### Watch the thermal bubble rise!
    #####

    # Initial mean ρ, ρe, ρs
    ρ̄ᵢ  = sum(interior(model.total_density)) / (Nx*Ny*Nz)
    tvar isa Energy  && (ρ̄ēᵢ = sum(interior(model.tracers.ρe)) / (Nx*Ny*Nz))
    tvar isa Entropy && (ρ̄s̄ᵢ = sum(interior(model.tracers.ρs)) / (Nx*Ny*Nz))

    if tvar isa Energy
        sim_parameters = (make_plots=make_plots, ρʰᵈ=ρʰᵈ, eʰᵈ=eʰᵈ, ρ̄ᵢ=ρ̄ᵢ, ρ̄ēᵢ=ρ̄ēᵢ)
    elseif tvar isa Entropy
        sim_parameters = (make_plots=make_plots, ρʰᵈ=ρʰᵈ, ρ̄ᵢ=ρ̄ᵢ, ρ̄s̄ᵢ=ρ̄s̄ᵢ)
    end

    simulation = Simulation(model, Δt=0.1, stop_time=end_time, progress_frequency=50,
                            progress=print_progress_and_make_plots, parameters=sim_parameters)
    run!(simulation)

    # Print min/max of ρ′ and w at t = 1000.
    ρ′₁₀₀₀ = interiorxz(model.tracers.ρ) .- ρʰᵈ
    w₁₀₀₀  = interiorxz(model.velocities.w)

    @printf("ρ′(t=1000): min=%.2e, max=%.2e\n", minimum(ρ′₁₀₀₀), maximum(ρ′₁₀₀₀))
    @printf(" w(t=1000): min=%.2e, max=%.2e\n", minimum(w₁₀₀₀), maximum(w₁₀₀₀))

    if make_plots
        @printf("Rendering MP4...\n")
        imgs = filter(x -> occursin("$(typeof(tvar))", x) && occursin(".png", x), readdir("frames"))
        imgorder = map(x -> split(split(x, ".")[1], "_")[end], imgs)
        p = sortperm(parse.(Int, imgorder))

        frames = []
        for img in imgs[p]
            push!(frames, convert.(RGB, load("frames/$img")))
        end

        encodevideo("thermal_bubble_$(typeof(tvar)).mp4", frames, framerate = 30)
    end

    return simulation
end

function print_progress_and_make_plots(simulation)
    model, Δt = simulation.model, simulation.Δt
    tvar = model.thermodynamic_variable
    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz

    if tvar isa Energy
        make_plots, ρʰᵈ, eʰᵈ, ρ̄ᵢ, ρ̄ēᵢ = simulation.parameters
    elseif tvar isa Entropy
        make_plots, ρʰᵈ, ρ̄ᵢ, ρ̄s̄ᵢ = simulation.parameters
    end

    ρ̄ = sum(interior(model.total_density)) / (Nx*Ny*Nz)

    if tvar isa Energy
        ρ̄ē = sum(interior(model.tracers.ρe)) / (Nx*Ny*Nz)
        @printf("t = %.2f s, CFL = %.2e, ρ̄ = %.2e (rerr = %.2e), ρ̄ē = %.2e (rerr = %.2e)\n",
                model.clock.time, cfl(model, Δt), ρ̄, (ρ̄ - ρ̄ᵢ)/ρ̄, ρ̄ē, (ρ̄ē - ρ̄ēᵢ)/ρ̄ē)
    elseif tvar isa Entropy
        ρ̄s̄ = sum(interior(model.tracers.ρs)) / (Nx*Ny*Nz)
        @printf("t = %.2f s, CFL = %.2e, ρ̄ = %.2e (rerr = %.2e), ρ̄s̄ = %.2e (rerr = %.2e)\n",
                model.clock.time, cfl(model, Δt), ρ̄, (ρ̄ - ρ̄ᵢ)/ρ̄, ρ̄s̄, (ρ̄s̄ - ρ̄s̄ᵢ)/ρ̄s̄)
    end

    if simulation.parameters.make_plots
        xC, yC, zC = model.grid.xC ./ km, model.grid.yC ./ km, model.grid.zC ./ km
        xF, yF, zF = model.grid.xF ./ km, model.grid.yF ./ km, model.grid.zF ./ km

        u_slice = rotr90(interiorxz(model.velocities.u))
        w_slice = rotr90(interiorxz(model.velocities.w))
        ρ_slice = rotr90(interiorxz(model.total_density) .- ρʰᵈ)

        u_title = @sprintf("u, t = %d s", round(Int, model.clock.time))
        u_plot = heatmap(xC, zC, u_slice, title=u_title, fill=true, levels=50,
                         xlims=(-5, 5), color=:balance, linecolor=nothing, clims=(-10, 10))
        w_plot = heatmap(xC, zF, w_slice, title="w", fill=true, levels=50,
                         xlims=(-5, 5), color=:balance, linecolor=nothing, clims=(-10, 10))
        ρ_plot = heatmap(xC, zC, ρ_slice, title="rho_prime", fill=true, levels=50,
                         xlims=(-5, 5), color=:balance, linecolor=nothing, clims=(-0.007, 0.007))

        if tvar isa Energy
            e_slice = rotr90((interiorxz(model.lazy_tracers.e) .- eʰᵈ) ./ interiorxz(model.total_density))
            tvar_plot = heatmap(xC, zC, e_slice, title="e_prime", fill=true, levels=50,
                                xlims=(-5, 5), color=:oxy_r, linecolor=nothing, clims = (0, 1200))
        elseif tvar isa Entropy
            s_slice = rotr90(interiorxz(model.lazy_tracers.s))
            tvar_plot = heatmap(xC, zC, s_slice, title="s", fill=true, levels=50,
                                xlims=(-5, 5), color=:oxy_r, linecolor = nothing, clims=(99, 105))
        end

        p = plot(u_plot, w_plot, ρ_plot, tvar_plot, layout=(2, 2), dpi=200, show=true)

        n = Int(model.clock.iteration / simulation.progress_frequency)
        n == 1 && !isdir("frames") && mkdir("frames")
        savefig(p, @sprintf("frames/thermal_bubble_%s_%03d.png", typeof(tvar), n))
    end

    return nothing
end
