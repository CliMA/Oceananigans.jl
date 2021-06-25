ENV["GKSwstype"] = "nul"
using Plots
using Measures

using Printf
using Statistics
using CUDA

using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

#####
##### The Bickley jet
#####

Ψ(y) = - tanh(y)
U(y) = sech(y)^2

# A sinusoidal tracer
C(y, L) = sin(2π * y / L)

# Slightly off-center vortical perturbations
ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)

# Vortical velocity fields (ũ, ṽ) = (-∂_y, +∂_x) ψ̃
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2) 
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x) 

"""
    run_bickley_jet(output_time_interval = 2, stop_time = 200, arch = CPU(), Nh = 64, ν = 0,
                    momentum_advection = VectorInvariant())

Run the Bickley jet validation experiment until `stop_time` using `momentum_advection`
scheme or formulation, with horizontal resolution `Nh`, viscosity `ν`, on `arch`itecture.
"""
function run_bickley_jet(; output_time_interval = 2, stop_time = 200, arch = CPU(), Nh = 64, ν = 0, advection = WENO5())

    # Regular model
    grid = RegularRectilinearGrid(size=(Nh, Nh), halo=(3, 3),
                                  x = (-2π, 2π), y=(-2π, 2π),
                                  topology = (Periodic, Bounded, Flat))

    regular_model = IncompressibleModel(architecture = arch,
                                        advection = advection,
                                        timestepper = :RungeKutta3,
                                        grid = grid,
                                        tracers = :c,
                                        closure = IsotropicDiffusivity(ν=ν, κ=ν),
                                        coriolis = nothing,
                                        buoyancy = nothing)

    # Non-regular model
    solid(x, y, z) = y > 2π

    expanded_grid = RegularRectilinearGrid(size=(Nh, Int(5Nh/4)), halo=(3, 3),
                                           x = (-2π, 2π), y=(-2π, 3π),
                                           topology = (Periodic, Bounded, Flat))

    immersed_grid = ImmersedBoundaryGrid(expanded_grid, GridFittedBoundary(solid))

    immersed_model = IncompressibleModel(architecture = arch,
                                         advection = advection,
                                         timestepper = :RungeKutta3,
                                         grid = immersed_grid,
                                         tracers = (:c, :mass),
                                         closure = IsotropicDiffusivity(ν=ν, κ=ν),
                                         coriolis = nothing,
                                         buoyancy = nothing)

    # ** Initial conditions **
    #
    # u, v: Large-scale jet + vortical perturbations
    #    c: Sinusoid

    # Parameters
    ϵ = 0.1 # perturbation magnitude
    ℓ = 0.5 # Gaussian width
    k = 0.5 # Sinusoidal wavenumber

    # Total initial conditions
    uᵢ(x, y, z) = U(y) + ϵ * ũ(x, y, ℓ, k)
    vᵢ(x, y, z) = ϵ * ṽ(x, y, ℓ, k)
    cᵢ(x, y, z) = C(y, grid.Ly)

    set!(regular_model, u=uᵢ, v=vᵢ, c=cᵢ)
    set!(immersed_model, u=uᵢ, v=vᵢ, c=cᵢ, mass=1)

    wall_clock = [time_ns()]

    function progress(sim)
        @info(@sprintf("Iter: %d, time: %.1f, Δt: %.3f, wall time: %s, max|u|: %.2f",
                       sim.model.clock.iteration,
                       sim.model.clock.time,
                       sim.Δt.Δt,
                       prettytime(1e-9 * (time_ns() - wall_clock[1])),
                       maximum(abs, sim.model.velocities.u.data.parent)))

        wall_clock[1] = time_ns()

        return nothing
    end

    models = (immersed_model, regular_model)
    @show experiment_name = "bickley_jet_Nh_$(Nh)_$(typeof(regular_model.advection).name.wrapper)"

    for m in models
        wizard = TimeStepWizard(cfl=0.1, Δt=0.1 * grid.Δx, max_change=1.1, max_Δt=10.0)

        simulation = Simulation(m, Δt=wizard, stop_time=stop_time, iteration_interval=10, progress=progress)

        # Output: primitive fields + computations
        u, v, w, c = merge(m.velocities, m.tracers)
        ζ = ComputedField(∂x(v) - ∂y(u))
        outputs = merge(m.velocities, m.tracers, (ζ=ζ,))

        output_name = m.grid isa ImmersedBoundaryGrid ?
                            "immersed_" * experiment_name :
                            "regular_" * experiment_name

        @show output_name

        simulation.output_writers[:fields] =
            JLD2OutputWriter(m, outputs,
                             schedule = TimeInterval(output_time_interval),
                             prefix = output_name,
                             field_slicer = nothing,
                             force = true)

        @info "Running a simulation of an unstable Bickley jet with $(Nh)² degrees of freedom..."

        start_time = time_ns()

        run!(simulation)
    end

    return experiment_name 
end
    
"""
    visualize_bickley_jet(experiment_name)

Visualize the Bickley jet data associated with `experiment_name`.
"""
function visualize_immersed_bickley_jet(experiment_name)

    @info "Making a fun movie about an unstable Bickley jet..."

    regular_filepath = "regular_" * experiment_name * ".jld2"
    immersed_filepath = "immersed_" * experiment_name * ".jld2"

    regular_u_timeseries = FieldTimeSeries(regular_filepath,  "u")
    immersed_u_timeseries = FieldTimeSeries(immersed_filepath, "u")

    regular_v_timeseries = FieldTimeSeries(regular_filepath,  "v")
    immersed_v_timeseries = FieldTimeSeries(immersed_filepath, "v")

    regular_ζ_timeseries = FieldTimeSeries(regular_filepath,  "ζ")
    immersed_ζ_timeseries = FieldTimeSeries(immersed_filepath, "ζ")

    regular_c_timeseries = FieldTimeSeries(regular_filepath,  "c")
    immersed_c_timeseries = FieldTimeSeries(immersed_filepath, "c")

    regular_grid = regular_c_timeseries.grid
    immersed_grid = immersed_c_timeseries.grid

    xu, yu, zu = nodes(regular_u_timeseries)
    xv, yv, zv = nodes(regular_v_timeseries)
    xζ, yζ, zζ = nodes(regular_ζ_timeseries)
    xc, yc, zc = nodes(regular_c_timeseries)

    anim = @animate for (i, t) in enumerate(regular_c_timeseries.times)

        @info "    Plotting frame $i of $(length(regular_c_timeseries.times))..."

        Nx, Ny, Nz = size(regular_grid)

        regular_u = regular_u_timeseries[i]
        regular_v = regular_v_timeseries[i]
        regular_ζ = regular_ζ_timeseries[i]
        regular_c = regular_c_timeseries[i]

        immersed_u = immersed_u_timeseries[i]
        immersed_v = immersed_v_timeseries[i]
        immersed_ζ = immersed_ζ_timeseries[i]
        immersed_c = immersed_c_timeseries[i]

        regular_ui = interior(regular_u)[:, :, 1]
        regular_vi = interior(regular_v)[:, :, 1]
        regular_ζi = interior(regular_ζ)[:, :, 1]
        regular_ci = interior(regular_c)[:, :, 1]

        Nx, Ny = size(regular_ui)

        immersed_ui = interior(immersed_u)[1:Nx, 1:Ny, 1]
        immersed_vi = interior(immersed_v)[1:Nx, 1:Ny+1, 1]
        immersed_ζi = interior(immersed_ζ)[1:Nx, 1:Ny+1, 1]
        immersed_ci = interior(immersed_c)[1:Nx, 1:Ny, 1]

        difference_ui = immersed_ui .- regular_ui 
        difference_vi = immersed_vi .- regular_vi
        difference_ζi = immersed_ζi .- regular_ζi
        difference_ci = immersed_ci .- regular_ci

        kwargs = Dict(:aspectratio => 1,
                      :linewidth => 0,
                      :colorbar => :none,
                      :ticks => nothing,
                      :clims => (-1, 1),
                      :xlims => (-regular_grid.Lx/2, regular_grid.Lx/2),
                      :ylims => (-regular_grid.Ly/2, regular_grid.Ly/2))

        regular_u_plot = heatmap(xu, yu, clamp.(regular_ui, -1, 1)'; color = :balance, kwargs...)
        regular_v_plot = heatmap(xv, yv, clamp.(regular_vi, -1, 1)'; color = :balance, kwargs...)
        regular_ζ_plot = heatmap(xζ, yζ, clamp.(regular_ζi, -1, 1)'; color = :balance, kwargs...)
        regular_c_plot = heatmap(xc, yc, clamp.(regular_ci, -1, 1)'; color = :thermal, kwargs...)

        immersed_u_plot = heatmap(xu, yu, clamp.(immersed_ui, -1, 1)'; color = :balance, kwargs...)
        immersed_v_plot = heatmap(xv, yv, clamp.(immersed_vi, -1, 1)'; color = :balance, kwargs...)
        immersed_ζ_plot = heatmap(xζ, yζ, clamp.(immersed_ζi, -1, 1)'; color = :balance, kwargs...)
        immersed_c_plot = heatmap(xc, yc, clamp.(immersed_ci, -1, 1)'; color = :thermal, kwargs...)

        δlim = 0.1
        kwargs[:clims] = (-δlim, δlim)

        difference_u_plot = heatmap(xu, yu, clamp.(difference_ui, -δlim, δlim)'; color = :balance, kwargs...)
        difference_v_plot = heatmap(xv, yv, clamp.(difference_vi, -δlim, δlim)'; color = :balance, kwargs...)
        difference_ζ_plot = heatmap(xζ, yζ, clamp.(difference_ζi, -δlim, δlim)'; color = :balance, kwargs...)
        difference_c_plot = heatmap(xc, yc, clamp.(difference_ci, -δlim, δlim)'; color = :thermal, kwargs...)

        r_u_title = @sprintf("regular u at t = %.1f", t)
        r_v_title = @sprintf("regular v at t = %.1f", t)
        r_ζ_title = @sprintf("regular ζ at t = %.1f", t)
        r_c_title = @sprintf("regular c at t = %.1f", t)

        i_u_title = @sprintf("immersed u at t = %.1f", t)
        i_v_title = @sprintf("immersed v at t = %.1f", t)
        i_ζ_title = @sprintf("immersed ζ at t = %.1f", t)
        i_c_title = @sprintf("immersed c at t = %.1f", t)

        d_u_title = @sprintf("Δu at t = %.1f", t)
        d_v_title = @sprintf("Δv at t = %.1f", t)
        d_ζ_title = @sprintf("Δζ at t = %.1f", t)
        d_c_title = @sprintf("Δc at t = %.1f", t)

        plot(regular_u_plot, regular_v_plot, regular_ζ_plot, regular_c_plot,
             immersed_u_plot, immersed_v_plot, immersed_ζ_plot, immersed_c_plot,
             difference_u_plot, difference_v_plot, difference_ζ_plot, difference_c_plot,
             title = [r_u_title r_v_title r_ζ_title r_c_title i_u_title i_v_title i_ζ_title i_c_title d_u_title d_v_title d_ζ_title d_c_title],
             layout = (3, 4), size = (2000, 1000))
    end

    mp4(anim, "differences_" * experiment_name * ".mp4", fps = 8)
end

"""
    analyze_bickley_jet(experiment_name)

Analyze the Bickley jet data associated with `experiment_name`.
"""

function analyze_immersed_bickley_jet(experiment_name)

    @info "    Analyzing IBM Results for Velocity and Tracer Concentration... "

    regular_filepath = "regular_" * experiment_name * ".jld2"
    immersed_filepath = "immersed_" * experiment_name * ".jld2"

    regular_v_timeseries = FieldTimeSeries(regular_filepath,  "v")
    immersed_v_timeseries = FieldTimeSeries(immersed_filepath, "v")

    immersed_m_timeseries = FieldTimeSeries(immersed_filepath, "mass")
    regular_c_timeseries = FieldTimeSeries(regular_filepath,  "c")
    
    xv, yv, zv = nodes(regular_v_timeseries)
    xm, ym, zm = nodes(immersed_m_timeseries)
        
    # Finding the rms surface normal velocity (should be zero)
    function rms_normal(r_v_series, im_v_series)
        
        time_amt = length(r_v_series.times);
        last_y_fluid = size(r_v_series)[2];
        
        r_norm_rms = sqrt.(sum(r_v_series[:,last_y_fluid,1,:].^2, dims = 2)./time_amt)
        im_norm_rms = sqrt.(sum(im_v_series[:,last_y_fluid,1,:].^2, dims = 2)./time_amt)
        
        return r_norm_rms, im_norm_rms
    end

    regular_norm_rms, immersed_norm_rms = rms_normal(regular_v_timeseries, immersed_v_timeseries);
    
    # largest value for plotting
    max_norm = round(maximum(immersed_norm_rms), sigdigits = 2, RoundUp);
    
    @info "    Plotting the surface normal velocity..."

    norm_plot = plot(regular_v_timeseries.grid.xC, regular_norm_rms, label = "regular", yformatter = :scientific,
    color = :red, lw = 3, xlabel = "x", ylabel = "Vⁿ", legend = :bottomright)
    plot!(regular_v_timeseries.grid.xC, immersed_norm_rms, label = "immersed", color = :blue, lw = 3)
    plot!(yticks = 0:max_norm/5:max_norm, guidefontsize = 14, titlefont=14,legendfont = 10, 
        tickfont = 8)
        
    # Finding the area integrated concentration in time

    function area_int_concentration(r_m_series, im_m_series, xm, ym)
        
        last_y = size(r_m_series)[2];
        
        im_C = (sum(im_m_series[1:last_y,1:last_y,1,:], dims = 1:2)[1,1,:])*(ym[2]-ym[1])*(xm[2]-xm[1]);
        
        return im_C
    end

    immersed_concent = area_int_concentration(regular_c_timeseries, immersed_m_timeseries, xm, ym);
    
    #taking the percent leakage between IBM and nonIBM (which does not change from initial)
    percent_leakage_m = (abs.(immersed_concent[1] .- immersed_concent) ./ immersed_concent[1]) * 100;
    
    @info "    Plotting the tracer concentration leakage..."
    
    concent_diff = plot(immersed_m_timeseries.times, percent_leakage_m, yformatter = :scientific,
        color = :blue, xlabel = "t", ylabel = "% Concentration Leakage", legend = false, lw = 3,
        guidefontsize = 14, titlefont=14,legendfont = 10, tickfont = 8)
        
    results_plot = plot(norm_plot, concent_diff,
             title = ["Normal Velocity" "% Change in Area Integrated Concentration"],
             layout = (1, 2), size = (1400, 500), left_margin=10mm, bottom_margin=10mm)
    
    Plots.savefig(results_plot, "Analysis_" * experiment_name * ".png")
    
end


advection = WENO5()
experiment_name = run_bickley_jet(advection=advection, Nh=64, stop_time=200)
visualize_immersed_bickley_jet(experiment_name)
analyze_immersed_bickley_jet(experiment_name)
