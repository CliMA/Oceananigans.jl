ENV["GKSwstype"] = "nul"
using Plots

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

# Vortical velocity fields (ũ, ṽ) = (-∂_y, +∂_x) ψ̃
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2) 
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x) 

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
                                        grid = grid,
                                        tracers = :c,
                                        closure = IsotropicDiffusivity(ν=ν, κ=ν),
                                        coriolis = nothing,
                                        buoyancy = nothing)

    # Non-regular model
    solid(x, y, z) = y >= 2π

    expanded_grid = RegularRectilinearGrid(size=(Nh, Int(5Nh/4)), halo=(3, 3),
                                           x = (-2π, 2π), y=(-2π, 3π),
                                           topology = (Periodic, Bounded, Flat))

    immersed_grid = ImmersedBoundaryGrid(expanded_grid, GridFittedBoundary(solid))

    immersed_model = IncompressibleModel(architecture = arch,
                                         advection = advection,
                                         grid = immersed_grid,
                                         tracers = :c,
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
    uᵢ(x, y, z) = U(y) + ϵ * ũ(x, y, ℓ, k)
    vᵢ(x, y, z) = ϵ * ṽ(x, y, ℓ, k)
    cᵢ(x, y, z) = C(y, grid.Ly)

    set!(regular_model, u=uᵢ, v=vᵢ, c=cᵢ)
    set!(immersed_model, u=uᵢ, v=vᵢ, c=cᵢ)

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
function visualize_bickley_jet(output_name)

    @info "Making a fun movie about an unstable Bickley jet..."

    filepath = output_name * ".jld2"

    ζ_timeseries = FieldTimeSeries(filepath, "ζ")
    c_timeseries = FieldTimeSeries(filepath, "c")

    grid = c_timeseries.grid

    xζ, yζ, zζ = nodes(ζ_timeseries)
    xc, yc, zc = nodes(c_timeseries)

    anim = @animate for (i, t) in enumerate(c_timeseries.times)

        @info "    Plotting frame $i of $(length(c_timeseries.times))..."

        ζ = ζ_timeseries[i]
        c = c_timeseries[i]

        ζi = interior(ζ)[:, :, 1]
        ci = interior(c)[:, :, 1]

        kwargs = Dict(:aspectratio => 1,
                      :linewidth => 0,
                      :colorbar => :none,
                      :ticks => nothing,
                      :clims => (-1, 1),
                      :xlims => (-grid.Lx/2, grid.Lx/2),
                      :ylims => (-grid.Ly/2, grid.Ly/2))

        ζ_plot = heatmap(xζ, yζ, clamp.(ζi, -1, 1)'; color = :balance, kwargs...)
        c_plot = heatmap(xc, yc, clamp.(ci, -1, 1)'; color = :thermal, kwargs...)

        ζ_title = @sprintf("ζ at t = %.1f", t)
        c_title = @sprintf("c at t = %.1f", t)

        plot(ζ_plot, c_plot, title = [ζ_title c_title], size = (4000, 2000))
    end

    mp4(anim, output_name * ".mp4", fps = 8)
end

"""
    visualize_bickley_jet(experiment_name)

Visualize the Bickley jet data associated with `experiment_name`.
"""
function visualize_differences(experiment_name)

    @info "Making a fun movie about the differences between a regular and immersed simulation of an unstable Bickley jet..."

    regular_filepath = "regular_" * experiment_name * ".jld2"
    immersed_filepath = "immersed_" * experiment_name * ".jld2"

    regular_c_timeseries = FieldTimeSeries(regular_filepath, "c")
    immersed_c_timeseries = FieldTimeSeries(immersed_filepath, "c")

    regular_grid = regular_c_timeseries.grid
    immersed_grid = immersed_c_timeseries.grid

    xc, yc, zc = nodes(regular_c_timeseries)

    Nx, Ny, Nz = size(regular_grid)

    anim = @animate for (i, t) in enumerate(c_timeseries.times)

        @info "    Plotting frame $i of $(length(c_timeseries.times))..."

        regular_c = regular_c_timeseries[i]
        immersed_c = immersed_c_timeseries[i]

        regular_ci = interior(regular_c)[:, :, 1]
        immersed_ci = interior(immersed_c)[1:Nx, 1:Ny, 1]

        δc = regular_ci .- immersed_ci

        clim = 0.01

        kwargs = Dict(:aspectratio => 1,
                      :linewidth => 0,
                      :size => (800, 600),
                      :colorbar => :none,
                      :ticks => nothing,
                      :title => @sprintf("Δc at t = %.1f", t),
                      :clims => (-clim, clim),
                      :xlims => (-grid.Lx/2, grid.Lx/2),
                      :ylims => (-grid.Ly/2, grid.Ly/2))
        
        heatmap(xc, yc, clamp.(δc, -clim, clim)'; color = :thermal, kwargs...)
    end

    mp4(anim, "regular_immersed_different_" * experiment_name * ".mp4", fps = 8)
end

advection = WENO5()
experiment_name = run_bickley_jet(advection=advection, Nh=128, stop_time=55)
experiment_name = "bickley_jet_Nh_128_WENO5"
visualize_bickley_jet("immersed_" * experiment_name)
visualize_bickley_jet("regular_" * experiment_name)
