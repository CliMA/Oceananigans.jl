ENV["GKSwstype"] = "nul"

using Plots

using Printf
using Statistics
using CUDA

using Oceananigans
using Oceananigans.Advection
using Oceananigans.Units

using Oceananigans.MultiRegion

using Oceananigans.Advection: EnergyConservingScheme
using Oceananigans.OutputReaders: FieldTimeSeries

using Oceananigans.Advection: ZWENO, WENOVectorInvariantVel, WENOVectorInvariantVort, VectorInvariant, VelocityStencil, VorticityStencil
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom   
using Oceananigans.Operators: Δx, Δy
using Oceananigans.TurbulenceClosures

include("../bickley_jet/bickley_utils.jl")

"""
    run_bickley_jet(output_time_interval = 2, stop_time = 200, arch = CPU(), Nh = 64, ν = 0,
                    momentum_advection = VectorInvariant())

Run the Bickley jet validation experiment until `stop_time` using `momentum_advection`
scheme or formulation, with horizontal resolution `Nh`, viscosity `ν`, on `arch`itecture.
"""
function run_bickley_jet(; output_time_interval = 2, stop_time = 200, arch = CPU(), Nh = 64, 
                           momentum_advection = VectorInvariant())

    grid = bickley_grid(arch=arch, Nh=Nh, halo=(4, 4, 4))
    
    @inline toplft(x, y) = (((x > π/2) & (x < 3π/2)) & (((y > π/3) & (y < 2π/3)) | ((y > 4π/3) & (y < 5π/3))))
    @inline botlft(x, y) = (((x > π/2) & (x < 3π/2)) & (((y < -π/3) & (y > -2π/3)) | ((y < -4π/3) & (y > -5π/3))))
    @inline toprgt(x, y) = (((x < -π/2) & (x > -3π/2)) & (((y > π/3) & (y < 2π/3)) | ((y > 4π/3) & (y < 5π/3))))
    @inline botrgt(x, y) = (((x < -π/2) & (x > -3π/2)) & (((y < -π/3) & (y > -2π/3)) | ((y < -4π/3) & (y > -5π/3))))
    @inline bottom(x, y) = Int(toplft(x, y) | toprgt(x, y) | botlft(x, y) | botrgt(x, y))

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -1))

    mrg  = MultiRegionGrid(grid, partition=XPartition(2), devices=(0, 1))
    c = sqrt(10.0)
    Δt = 0.1 * grid.Δxᶜᵃᵃ / c

    timescale = (5days / (6minutes) * Δt)
    @show prettytime(timescale)

    @inline νhb(i, j, k, grid, lx, ly, lz) = (1 / (1 / Δx(i, j, k, grid, lx, ly, lz)^2 + 1 / Δy(i, j, k, grid, lx, ly, lz)^2 ))^2 / timescale
    biharmonic_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=νhb, discrete_form=true) 

    model = HydrostaticFreeSurfaceModel(momentum_advection = momentum_advection,
                                        tracer_advection = WENO5(),
                                        grid = mrg,
                                        tracers = :c,
                                        closure = nothing,
                                        free_surface = ExplicitFreeSurface(gravitational_acceleration=10.0),
                                        coriolis = nothing,
                                        buoyancy = nothing)

    # ** Initial conditions **
    #
    # u, v: Large-scale jet + vortical perturbations
    #    c: Sinusoid
    
    set_bickley_jet!(model)

    wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=10.0)

    simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

    progress(sim) = @printf("Iter: %d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f \n",
                            iteration(sim), prettytime(sim), prettytime(sim.Δt),
                            maximum(abs, model.velocities.u), maximum(abs, model.free_surface.η))

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))
    wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=10.0)

    simulation.callbacks[:wizard]   = Callback(wizard, IterationInterval(10))

    # Output: primitive fields + computations
    u, v, w, c = merge(model.velocities, model.tracers)

    ζ = u

    outputs = merge(model.velocities, model.tracers, (ζ=ζ, η=model.free_surface.η))

    name = typeof(model.advection.momentum).name.wrapper
    if model.advection.momentum isa WENOVectorInvariantVel
        name = "WENOVectorInvariantVel"
    end
    if model.advection.momentum isa WENOVectorInvariantVort
        name = "WENOVectorInvariantVort"
    end

    @show experiment_name = "bickley_jet_Nh_$(Nh)_Upwind"

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, outputs,
                                schedule = TimeInterval(output_time_interval),
                                prefix = experiment_name,
                                force = true)

    @info "Running a simulation of an unstable Bickley jet with $(Nh)² degrees of freedom..."

    start_time = time_ns()

    run!(simulation)

    return experiment_name 
end
    
"""
    visualize_bickley_jet(experiment_name)

Visualize the Bickley jet data associated with `experiment_name`.
"""
function visualize_bickley_jet(experiment_name)

    @info "Making a fun movie about an unstable Bickley jet..."

    filepath = experiment_name * ".jld2"

    ζ_timeseries = FieldTimeSeries(filepath, "ζ", boundary_conditions=nothing, location=(Face, Face, Center))
    c_timeseries = FieldTimeSeries(filepath, "c", boundary_conditions=nothing, location=(Face, Center, Center))

    grid = c_timeseries.grid

    xζ, yζ, zζ = nodes(ζ_timeseries)
    xc, yc, zc = nodes(c_timeseries)

    anim = @animate for (i, iteration) in enumerate(c_timeseries.times)

        @info "    Plotting frame $i from iteration $iteration..."

        ζ = ζ_timeseries[i]
        c = c_timeseries[i]
        t = ζ_timeseries.times[i]

        ζi = interior(ζ)[:, :, 1]
        ci = interior(c)[:, :, 1]

        kwargs = Dict(
                      :aspectratio => 1,
                      :linewidth => 0,
                      :colorbar => :none,
                      :ticks => nothing,
                      :clims => (-1, 1),
                      :xlims => (-grid.Lx/2, grid.Lx/2),
                      :ylims => (-grid.Ly/2, grid.Ly/2)
                     )

        ζ_plot = heatmap(xζ, yζ, clamp.(ζi, -1, 1)'; color = :balance, kwargs...)
        c_plot = heatmap(xc, yc, clamp.(ci, -1, 1)'; color = :thermal, kwargs...)

        ζ_title = @sprintf("ζ at t = %.1f", t)
        c_title = @sprintf("u at t = %.1f", t)

        plot(ζ_plot, c_plot, title = [ζ_title c_title], size = (4000, 2000))
    end

    mp4(anim, experiment_name * ".mp4", fps = 8)
end

advection_schemes = [WENO5(vector_invariant=VelocityStencil()),
                     WENO5(vector_invariant=VorticityStencil()),
                     WENO5(),
                     VectorInvariant()]

advection_schemes = [WENO5(vector_invariant = VelocityStencil())]

for Nx in [128]
    for advection in advection_schemes
        experiment_name = run_bickley_jet(arch=GPU(), momentum_advection=advection, Nh=Nx)
        visualize_bickley_jet(experiment_name)
    end
end
