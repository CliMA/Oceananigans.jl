using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: VelocityStencil, VorticityStencil

using Printf
using GLMakie

include("bickley_utils.jl")

"""
    run_bickley_jet(output_time_interval = 2, stop_time = 200, arch = CPU(), Nh = 64, ν = 0,
                    momentum_advection = VectorInvariant())

Run the Bickley jet validation experiment until `stop_time` using `momentum_advection`
scheme or formulation, with horizontal resolution `Nh`, viscosity `ν`, on `arch`itecture.
"""
function run_bickley_jet(;
                         output_time_interval = 2,
                         stop_time = 200,
                         arch = CPU(),
                         Nh = 64, 
                         free_surface = ImplicitFreeSurface(gravitational_acceleration=10.0),
                         momentum_advection = WENO5(),
                         tracer_advection = WENO5(),
                         experiment_name = string(nameof(typeof(momentum_advection))))

    grid = bickley_grid(; arch, Nh)
    model = HydrostaticFreeSurfaceModel(; grid, momentum_advection, tracer_advection,
                                        free_surface, tracers = :c, buoyancy=nothing)
    set_bickley_jet!(model)

    Δt = 0.2 * 2π / Nh
    wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=10.0)
    simulation = Simulation(model; Δt, stop_time)

    progress(sim) = @printf("Iter: %d, time: %.1f, Δt: %.1e, max|u|: %.3f, max|η|: %.3f\n",
                            iteration(sim), time(sim), sim.Δt,
                            maximum(abs, model.velocities.u),
                            maximum(abs, model.free_surface.η))

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=10.0)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    # Output: primitive fields + computations
    u, v, w = model.velocities
    outputs = merge(model.velocities, model.tracers, (ζ=∂x(v) - ∂y(u), η=model.free_surface.η))

    @show output_name = "bickley_jet_Nh_$(Nh)_" * experiment_name

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, outputs,
                                schedule = TimeInterval(output_time_interval),
                                prefix = output_name,
                                force = true)

    @info "Running a simulation of an unstable Bickley jet with $(Nh)² degrees of freedom..."

    start_time = time_ns()

    run!(simulation)

    elapsed = 1e-9 * (time_ns() - start_time)
    @info "... the bickley jet simulation took " * prettytime(elapsed)

    return output_name
end
    
"""
    visualize_bickley_jet(experiment_name)

Visualize the Bickley jet data in `name * ".jld2"`.
"""
function visualize_bickley_jet(name)
    @info "Making a fun movie about an unstable Bickley jet..."

    filepath = name * ".jld2"

    ζt = FieldTimeSeries(filepath, "ζ")
    ct = FieldTimeSeries(filepath, "c")
    t = ζt.times
    Nt = length(t)

    fig = Figure(resolution=(1400, 800))
    slider = Slider(fig[2, 1:2], range=1:Nt, startvalue=1)
    n = slider.value

    ζtitle = @lift @sprintf("ζ at t = %.1f", t[$n])
    ctitle = @lift @sprintf("c at t = %.1f", t[$n])

    ax_ζ = Axis(fig[1, 1], title=ζtitle, aspect=1)
    ax_c = Axis(fig[1, 2], title=ctitle, aspect=1)

    ζ = @lift interior(ζt[$n], :, :, 1)
    c = @lift interior(ct[$n], :, :, 1)

    heatmap!(ax_ζ, ζ, colorrange=(-1, 1), colormap=:redblue)
    heatmap!(ax_c, c, colorrange=(-1, 1), colormap=:thermal)

    record(fig, name * ".mp4", 1:Nt, framerate=24) do nn
        @info "Drawing frame $nn of $Nt..."
        n[] = nn
    end
end

advection_schemes = [CenteredSecondOrder(), WENO5()]

#=
advection_schemes = [WENO5(vector_invariant=VelocityStencil()),
                     WENO5(vector_invariant=VorticityStencil()),
                     WENO5(),
                     VectorInvariant()]
=#

arch = CPU()
for Nh in [128]
    for momentum_advection in advection_schemes
        name = run_bickley_jet(; arch, momentum_advection, Nh)
        visualize_bickley_jet(name)
    end
end
