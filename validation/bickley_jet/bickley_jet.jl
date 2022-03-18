ENV["GKSwstype"] = "nul"
# using Plots

using Printf
using Statistics
using CUDA

using Oceananigans
using Oceananigans.Advection
using Oceananigans.AbstractOperations
using Oceananigans.OutputWriters
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Utils: prettytime
using Oceananigans.MultiRegion

using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, ExplicitFreeSurface
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant

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
function run_bickley_jet(; output_time_interval = 2, stop_time = 200, arch = CPU(), Nh = 64, ν = 0,
                           momentum_advection = VectorInvariant(), devices = nothing)

    grid = RectilinearGrid(arch, size=(Nh, Nh, 1),
                                x = (-2π, 2π), y=(-2π, 2π), z=(0, 1),
                                topology = (Periodic, Periodic, Bounded))

    if devices isa Nothing
        mrg = grid
    else
        mrg = MultiRegionGrid(grid, partition = XPartition(length(devices)), devices = devices)
    end

    model = HydrostaticFreeSurfaceModel(momentum_advection = momentum_advection,
                                          tracer_advection = WENO5(),
                                                      grid = mrg,
                                                   tracers = :c,
                                                   closure = ScalarDiffusivity(ν=ν, κ=ν),
                                              free_surface = ExplicitFreeSurface(gravitational_acceleration=10.0),
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

    set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

    c = sqrt(model.free_surface.gravitational_acceleration)
    Δt = 0.1 * grid.Δxᶜᵃᵃ / c

    simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

    progress(sim) = @printf("Iter: %d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f \n",
                            iteration(sim), prettytime(sim), prettytime(sim.Δt),
                            maximum(abs, model.velocities.u), maximum(abs, model.free_surface.η))

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    # Output: primitive fields + computations
    u, v, w, c = merge(model.velocities, model.tracers)


    outputs = merge(model.velocities, model.tracers)

    @show experiment_name = "bickley_jet_Nh_$(Nh)_$(typeof(model.advection.momentum).name.wrapper)"

    # simulation.output_writers[:fields] =
    #     JLD2OutputWriter(model, outputs,
    #                             schedule = TimeInterval(output_time_interval),
    #                             prefix = experiment_name,
    #                             force = true)

    @info "Running a simulation of an unstable Bickley jet with $(Nh)² degrees of freedom..."

    start_time = time_ns()

    # run!(simulation)

    return simulation
end
    
"""
    visualize_bickley_jet(experiment_name)

Visualize the Bickley jet data associated with `experiment_name`.
# """
# function visualize_bickley_jet(experiment_name)

#     @info "Making a fun movie about an unstable Bickley jet..."

#     filepath = experiment_name * ".jld2"

#     ζ_timeseries = DiskTimeSeries(:ζ, filepath)
#     c_timeseries = DiskTimeSeries(:c, filepath)

#     grid = c_timeseries.grid

#     xζ, yζ, zζ = nodes(ζ_timeseries)
#     xc, yc, zc = nodes(c_timeseries)

#     anim = @animate for (i, iteration) in enumerate(c_timeseries.iterations)

#         @info "    Plotting frame $i from iteration $iteration..."

#         ζ = ζ_timeseries[i]
#         c = c_timeseries[i]
#         t = ζ_timeseries.times[i]

#         ζi = interior(ζ)[:, :, 1]
#         ci = interior(c)[:, :, 1]

#         kwargs = Dict(
#                       :aspectratio => 1,
#                       :linewidth => 0,
#                       :colorbar => :none,
#                       :ticks => nothing,
#                       :clims => (-1, 1),
#                       :xlims => (-grid.Lx/2, grid.Lx/2),
#                       :ylims => (-grid.Ly/2, grid.Ly/2)
#                      )

#         ζ_plot = heatmap(xζ, yζ, clamp.(ζi, -1, 1)'; color = :balance, kwargs...)
#         c_plot = heatmap(xc, yc, clamp.(ci, -1, 1)'; color = :thermal, kwargs...)

#         ζ_title = @sprintf("ζ at t = %.1f", t)
#         c_title = @sprintf("c at t = %.1f", t)

#         plot(ζ_plot, c_plot, title = [ζ_title c_title], size = (4000, 2000))
#     end

#     mp4(anim, experiment_name * ".mp4", fps = 8)
# end

# for momentum_advection in (WENO5(), CenteredSecondOrder(), VectorInvariant())
#     experiment_name = run_bickley_jet(momentum_advection=momentum_advection, Nh=64)
#     visualize_bickley_jet(experiment_name)
# end

simulation_serial = run_bickley_jet(arch = GPU(), Nh=12)
simulation_parall = run_bickley_jet(devices=(0, 1), Nh=12)