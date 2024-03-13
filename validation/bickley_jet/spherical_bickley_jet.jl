ENV["GKSwstype"] = "nul"

using Printf
using Statistics
using Plots

using Oceananigans
using Oceananigans.Operators: ζ₃ᶠᶠᶜ
using Oceananigans.Advection
using Oceananigans.AbstractOperations
using Oceananigans.OutputWriters
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Utils: prettytime
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, ExplicitFreeSurface
using Oceananigans.OutputReaders: FieldTimeSeries

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom

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
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + (y + ℓ/10) / ℓ^2)
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x) 

"""
    run_bickley_jet(output_time_interval = 2, stop_time = 200, arch = CPU(), Nh = 64, ν = 0,
                    momentum_advection = VectorInvariant())

Run the Bickley jet validation experiment until `stop_time` using `momentum_advection`
scheme or formulation, with horizontal resolution `Nh`, viscosity `ν`, on `arch`itecture.
"""
function run_bickley_jet(; output_time_interval = 2, stop_time = 200, arch = CPU(), Nh = 64, ν = 0,
                           momentum_advection = VectorInvariant(),
                           experiment_name = string(nameof(typeof(momentum_advection))))

    Nφ = Int(Nh/2)

    grid = LatitudeLongitudeGrid(arch, size=(Nh, Nφ, 1),
                                 radius = 1,
                                 longitude = (-180, 180), latitude=(-80, 80), z=(0, 1), halo = (5, 5, 5),
                                 precompute_metrics = true)

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -1))
                            
    free_surface = ExplicitFreeSurface(gravitational_acceleration=1)

    model = HydrostaticFreeSurfaceModel(momentum_advection = momentum_advection,
                                          tracer_advection = WENO(),
                                                      grid = grid,
                                                   tracers = :c,
                                                   closure = nothing,
                                              free_surface = free_surface,
                                                  coriolis = HydrostaticSphericalCoriolis(rotation_rate=1),
                                                  buoyancy = nothing)

    # ** Initial conditions **
    #
    # u, v: Large-scale jet + vortical perturbations
    #    c: Sinusoid

    # Parameters
    ϵ = 0.1 # perturbation magnitude
    ℓ = 0.5 # Gaussian width
    k = 0.5 # Sinusoidal wavenumber

    dr(x) = deg2rad(x)

    # Total initial conditions
    uᵢ(x, y, z) = U(dr(y)*8) + ϵ * ũ(dr(x)*2, dr(y)*8, ℓ, k)
    vᵢ(x, y, z) = ϵ * ṽ(dr(x)*2, dr(y)*4, ℓ, k)
    cᵢ(x, y, z) = C(dr(y)*8, grid.Ly)

    set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

    wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=10.0)

    c = sqrt(model.free_surface.gravitational_acceleration)
    Δt = 0.1 * Array(model.grid.Δxᶜᶠᵃ.parent)[5] / c

    simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

    progress(sim) = @printf("Iter: %d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f \n",
                            iteration(sim), prettytime(sim), prettytime(sim.Δt),
                            maximum(abs, model.velocities.u), maximum(abs, model.free_surface.η))

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=10.0)

    simulation.callbacks[:wizard]   = Callback(wizard, IterationInterval(10))

    # Output: primitive fields + computations
    u, v, w, c = merge(model.velocities, model.tracers)

    ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v)

    ζ = Field(ζ_op)
    outputs = merge(model.velocities, model.tracers, (; ζ=ζ))

    @show output_name = "spherical_bickley_jet_Nh_$(Nh)_" * experiment_name

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, outputs,
                         schedule = TimeInterval(output_time_interval),
                         filename = output_name,
                         overwrite_existing = true)

    @info "Running a simulation of an unstable Bickley jet with $(Nh)² degrees of freedom..."

    run!(simulation)

    return output_name
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

        ζ_plot = Plots.heatmap(xζ, yζ, clamp.(ζi, -1, 1)'; color = :balance, kwargs...)
        c_plot = Plots.heatmap(xc, yc, clamp.(ci, -1, 1)'; color = :thermal, kwargs...)

        ζ_title = @sprintf("ζ at t = %.1f", t)
        c_title = @sprintf("u at t = %.1f", t)

        Plots.plot(ζ_plot, c_plot, title = [ζ_title c_title], size = (4000, 2000))
    end

    mp4(anim, experiment_name * ".mp4", fps = 8)
end

advection_schemes = [VectorInvariant(vorticity_scheme = WENO()),
                     WENO(),
                     VectorInvariant()]

experiment_names = ["VectorInvariant_VorticityScheme_WENO", "WENO", "VectorInvariant"]

for Nx in [128] # for Nx in [64, 128, 256, 512, 1024]
    for (i, advection) in enumerate(advection_schemes)
        output_name = run_bickley_jet(arch=CPU(), momentum_advection=advection, Nh=Nx,
                                      experiment_name = experiment_names[i])
    end
end