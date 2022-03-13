ENV["GKSwstype"] = "nul"

using Printf
using Statistics

using Oceananigans
using Oceananigans.Operators: ζ₃ᶠᶠᶜ
using Oceananigans.Advection
using Oceananigans.AbstractOperations
using Oceananigans.OutputWriters
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Utils: prettytime

using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, ExplicitFreeSurface
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant
using Oceananigans.OutputReaders: FieldTimeSeries

using Oceananigans.Advection: ZWENO, WENOVectorInvariant
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
                           momentum_advection = VectorInvariant())

    Nφ = Int(Nh/2)

    grid = LatitudeLongitudeGrid(arch, size=(Nh, Nφ, 1),
                                radius = 1,
                                longitude = (-180, 180), latitude=(-45, 45), z=(0, 1), halo = (4, 4, 4),
                                precompute_metrics = true)
    
    free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver, gravitational_acceleration=1)

    model = HydrostaticFreeSurfaceModel(momentum_advection = momentum_advection,
                                          tracer_advection = WENO5(),
                                                      grid = grid,
                                                   tracers = :c,
                                                   closure = nothing,
                                              free_surface = free_surface,
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

    dr(x) = deg2rad(x)

    # Total initial conditions
    uᵢ(x, y, z) = U(dr(y)*8) + ϵ * ũ(dr(x)*2, dr(y)*8, ℓ, k)
    vᵢ(x, y, z) = ϵ * ṽ(dr(x)*2, dr(y)*4, ℓ, k)
    cᵢ(x, y, z) = C(dr(y)*8, grid.Ly)

    set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

    wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=10.0)

    c = sqrt(model.free_surface.gravitational_acceleration)
    Δt = 0.1 * model.grid.Δxᶜᶠᵃ[1] / c

    simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

    progress(sim) = @printf("Iter: %d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f \n",
                            iteration(sim), prettytime(sim), prettytime(sim.Δt),
                            maximum(abs, model.velocities.u), maximum(abs, model.free_surface.η))

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=10.0)

    simulation.callbacks[:wizard]   = Callback(wizard, IterationInterval(10))

    # Output: primitive fields + computations
    u, v, w, c = merge(model.velocities, model.tracers)

    ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid; computed_dependencies=(u, v))

    ζ = Field(ζ_op)
    outputs = merge(model.velocities, model.tracers, (ζ=ζ, η=model.free_surface.η))

    name = typeof(model.advection.momentum).name.wrapper
    if model.advection.momentum isa ZWENO
        name = "ZWENO"
    elseif model.advection.momentum isa WENOVectorInvariant
        name = "WENOVectorInvariant"
    end

    @show experiment_name = "spherical_bickley_jet_Nh_$(Nh)_$(name)"

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, outputs,
                                schedule =TimeInterval(output_time_interval),
                                prefix = experiment_name,
                                field_slicer = nothing,
                                force = true)

    @info "Running a simulation of an unstable Bickley jet with $(Nh)² degrees of freedom..."

    run!(simulation)

    return experiment_name 
end

# experiment_name = run_bickley_jet(momentum_advection=VectorInvariant(), Nh=128)
# experiment_name = run_bickley_jet(momentum_advection=WENO5(zweno=true, vector_invariant=true), Nh=128)

for Nx in [256]
    for momentum_advection in [WENO5(zweno=true, vector_invariant=true)]
        experiment_name = run_bickley_jet(momentum_advection=momentum_advection, Nh=Nx)
        # visualize_bickley_jet(experiment_name)
    end
end

