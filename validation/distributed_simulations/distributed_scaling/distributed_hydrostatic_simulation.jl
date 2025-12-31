# Distributed hydrostatic simulation for scaling tests
#
# Run with:
#
#   mpiexec -n 4 julia --project distributed_scaling/distributed_hydrostatic_simulation.jl
#
# Environment variables:
#   NX, NY, NZ: grid size (default: 72, 30, 10)
#

using MPI
MPI.Init()

using JLD2
using Statistics: mean
using Printf
using Oceananigans
using Oceananigans.Utils: prettytime
using Oceananigans.DistributedComputations
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity
using Oceananigans.Units
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

function double_drake_bathymetry(λ, φ)
    if φ > -35
        (λ >  0 && λ < 1)  && return 0.0
        (λ > 90 && λ < 91) && return 0.0
    end
    return -10000.0
end

function run_hydrostatic_simulation!(grid_size;
                                     output_name = nothing,
                                     timestepper = :QuasiAdamsBashforth2,
                                     CFL = 0.35,
                                     barotropic_CFL = 0.75)

    arch = Distributed(CPU())
    grid = LatitudeLongitudeGrid(arch; size = grid_size,
                                 longitude = (-180, 180),
                                 latitude = (-75, 75),
                                 z = (-5500, 0),
                                 halo = (7, 7, 7))

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(double_drake_bathymetry))

    momentum_advection = WENOVectorInvariant()
    tracer_advection   = WENO(order = 7)

    buoyancy = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState())
    coriolis = HydrostaticSphericalCoriolis()
    closure  = CATKEVerticalDiffusivity()

    max_Δt = 45 * 48 / grid.Δλᶠᵃᵃ

    free_surface = SplitExplicitFreeSurface(grid; cfl = barotropic_CFL, fixed_Δt = max_Δt)

    model = HydrostaticFreeSurfaceModel(; grid,
                                        momentum_advection,
                                        tracer_advection,
                                        coriolis,
                                        closure,
                                        free_surface,
                                        tracers = (:T, :S),
                                        buoyancy,
                                        timestepper)

    wtime = Ref(time_ns())

    function progress(sim)
        @info @sprintf("iteration: %d, Δt: %.2e, wall time: %s, (|u|, |v|, |w|): %.2e %.2e %.2e, T: %.2e",
                       sim.model.clock.iteration, sim.Δt, prettytime((time_ns() - wtime[]) * 1e-9),
                       maximum(abs, sim.model.velocities.u), maximum(abs, sim.model.velocities.v),
                       maximum(abs, sim.model.velocities.w), maximum(abs, sim.model.tracers.T))
        wtime[] = time_ns()
    end

    simulation = Simulation(model; Δt = max_Δt, stop_time = 20days, stop_iteration = 100)

    # Adaptive time-stepping
    wizard = TimeStepWizard(cfl = CFL; max_change = 1.1, min_Δt = 10, max_Δt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    if !isnothing(output_name)
        simulation.output_writers[:fields] = JLD2Writer(model, merge(model.velocities, model.tracers),
                                                        filename = output_name * "_$(rank)",
                                                        schedule = TimeInterval(1day),
                                                        overwrite_existing = true)
    end

    run!(simulation)

    @info "Simulation completed on rank $rank"

    return nothing
end

# Reduced resolution for testing
Nx = parse(Int, get(ENV, "NX", "72"))
Ny = parse(Int, get(ENV, "NY", "30"))
Nz = parse(Int, get(ENV, "NZ", "10"))

grid_size = (Nx, Ny, Nz)

@info "Running HydrostaticFreeSurface model with grid size $grid_size"
run_hydrostatic_simulation!(grid_size)
