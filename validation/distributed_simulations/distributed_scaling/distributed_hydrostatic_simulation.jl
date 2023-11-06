using MPI
MPI.Init()

using JLD2
using Statistics: mean
using Printf
using Oceananigans
using Oceananigans.Utils: prettytime
using Oceananigans.DistributedComputations
using Oceananigans.Grids: node
using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity
using Oceananigans.Units
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

function double_drake_bathymetry(λ, φ) 
    if φ > -35
        (λ >  0 && λ < 1)  && return 0.0
        (λ > 90 && λ < 91) && return 0.0
    end

    return -10000.0
end

function run_hydrostatic_simulation!(grid_size, ranks, FT::DataType = Float64; 
                                     output_name = nothing, 
                                     timestepper = :QuasiAdamsBashforth2,
                                     CFL = 0.35,
                                     barotropic_CFL = 0.75)
        
    arch  = Distributed(GPU(), FT; partition = Partition(ranks...))
    grid  = LatitudeLongitudeGrid(arch; size = grid_size, longitude = (-180, 180),
			    		          latitude = (-75, 75),
					              z = (-5500, 0),
                                  halo = (7, 7, 7))

    grid  = ImmersedBoundaryGrid(grid, GridFittedBottom(double_drake_bathymetry)) 

    momentum_advection = WENOVectorInvariant(FT)
    tracer_advection   = WENO(grid, order = 7)
    
    buoyancy = SeawaterBuoyancy(FT; equation_of_state = TEOS10EquationOfState(FT))
    coriolis = HydrostaticSphericalCoriolis(FT)
    closure  = CATKEVerticalDiffusivity(FT) 

    max_Δt = 45 * 48 / grid.Δλᶠᵃᵃ

    free_surface = SplitExplicitFreeSurface(FT; grid, cfl = barotropic_CFL, fixed_Δt = max_Δt)

    model = HydrostaticFreeSurfaceModel(; grid, 
                                          momentum_advection,
                                          tracer_advection,
                                          coriolis, 
                                          closure,
                                          free_surface,
                                          tracers = (:T, :S, :e),
                                          buoyancy, 
                                          timestepper)

    wtime = Ref(time_ns())
    
    function progress(sim) 
        @info @sprintf("iteration: %d, Δt: %2e, wall time: %s (|u|, |v|, |w|): %.2e %.2e %.2e, b: %.2e \n", 
              sim.model.clock.iteration, sim.Δt, prettytime((time_ns() - wtime[])*1e-9),
              maximum(abs, sim.model.velocities.u), maximum(abs, sim.model.velocities.v), 
              maximum(abs, sim.model.velocities.w), maximum(abs, sim.model.tracers.b))
       wtime[] = time_ns()
    end

    simulation = Simulation(model; Δt=max_Δt, stop_time = 20days, stop_iteration = 100)
                        
    # Adaptive time-stepping
    wizard = TimeStepWizard(cfl=CFL; max_change=1.1, min_Δt=10, max_Δt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))
   
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    if !isnothing(output_name)
        simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                            filename = output_name * "_$(rank)",
                                                            schedule = TimeInterval(1day),
                                                            overwrite_existing = true)
    end
    
    run!(simulation)

    return nothing
end

rx = parse(Int, get(ENV, "RX", "1"))
ry = parse(Int, get(ENV, "RY", "1"))

ranks = (rx, ry, 1)

Nx = parse(Int, get(ENV, "NX", "1440"))
Ny = parse(Int, get(ENV, "NY", "600"))
Nz = parse(Int, get(ENV, "NZ", "100"))

grid_size = (Nx, Ny, Nz)

@info "Running Nonhydrostatic model with ranks $ranks"
run_hydrostatic_simulation!(grid_size, ranks)