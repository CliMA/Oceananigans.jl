using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Models.HydrostaticFreeSurfaceModels: FFTImplicitFreeSurfaceSolver, MGImplicitFreeSurfaceSolver, finalize_solver!
using Printf
using Oceananigans.Solvers: initialize_AMGX, finalize_AMGX
using TimerOutputs

"""
Benchmarks the bumpy baroclinic adjustment problem with various implicit free-surface solvers.
"""

const to = TimerOutput()

arch = GPU()
initialize_AMGX(arch)

for N in 10:10:250
    @info "N=$N"
    println("")
    
    underlying_grid = RectilinearGrid(arch,
                                    topology = (Periodic, Bounded, Bounded), 
                                    size = (N, N, 24),
                                    x = (-500kilometers, 500kilometers),
                                    y = (-500kilometers, 500kilometers),
                                    z = (-1kilometers, 0),
                                    halo = (4, 4, 4))

    Lz_u = underlying_grid.Lz
    width = 50kilometers
    bump(x, y) = - Lz_u * (1 - 2 * exp(-(x^2 + y^2) / 2width^2))
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bump))

    # Physics
    Δx, Δz = grid.Lx / grid.Nx, grid.Lz / grid.Nz
    𝒜 = Δz/Δx # Grid cell aspect ratio.

    κh = 0.1    # [m² s⁻¹] horizontal diffusivity
    νh = 0.1    # [m² s⁻¹] horizontal viscosity
    κz = 𝒜 * κh # [m² s⁻¹] vertical diffusivity
    νz = 𝒜 * νh # [m² s⁻¹] vertical viscosity

    horizontal_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)

    diffusive_closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization();
                                                ν = νz, κ = κz)

    implicit_free_surface_solvers = (#:FastFourierTransform,
                                     #:PreconditionedConjugateGradient,
                                     :HeptadiagonalIterativeSolver,
                                     :Multigrid,
                                    #  :HeptadiagonalIterativeSolver_withMGpreconditioner,
                                     #:PreconditionedConjugateGradient_withFFTpreconditioner,
                                     #:PreconditionedConjugateGradient_withMGpreconditioner,
                                    )

    for implicit_free_surface_solver in implicit_free_surface_solvers

        if implicit_free_surface_solver == :PreconditionedConjugateGradient_withFFTpreconditioner
            fft_preconditioner = FFTImplicitFreeSurfaceSolver(grid)
            free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient, preconditioner=fft_preconditioner, reltol=sqrt(eps(eltype(grid))), abstol=0)
        elseif implicit_free_surface_solver == :PreconditionedConjugateGradient_withMGpreconditioner
            maxiter = 2
            mg_preconditioner = MGImplicitFreeSurfaceSolver(grid, Dict(:maxiter => maxiter))
            free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient, preconditioner=mg_preconditioner, reltol=sqrt(eps(eltype(grid))), abstol=0)
        elseif implicit_free_surface_solver == :HeptadiagonalIterativeSolver
            free_surface = ImplicitFreeSurface(solver_method=implicit_free_surface_solver, tolerance=sqrt(eps(eltype(grid))))
        elseif implicit_free_surface_solver == :HeptadiagonalIterativeSolver_withMGpreconditioner
            free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver, tolerance=sqrt(eps(eltype(grid))), preconditioner_method=:Multigrid)
        else
            free_surface = ImplicitFreeSurface(solver_method=implicit_free_surface_solver, reltol=sqrt(eps(eltype(grid))), abstol=0)
        end

        model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                            coriolis = BetaPlane(latitude = -45),
                                            buoyancy = BuoyancyTracer(),
                                            closure = (horizontal_closure, ),
                                            tracers = :b,
                                            momentum_advection = WENO(),
                                            tracer_advection = WENO())

        # Initial condition: a baroclinically unstable situation!
        ramp(y, δy) = min(max(0, y/δy + 1/2), 1)

        # Parameters
        N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
        M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

        δy = 50kilometers
        δb = δy * M²
        ϵb = 1e-2 * δb # noise amplitude

        bᵢ(x, y, z) = N² * z + δb * ramp(y, δy) + ϵb * randn()

        set!(model, b=bᵢ)

        Δt = 10minutes
        simulation = Simulation(model; Δt, stop_time=200days)

        #= only uncomment the print_progress callback for debugging
        wall_clock = Ref(time_ns())
        function print_progress(sim)
            elapsed = 1e-9 * (time_ns() - wall_clock[])
            msg = @sprintf("Iter: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹, next Δt: %s\n",
                        iteration(sim), prettytime(sim), prettytime(elapsed),
                        maximum(abs, sim.model.velocities.w), prettytime(sim.Δt))
            wall_clock[] = time_ns()
            try
                solver_iterations = sim.model.free_surface.implicit_step_solver.preconditioned_conjugate_gradient_solver.iteration
                msg *= @sprintf("solver iterations: %d", solver_iterations)
            catch
            end
            @info msg
            return nothing
        end
        simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(10))
        =#

        simulation.stop_iteration = 200
        
        run!(simulation)

        simulation.stop_iteration = 1200

        @info "Benchmark with $implicit_free_surface_solver free surface implicit solver:"
        @timeit to "$implicit_free_surface_solver and N=$N" run!(simulation)

        finalize_solver!(model.free_surface.implicit_step_solver)
    end
    # show(to)
end

finalize_AMGX(arch)
