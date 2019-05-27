using .TurbulenceClosures

mutable struct Model{A<:Architecture, G, TC, T}
              arch :: A                  # Computer `Architecture` on which `Model` is run.
              grid :: G                  # Grid of physical points on which `Model` is solved.
             clock :: Clock{T}           # Tracks iteration number and simulation time of `Model`.
               eos :: EquationOfState    # Defines relationship between temperature,  salinity, and 
                                         # buoyancy in the Boussinesq vertical momentum equation.
         constants :: PlanetaryConstants # A set of physical constants, inc. gravitational acceleration.
        velocities :: VelocityFields     # A container for velocity fields `u`, `v`, and `w`.
           tracers :: TracerFields       # A container for tracer fields.
         pressures :: PressureFields     # A container for hydrostatic and nonhydrostatic pressure.
           forcing                       # A container for forcing functions defined by the user
           closure :: TC                 # Diffusive 'turbulence closure' for all model fields
    boundary_conditions :: ModelBoundaryConditions # A container for 3d boundary conditions on all model fields.
                 G :: SourceTerms        # A container for the right-hand-side of the PDE that governs `Model`
                Gp :: SourceTerms        # The rhs at a previous time-step used for Adams-Bashforth time integration
    poisson_solver                       # ::PoissonSolver or ::PoissonSolverGPU
       stepper_tmp :: StepperTemporaryFields # Temporary fields used for the Poisson solver.
    output_writers :: Array{OutputWriter, 1} # Array of objects that write data to disk.
       diagnostics :: Array{Diagnostic, 1}   # Array of objects that calc diagnostics on-line during a simulation.
end


"""
    Model(; kwargs...)

Construct an `Oceananigans.jl` model.
"""
function Model(;
    # Model resolution and domain size
             N,
             L,
    # Model architecture and floating point precision
          arch = CPU(),
    float_type = Float64,
          grid = RegularCartesianGrid(float_type, N, L),
    # Isotropic transport coefficients (exposed to `Model` constructor for convenience)
             ν = 1.05e-6,
             κ = 1.43e-7,
       closure = ConstantIsotropicDiffusivity(ν=ν, κ=κ),
    # Time stepping
    start_time = 0,
     iteration = 0,
         clock = Clock{float_type}(start_time, iteration),
    # Fluid and physical parameters
     constants = Earth(),
           eos = LinearEquationOfState(),
    # Forcing and boundary conditions for (u, v, w, T, S)
       forcing = Forcing(nothing, nothing, nothing, nothing, nothing),
    boundary_conditions = ModelBoundaryConditions(),
    # Output and diagonstics
    output_writers = OutputWriter[],
       diagnostics = Diagnostic[]
)

    arch == GPU() && !HAVE_CUDA && throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))

    # Model components not initialized in function signature:
         velocities = VelocityFields(arch, grid)
            tracers = TracerFields(arch, grid)
          pressures = PressureFields(arch, grid)
                  G = SourceTerms(arch, grid)
                 Gp = SourceTerms(arch, grid)
        stepper_tmp = StepperTemporaryFields(arch, grid)
     poisson_solver = init_poisson_solver(arch, grid, stepper_tmp.fCC1)

    # Set the default initial condition
    initialize_with_defaults!(eos, tracers, velocities, G, Gp)

    Model(arch, grid, clock, eos, constants,
          velocities, tracers, pressures, forcing, closure, boundary_conditions,
          G, Gp, poisson_solver, stepper_tmp, output_writers, diagnostics)
end

arch(model::Model{A}) where A <: Architecture = A
float_type(m::Model) = eltype(model.grid)
add_bcs!(model::Model; kwargs...) = add_bcs(model.boundary_conditions; kwargs...)

function initialize_with_defaults!(eos, tracers, sets...)

    # Default tracer initial condition is deteremined by eos.
    tracers.S.data    .= eos.S₀
    tracers.T.data    .= eos.T₀

    # Set all further fields to 0
    for set in sets
        for fldname in propertynames(set)
            fld = getproperty(set, fldname)
            fld.data .= 0 # promotes to eltype of fld.data
        end
    end
    
    return nothing
end
