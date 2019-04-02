mutable struct Model{A<:Architecture}
    configuration::ModelConfiguration
    boundary_conditions::ModelBoundaryConditions
    constants::PlanetaryConstants
    eos::EquationOfState
    grid::Grid
    velocities::VelocityFields
    tracers::TracerFields
    pressures::PressureFields
    G::SourceTerms
    Gp::SourceTerms
    forcing  # ::Forcing  # No type so we can set it to nothing while checkpointing.
    stepper_tmp::StepperTemporaryFields
    poisson_solver  # ::PoissonSolver or ::PoissonSolverGPU
    clock::Clock
    output_writers::Array{OutputWriter,1}
    diagnostics::Array{Diagnostic,1}
end

"""
    Model(; kwargs...)

Construct an `Oceananigans.jl` model. The keyword arguments are:

          N (tuple) : model resolution in (x, y, z)
          L (tuple) : model domain in (x, y, z)
         dt (float) : time step
 start_time (float) : start time for the simulation
  iteration (float) : ?
         arch (sym) : architecture (:cpu or :gpu)
  float_type (type) : floating point type for model data (typically Float32 or Float64)
          constants : planetary constants (?)
                eos : equation of state to infer density from temperature and salinity
            forcing : forcing functions for (u, v, w, T, S)
boundary_conditions : boundary conditions
     output_writers : output writer
        diagonstics : diagnostics

        ... and more.
"""
function Model(;
    # Model resolution and domain size
             N,
             L,
    # Molecular parameters
             ν = 1.05e-6, νh = ν, νv = ν,
             κ = 1.43e-7, κh = κ, κv = κ,
    # Time stepping
    start_time = 0,
     iteration = 0,
    # Model architecture and floating point precision
          arch = CPU(),
    float_type = Float64,
     constants = Earth(),
    # Equation of State
           eos = LinearEquationOfState(),
    # Forcing and boundary conditions for (u, v, w, T, S)
       forcing = Forcing(nothing, nothing, nothing, nothing, nothing),
    boundary_conditions = ModelBoundaryConditions(),
    # Output and diagonstics
    output_writers = OutputWriter[],
       diagnostics = Diagnostic[]
)

    arch == GPU() && !HAVE_CUDA && throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))

    # Initialize model basics.
    configuration = ModelConfiguration(νh, νv, κh, κv)
             grid = RegularCartesianGrid(float_type, N, L)
            clock = Clock{float_type}(start_time, iteration)

    # Initialize fields, including source terms and temporary variables.
      velocities = VelocityFields(arch, grid)
         tracers = TracerFields(arch, grid)
       pressures = PressureFields(arch, grid)
               G = SourceTerms(arch, grid)
              Gp = SourceTerms(arch, grid)
     stepper_tmp = StepperTemporaryFields(arch, grid)

     poisson_solver = init_poisson_solver(arch, grid, stepper_tmp.fCC1)

    # Default initial condition
    velocities.u.data .= 0
    velocities.v.data .= 0
    velocities.w.data .= 0
    tracers.S.data .= eos.S₀
    tracers.T.data .= eos.T₀

    Model{typeof(arch)}(configuration, boundary_conditions, constants, eos, grid,
                        velocities, tracers, pressures, G, Gp, forcing,
                        stepper_tmp, poisson_solver, clock, output_writers, diagnostics)
end

"Legacy constructor for `Model`."
Model(N, L; arch=:CPU, float_type=Float64) = Model(N=N, L=L; arch=arch, float_type=float_type)

arch(model::Model{A}) where A <: Architecture = A
float_type(m::Model) = eltype(model.grid)

add_bcs!(model::Model; kwargs...) = add_bcs(model.boundary_conditions; kwargs...)
