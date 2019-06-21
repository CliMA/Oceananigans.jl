using .TurbulenceClosures

mutable struct Model{A<:Architecture, G, TC, T}
              arch :: A                  # Computer `Architecture` on which `Model` is run.
              grid :: G                  # Grid of physical points on which `Model` is solved.
             clock :: Clock{T}           # Tracks iteration number and simulation time of `Model`.
               eos :: EquationOfState    # Defines relationship between temperature,  salinity, and 
                                         # buoyancy in the Boussinesq vertical momentum equation.
         constants :: PlanetaryConstants # Set of physical constants, inc. gravitational acceleration.
        velocities :: VelocityFields     # Container for velocity fields `u`, `v`, and `w`.
           tracers :: TracerFields       # Container for tracer fields.
         pressures :: PressureFields     # Container for hydrostatic and nonhydrostatic pressure.
           forcing                       # Container for forcing functions defined by the user
           closure :: TC                 # Diffusive 'turbulence closure' for all model fields
    boundary_conditions :: ModelBoundaryConditions # Container for 3d bcs on all fields.
                 G :: SourceTerms        # Container for right-hand-side of PDE that governs `Model`
                Gp :: SourceTerms        # RHS at previous time-step (for Adams-Bashforth time integration)
    poisson_solver                       # ::PoissonSolver or ::PoissonSolverGPU
       stepper_tmp :: StepperTemporaryFields # Temporary fields used for the Poisson solver.
    output_writers :: Array{OutputWriter, 1} # Objects that write data to disk.
       diagnostics :: Array{Diagnostic, 1}   # Objects that calc diagnostics on-line during simulation.
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
             ν = 1.05e-6, νh=ν, νv=ν, 
             κ = 1.43e-7, κh=κ, κv=κ, 
       closure = ConstantAnisotropicDiffusivity(float_type, νh=νh, νv=νv, κh=κh, κv=κv),
    # Time stepping
    start_time = 0,
     iteration = 0,
         clock = Clock{float_type}(start_time, iteration),
    # Fluid and physical parameters
     constants = Earth(float_type),
           eos = LinearEquationOfState(float_type),
    # Forcing and boundary conditions for (u, v, w, T, S)
       forcing = Forcing(nothing, nothing, nothing, nothing, nothing),
    boundary_conditions = ModelBoundaryConditions(),
    # Output and diagonstics
    output_writers = OutputWriter[],
       diagnostics = Diagnostic[]
)

    arch == GPU() && !HAVE_CUDA && throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))

    # Initialize fields, including source terms and temporary variables.
      velocities = VelocityFields(arch, grid)
         tracers = TracerFields(arch, grid)
       pressures = PressureFields(arch, grid)
               G = SourceTerms(arch, grid)
              Gp = SourceTerms(arch, grid)
     stepper_tmp = StepperTemporaryFields(arch, grid)

    # Initialize Poisson solver.
    poisson_solver = PoissonSolver(arch, grid)

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
    underlying_data(tracers.S) .= eos.S₀
    underlying_data(tracers.T) .= eos.T₀

    # Set all further fields to 0
    for set in sets
        for fldname in propertynames(set)
            fld = getproperty(set, fldname)
            underlying_data(fld) .= 0 # promotes to eltype of fld.data
        end
    end
end

"""
    add_u_forcing!(model::Model, Fu::Function)

Appends Fu to the model's u forcing function.
"""
function add_u_forcing!(model::Model, Fu::Function)
    old_Fu, old_Fv, old_Fw = model.forcing.u, model.forcing.v, model.forcing.w
    old_FT, old_FS         = model.forcing.T, model.forcing.S

    @inline new_Fu(args...) = old_Fu(args...) + Fu(args...)

    model.forcing = Forcing(Fu=new_Fu, Fv=old_Fv, Fw=old_Fw, FT=old_FT, FS=old_FS)
end

"""
    add_v_forcing!(model::Model, Fv::Function)

Appends Fv to the model's v forcing function.
"""
function add_v_forcing!(model::Model, Fv::Function)
    old_Fu, old_Fv, old_Fw = model.forcing.u, model.forcing.v, model.forcing.w
    old_FT, old_FS         = model.forcing.T, model.forcing.S

    @inline new_Fv(args...) = old_Fv(args...) + Fv(args...)

    model.forcing = Forcing(Fu=old_Fu, Fv=new_Fv, Fw=old_Fw, FT=old_FT, FS=old_FS)
end

"""
    add_w_forcing!(model::Model, Fw::Function)

Appends Fw to the model's w forcing function.
"""
function add_w_forcing!(model::Model, Fw::Function)
    old_Fu, old_Fv, old_Fw = model.forcing.u, model.forcing.v, model.forcing.w
    old_FT, old_FS         = model.forcing.T, model.forcing.S

    @inline new_Fw(args...) = old_Fw(args...) + Fw(args...)

    model.forcing = Forcing(Fu=old_Fu, Fv=old_Fv, Fw=new_Fw, FT=old_FT, FS=old_FS)
end

"""
    add_T_forcing!(model::Model, FT::Function)

Appends FT to the model's T forcing function.
"""
function add_T_forcing!(model::Model, FT::Function)
    old_Fu, old_Fv, old_Fw = model.forcing.u, model.forcing.v, model.forcing.w
    old_FT, old_FS         = model.forcing.T, model.forcing.S

    @inline new_FT(args...) = old_FT(args...) + FT(args...)

    model.forcing = Forcing(Fu=old_Fu, Fv=old_Fv, Fw=old_Fw, FT=new_FT, FS=old_FS)
end

"""
    add_S_forcing!(model::Model, FS::Function)

Appends FS to the model's S forcing function.
"""
function add_S_forcing!(model::Model, FS::Function)
    old_Fu, old_Fv, old_Fw = model.forcing.u, model.forcing.v, model.forcing.w
    old_FT, old_FS         = model.forcing.T, model.forcing.S

    @inline new_FS(args...) = old_FS(args...) + FS(args...)

    model.forcing = Forcing(Fu=old_Fu, Fv=old_Fv, Fw=old_Fw, FT=old_FT, FS=new_FS)
end

"""
    add_sponge_layer!(model; damping_timescale)

Adds a sponge layer to the bottom layer of the `model`. The purpose of the
sponge layer is to effectively dampen out waves reaching the bottom of the
domain and avoid having waves being continuously generated and reflecting from
the bottom, some of which may grow unphysically large in amplitude.

Numerically the sponge layer acts as an extra source term in the momentum
equations. It takes on the form Gu[i, j, k] += -u[i, j, k]/τ for each momentum
source term where τ is a damping timescale. Typially, Δt << τ.
"""
function add_sponge_layer!(model; damping_timescale)
    τ = damping_timescale

    @inline wave_damping_u(grid, u, v, w, T, S, i, j, k) = ifelse(k == grid.Nz, -u[i, j, k] / τ, 0)
    @inline wave_damping_v(grid, u, v, w, T, S, i, j, k) = ifelse(k == grid.Nz, -v[i, j, k] / τ, 0)
    @inline wave_damping_w(grid, u, v, w, T, S, i, j, k) = ifelse(k == grid.Nz, -w[i, j, k] / τ, 0)

    add_u_forcing!(model, wave_damping_u)
    add_v_forcing!(model, wave_damping_v)
    add_w_forcing!(model, wave_damping_w)
end
