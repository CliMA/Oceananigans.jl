using CUDA: has_cuda
using OrderedCollections: OrderedDict

using Oceananigans: AbstractModel, AbstractOutputWriter, AbstractDiagnostic

using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.Distributed: MultiArch
using Oceananigans.AdvectionDivergence: CenteredSecondOrder
using Oceananigans.BuoyancyModels: validate_buoyancy, regularize_buoyancy, SeawaterBuoyancy
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Fields: BackgroundFields, Field, tracernames, VelocityFields, TracerFields, PressureFields
using Oceananigans.Forcings: model_forcing
using Oceananigans.Grids: inflate_halo_size, with_halo, architecture
using Oceananigans.Solvers: FFTBasedPoissonSolver
using Oceananigans.TimeSteppers: Clock, TimeStepper, update_state!
using Oceananigans.TurbulenceClosures: validate_closure, with_tracers, DiffusivityFields, time_discretization, implicit_diffusion_solver
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: FlavorOfCATKE
using Oceananigans.LagrangianParticleTracking: LagrangianParticles
using Oceananigans.Utils: tupleit
using Oceananigans.Grids: topology

import Oceananigans.Architectures: architecture

const ParticlesOrNothing = Union{Nothing, LagrangianParticles}

mutable struct NonhydrostaticModel{TS, E, A<:AbstractArchitecture, G, T, B, R, SD, U, C, Φ, F,
                                   V, S, K, BG, P, I, AF} <: AbstractModel{TS}

         architecture :: A        # Computer `Architecture` on which `Model` is run
                 grid :: G        # Grid of physical points on which `Model` is solved
                clock :: Clock{T} # Tracks iteration number and simulation time of `Model`
            advection :: V        # Advection scheme for velocities _and_ tracers
             buoyancy :: B        # Set of parameters for buoyancy model
             coriolis :: R        # Set of parameters for the background rotation rate of `Model`
         stokes_drift :: SD       # Set of parameters for surfaces waves via the Craik-Leibovich approximation
              forcing :: F        # Container for forcing functions defined by the user
              closure :: E        # Diffusive 'turbulence closure' for all model fields
    background_fields :: BG       # Background velocity and tracer fields
            particles :: P        # Particle set for Lagrangian tracking
           velocities :: U        # Container for velocity fields `u`, `v`, and `w`
              tracers :: C        # Container for tracer fields
            pressures :: Φ        # Container for hydrostatic and nonhydrostatic pressure
   diffusivity_fields :: K        # Container for turbulent diffusivities
          timestepper :: TS       # Object containing timestepper fields and parameters
      pressure_solver :: S        # Pressure/Poisson solver
    immersed_boundary :: I        # Models the physics of immersed boundaries within the grid
     auxiliary_fields :: AF       # User-specified auxiliary fields for forcing functions and boundary conditions
end

"""
    NonhydrostaticModel(;     grid,
                              clock = Clock{eltype(grid)}(0, 0, 1),
                          advection = CenteredSecondOrder(),
                           buoyancy = nothing,
                           coriolis = nothing,
                       stokes_drift = nothing,
                forcing::NamedTuple = NamedTuple(),
                            closure = nothing,
    boundary_conditions::NamedTuple = NamedTuple(),
                            tracers = (),
                        timestepper = :QuasiAdamsBashforth2,
      background_fields::NamedTuple = NamedTuple(),
      particles::ParticlesOrNothing = nothing,
                         velocities = nothing,
                          pressures = nothing,
                 diffusivity_fields = nothing,
                    pressure_solver = nothing,
                  immersed_boundary = nothing,
                   auxiliary_fields = NamedTuple(),
    )

Construct a model for a non-hydrostatic, incompressible fluid, using the Boussinesq approximation
when `buoyancy != nothing`. By default, all Bounded directions are rigid and impenetrable.

Keyword arguments
=================

  - `grid`: (required) The resolution and discrete geometry on which `model` is solved. The
            architecture (CPU/GPU) that the model is solve is inferred from the architecture
            of the grid.
  - `advection`: The scheme that advects velocities and tracers. See `Oceananigans.Advection`.
  - `buoyancy`: The buoyancy model. See `Oceananigans.BuoyancyModels`.
  - `coriolis`: Parameters for the background rotation rate of the model.
  - `stokes_drift`: Parameters for Stokes drift fields associated with surface waves. Default: `nothing`.
  - `forcing`: `NamedTuple` of user-defined forcing functions that contribute to solution tendencies.
  - `closure`: The turbulence closure for `model`. See `Oceananigans.TurbulenceClosures`.
  - `boundary_conditions`: `NamedTuple` containing field boundary conditions.
  - `tracers`: A tuple of symbols defining the names of the modeled tracers, or a `NamedTuple` of
               preallocated `CenterField`s.
  - `timestepper`: A symbol that specifies the time-stepping method. Either `:QuasiAdamsBashforth2` or
                   `:RungeKutta3`.
  - `background_fields`: `NamedTuple` with background fields (e.g., background flow). Default: `nothing`.
  - `particles`: Lagrangian particles to be advected with the flow. Default: `nothing`.
  - `velocities`: The model velocities. Default: `nothing`.
  - `pressures`: Hydrostatic and non-hydrostatic pressure fields. Default: `nothing`.
  - `diffusivity_fields`: Diffusivity fields. Default: `nothing`.
  - `pressure_solver`: Pressure solver to be used in the model. If `nothing` (default), the model constructor
    chooses the default based on the `grid` provide.
  - `immersed_boundary`: The immersed boundary. Default: `nothing`.
  - `auxiliary_fields`: `NamedTuple` of auxiliary fields. Default: `nothing`.               
"""
function NonhydrostaticModel(;    grid,
                                 clock = Clock{eltype(grid)}(0, 0, 1),
                             advection = CenteredSecondOrder(),
                              buoyancy = nothing,
                              coriolis = nothing,
                          stokes_drift = nothing,
                   forcing::NamedTuple = NamedTuple(),
                               closure = nothing,
       boundary_conditions::NamedTuple = NamedTuple(),
                               tracers = (),
                           timestepper = :QuasiAdamsBashforth2,
         background_fields::NamedTuple = NamedTuple(),
         particles::ParticlesOrNothing = nothing,
                            velocities = nothing,
                             pressures = nothing,
                    diffusivity_fields = nothing,
                       pressure_solver = nothing,
                     immersed_boundary = nothing,
                      auxiliary_fields = NamedTuple(),
    )

    arch = architecture(grid)

    if arch == GPU() && !has_cuda()
         throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))
    end

    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    # We don't support CAKTE for NonhydrostaticModel yet.
    closure = validate_closure(closure)
    first_closure = closure isa Tuple ? first(closure) : closure
    first_closure isa FlavorOfCATKE &&
        error("CATKEVerticalDiffusivity is not supported for " *
              "NonhydrostaticModel --- yet!")

    validate_buoyancy(buoyancy, tracernames(tracers))
    buoyancy = regularize_buoyancy(buoyancy)

    # Adjust halos when the advection scheme or turbulence closure requires it.
    # Note that halos are isotropic by default; however we respect user-input here
    # by adjusting each (x, y, z) halo individually.
    user_halo = grid.Hx, grid.Hy, grid.Hz
    required_halo = Hx, Hy, Hz = inflate_halo_size(user_halo..., topology(grid), advection, closure)
    if any(user_halo .< required_halo) # Replace grid
        @warn "Inflating model grid halo size to ($Hx, $Hy, $Hz) and recreating grid. " *
              "The model grid will be different from the input grid. To avoid this warning, " *
              "pass halo=($Hx, $Hy, $Hz) when constructing the grid."

        grid = with_halo((Hx, Hy, Hz), grid)
    end

    # Collect boundary conditions for all model prognostic fields and, if specified, some model
    # auxiliary fields. Boundary conditions are "regularized" based on the _name_ of the field:
    # boundary conditions on u, v, w are regularized assuming they represent momentum at appropriate
    # staggered locations. All other fields are regularized assuming they are tracers.
    # Note that we do not regularize boundary conditions contained in *tupled* diffusivity fields right now.
    #
    # First, we extract boundary conditions that are embedded within any _user-specified_ field tuples:
    embedded_boundary_conditions = merge(extract_boundary_conditions(velocities),
                                         extract_boundary_conditions(tracers),
                                         extract_boundary_conditions(pressures),
                                         extract_boundary_conditions(diffusivity_fields))

    # Next, we form a list of default boundary conditions:
    prognostic_field_names = (:u, :v, :w, tracernames(tracers)...)
    default_boundary_conditions = NamedTuple{prognostic_field_names}(FieldBoundaryConditions() for name in prognostic_field_names)

    # Finally, we merge specified, embedded, and default boundary conditions. Specified boundary conditions
    # have precedence, followed by embedded, followed by default.
    boundary_conditions = merge(default_boundary_conditions, embedded_boundary_conditions, boundary_conditions)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, prognostic_field_names)

    # Ensure `closure` describes all tracers
    closure = with_tracers(tracernames(tracers), closure)

    # Either check grid-correctness, or construct tuples of fields
    velocities         = VelocityFields(velocities, grid, boundary_conditions)
    tracers            = TracerFields(tracers,      grid, boundary_conditions)
    pressures          = PressureFields(pressures,  grid, boundary_conditions)
    diffusivity_fields = DiffusivityFields(diffusivity_fields, grid, tracernames(tracers), boundary_conditions, closure)

    if isnothing(pressure_solver)
        pressure_solver = PressureSolver(arch, grid)
    end

    # Materialize background fields
    background_fields = BackgroundFields(background_fields, tracernames(tracers), grid, clock)

    # Instantiate timestepper if not already instantiated
    implicit_solver = implicit_diffusion_solver(time_discretization(closure), grid)
    timestepper = TimeStepper(timestepper, grid, tracernames(tracers), implicit_solver=implicit_solver)

    # Regularize forcing for model tracer and velocity fields.
    model_fields = merge(velocities, tracers)
    forcing = model_forcing(model_fields; forcing...)

    model = NonhydrostaticModel(arch, grid, clock, advection, buoyancy, coriolis, stokes_drift,
                                forcing, closure, background_fields, particles, velocities, tracers,
                                pressures, diffusivity_fields, timestepper, pressure_solver, immersed_boundary,
                                auxiliary_fields)

    update_state!(model)
    
    return model
end

architecture(model::NonhydrostaticModel) = model.architecture

#####
##### Recursive util for building NamedTuples of boundary conditions from NamedTuples of fields
#####
##### Note: ignores tuples, including tuples of Symbols (tracer names) and
##### tuples of DiffusivityFields (which occur for tupled closures)
#####

extract_boundary_conditions(::Nothing) = NamedTuple()
extract_boundary_conditions(::Tuple) = NamedTuple()

function extract_boundary_conditions(field_tuple::NamedTuple)
    names = propertynames(field_tuple)
    bcs = Tuple(extract_boundary_conditions(field) for field in field_tuple)
    return NamedTuple{names}(bcs)
end

extract_boundary_conditions(field::Field) = field.boundary_conditions
