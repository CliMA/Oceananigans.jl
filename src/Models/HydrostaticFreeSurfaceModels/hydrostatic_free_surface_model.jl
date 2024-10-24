using CUDA: has_cuda
using OrderedCollections: OrderedDict

using Oceananigans.DistributedComputations
using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.Advection: AbstractAdvectionScheme, CenteredSecondOrder, VectorInvariant, adapt_advection_order
using Oceananigans.BuoyancyModels: validate_buoyancy, regularize_buoyancy, SeawaterBuoyancy, g_Earth
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Biogeochemistry: validate_biogeochemistry, AbstractBiogeochemistry, biogeochemical_auxiliary_fields
using Oceananigans.Fields: Field, CenterField, tracernames, VelocityFields, TracerFields
using Oceananigans.Forcings: model_forcing
using Oceananigans.Grids: AbstractCurvilinearGrid, AbstractHorizontallyCurvilinearGrid, architecture, halo_size
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Models: AbstractModel, validate_model_halo, NaNChecker, validate_tracer_advection, extract_boundary_conditions
using Oceananigans.TimeSteppers: Clock, TimeStepper, update_state!, AbstractLagrangianParticles
using Oceananigans.TurbulenceClosures: validate_closure, with_tracers, DiffusivityFields, add_closure_specific_boundary_conditions
using Oceananigans.TurbulenceClosures: time_discretization, implicit_diffusion_solver
using Oceananigans.Utils: tupleit

import Oceananigans: initialize!
import Oceananigans.Models: total_velocities, default_nan_checker, timestepper

PressureField(grid) = (; pHY′ = CenterField(grid))

const ParticlesOrNothing = Union{Nothing, AbstractLagrangianParticles}
const AbstractBGCOrNothing = Union{Nothing, AbstractBiogeochemistry}

mutable struct HydrostaticFreeSurfaceModel{TS, E, A<:AbstractArchitecture, S,
                                           G, T, V, B, R, F, P, BGC, U, C, Φ, K, AF} <: AbstractModel{TS}

          architecture :: A        # Computer `Architecture` on which `Model` is run
                  grid :: G        # Grid of physical points on which `Model` is solved
                 clock :: Clock{T} # Tracks iteration number and simulation time of `Model`
             advection :: V        # Advection scheme for tracers
              buoyancy :: B        # Set of parameters for buoyancy model
              coriolis :: R        # Set of parameters for the background rotation rate of `Model`
          free_surface :: S        # Free surface parameters and fields
               forcing :: F        # Container for forcing functions defined by the user
               closure :: E        # Diffusive 'turbulence closure' for all model fields
             particles :: P        # Particle set for Lagrangian tracking
       biogeochemistry :: BGC      # Biogeochemistry for Oceananigans tracers
            velocities :: U        # Container for velocity fields `u`, `v`, and `w`
               tracers :: C        # Container for tracer fields
              pressure :: Φ        # Container for hydrostatic pressure
    diffusivity_fields :: K        # Container for turbulent diffusivities
           timestepper :: TS       # Object containing timestepper fields and parameters
      auxiliary_fields :: AF       # User-specified auxiliary fields for forcing functions and boundary conditions
end

default_free_surface(grid::XYRegularRG; gravitational_acceleration=g_Earth) =
    ImplicitFreeSurface(; gravitational_acceleration)

default_free_surface(grid; gravitational_acceleration=g_Earth) =
    SplitExplicitFreeSurface(grid; cfl = 0.7, gravitational_acceleration)

"""
    HydrostaticFreeSurfaceModel(; grid,
                                clock = Clock{eltype(grid)}(time = 0),
                                momentum_advection = VectorInvariant(),
                                tracer_advection = CenteredSecondOrder(),
                                buoyancy = SeawaterBuoyancy(eltype(grid)),
                                coriolis = nothing,
                                free_surface = default_free_surface(grid, gravitational_acceleration=g_Earth),
                                forcing::NamedTuple = NamedTuple(),
                                closure = nothing,
                                boundary_conditions::NamedTuple = NamedTuple(),
                                tracers = (:T, :S),
                                particles::ParticlesOrNothing = nothing,
                                biogeochemistry::AbstractBGCOrNothing = nothing,
                                velocities = nothing,
                                pressure = nothing,
                                diffusivity_fields = nothing,
                                auxiliary_fields = NamedTuple())

Construct a hydrostatic model with a free surface on `grid`.

Keyword arguments
=================

  - `grid`: (required) The resolution and discrete geometry on which `model` is solved. The
            architecture (CPU/GPU) that the model is solved is inferred from the architecture
            of the `grid`.
  - `momentum_advection`: The scheme that advects velocities. See `Oceananigans.Advection`.
  - `tracer_advection`: The scheme that advects tracers. See `Oceananigans.Advection`.
  - `buoyancy`: The buoyancy model. See `Oceananigans.BuoyancyModels`.
  - `coriolis`: Parameters for the background rotation rate of the model.
  - `free_surface`: The free surface model. The default free-surface solver depends on the
                    geometry of the `grid`. If the `grid` is a `RectilinearGrid` that is
                    regularly spaced in the horizontal the default is an `ImplicitFreeSurface`
                    solver with `solver_method = :FFTBasedPoissonSolver`. In all other cases,
                    the default is a `SplitExplicitFreeSurface`.
  - `tracers`: A tuple of symbols defining the names of the modeled tracers, or a `NamedTuple` of
               preallocated `CenterField`s.
  - `forcing`: `NamedTuple` of user-defined forcing functions that contribute to solution tendencies.
  - `closure`: The turbulence closure for `model`. See `Oceananigans.TurbulenceClosures`.
  - `boundary_conditions`: `NamedTuple` containing field boundary conditions.
  - `particles`: Lagrangian particles to be advected with the flow. Default: `nothing`.
  - `biogeochemistry`: Biogeochemical model for `tracers`.
  - `velocities`: The model velocities. Default: `nothing`.
  - `pressure`: Hydrostatic pressure field. Default: `nothing`.
  - `diffusivity_fields`: Diffusivity fields. Default: `nothing`.
  - `auxiliary_fields`: `NamedTuple` of auxiliary fields. Default: `nothing`.
"""
function HydrostaticFreeSurfaceModel(; grid,
                                     clock = Clock{eltype(grid)}(time = 0),
                                     momentum_advection = VectorInvariant(),
                                     tracer_advection = CenteredSecondOrder(),
                                     buoyancy = nothing,
                                     coriolis = nothing,
                                     free_surface = default_free_surface(grid, gravitational_acceleration=g_Earth),
                                     tracers = nothing,
                                     forcing::NamedTuple = NamedTuple(),
                                     closure = nothing,
                                     boundary_conditions::NamedTuple = NamedTuple(),
                                     particles::ParticlesOrNothing = nothing,
                                     biogeochemistry::AbstractBGCOrNothing = nothing,
                                     velocities = nothing,
                                     pressure = nothing,
                                     diffusivity_fields = nothing,
                                     auxiliary_fields = NamedTuple())

    # Check halos and throw an error if the grid's halo is too small
    @apply_regionally validate_model_halo(grid, momentum_advection, tracer_advection, closure)

    # Validate biogeochemistry (add biogeochemical tracers automagically)
    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)
    biogeochemical_fields = merge(auxiliary_fields, biogeochemical_auxiliary_fields(biogeochemistry))
    tracers, auxiliary_fields = validate_biogeochemistry(tracers, biogeochemical_fields, biogeochemistry, grid, clock)

    # Reduce the advection order in directions that do not have enough grid points
    @apply_regionally momentum_advection = validate_momentum_advection(momentum_advection, grid)
    default_tracer_advection, tracer_advection = validate_tracer_advection(tracer_advection, grid)
    default_generator(name, tracer_advection) = default_tracer_advection

    # Generate tracer advection scheme for each tracer
    tracer_advection_tuple = with_tracers(tracernames(tracers), tracer_advection, default_generator, with_velocities=false)
    momentum_advection_tuple = (; momentum = momentum_advection)
    advection = merge(momentum_advection_tuple, tracer_advection_tuple)
    advection = NamedTuple(name => adapt_advection_order(scheme, grid) for (name, scheme) in pairs(advection))

    validate_buoyancy(buoyancy, tracernames(tracers))
    buoyancy = regularize_buoyancy(buoyancy)

    # Collect boundary conditions for all model prognostic fields and, if specified, some model
    # auxiliary fields. Boundary conditions are "regularized" based on the _name_ of the field:
    # boundary conditions on u, v are regularized assuming they represent momentum at appropriate
    # staggered locations. All other fields are regularized assuming they are tracers.
    # Note that we do not regularize boundary conditions contained in *tupled* diffusivity fields right now.
    #
    # First, we extract boundary conditions that are embedded within any _user-specified_ field tuples:
    embedded_boundary_conditions = merge(extract_boundary_conditions(velocities),
                                         extract_boundary_conditions(tracers),
                                         extract_boundary_conditions(pressure),
                                         extract_boundary_conditions(diffusivity_fields))

    # Next, we form a list of default boundary conditions:
    prognostic_field_names = (:u, :v, :w, tracernames(tracers)..., :η, keys(auxiliary_fields)...)
    default_boundary_conditions = NamedTuple{prognostic_field_names}(Tuple(FieldBoundaryConditions()
                                                                           for name in prognostic_field_names))

    # Then we merge specified, embedded, and default boundary conditions. Specified boundary conditions
    # have precedence, followed by embedded, followed by default.
    boundary_conditions = merge(default_boundary_conditions, embedded_boundary_conditions, boundary_conditions)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, prognostic_field_names)

    # Finally, we ensure that closure-specific boundary conditions, such as
    # those required by CATKEVerticalDiffusivity, are enforced:
    boundary_conditions = add_closure_specific_boundary_conditions(closure,
                                                                   boundary_conditions,
                                                                   grid,
                                                                   tracernames(tracers),
                                                                   buoyancy)

    # Ensure `closure` describes all tracers
    closure = with_tracers(tracernames(tracers), closure)

    # Put CATKE first in the list of closures
    closure = validate_closure(closure)

    # Either check grid-correctness, or construct tuples of fields
    velocities         = HydrostaticFreeSurfaceVelocityFields(velocities, grid, clock, boundary_conditions)
    tracers            = TracerFields(tracers, grid, boundary_conditions)
    pressure           = PressureField(grid)
    diffusivity_fields = DiffusivityFields(diffusivity_fields, grid, tracernames(tracers), boundary_conditions, closure)

    @apply_regionally validate_velocity_boundary_conditions(grid, velocities)

    arch = architecture(grid)
    free_surface = validate_free_surface(arch, free_surface)
    free_surface = materialize_free_surface(free_surface, velocities, grid)

    # Instantiate timestepper if not already instantiated
    implicit_solver = implicit_diffusion_solver(time_discretization(closure), grid)
    timestepper = TimeStepper(:QuasiAdamsBashforth2, grid, tracernames(tracers);
                              implicit_solver = implicit_solver,
                              Gⁿ = HydrostaticFreeSurfaceTendencyFields(velocities, free_surface, grid, tracernames(tracers)),
                              G⁻ = HydrostaticFreeSurfaceTendencyFields(velocities, free_surface, grid, tracernames(tracers)))

    # Regularize forcing for model tracer and velocity fields.
    model_fields = merge(hydrostatic_prognostic_fields(velocities, free_surface, tracers), auxiliary_fields)
    forcing = model_forcing(model_fields; forcing...)
    
    model = HydrostaticFreeSurfaceModel(arch, grid, clock, advection, buoyancy, coriolis,
                                        free_surface, forcing, closure, particles, biogeochemistry, velocities, tracers,
                                        pressure, diffusivity_fields, timestepper, auxiliary_fields)

    update_state!(model; compute_tendencies = false)

    return model
end

validate_velocity_boundary_conditions(grid, velocities) = validate_vertical_velocity_boundary_conditions(velocities.w)

function validate_vertical_velocity_boundary_conditions(w)
    w.boundary_conditions.top === nothing || error("Top boundary condition for HydrostaticFreeSurfaceModel velocities.w
                                                    must be `nothing`!")
    return nothing
end

validate_free_surface(::Distributed, free_surface::SplitExplicitFreeSurface) = free_surface
validate_free_surface(::Distributed, free_surface::ExplicitFreeSurface)      = free_surface
validate_free_surface(arch::Distributed, free_surface) = error("$(typeof(free_surface)) is not supported with $(typeof(arch))")
validate_free_surface(arch, free_surface) = free_surface

validate_momentum_advection(momentum_advection, ibg::ImmersedBoundaryGrid) = validate_momentum_advection(momentum_advection, ibg.underlying_grid)
validate_momentum_advection(momentum_advection, grid::RectilinearGrid)                     = momentum_advection
validate_momentum_advection(momentum_advection, grid::AbstractHorizontallyCurvilinearGrid) = momentum_advection
validate_momentum_advection(momentum_advection::Nothing,         grid::OrthogonalSphericalShellGrid) = momentum_advection
validate_momentum_advection(momentum_advection::VectorInvariant, grid::OrthogonalSphericalShellGrid) = momentum_advection
validate_momentum_advection(momentum_advection, grid::OrthogonalSphericalShellGrid) = error("$(typeof(momentum_advection)) is not supported with $(typeof(grid))")

initialize!(model::HydrostaticFreeSurfaceModel) = initialize_free_surface!(model.free_surface, model.grid, model.velocities)
initialize_free_surface!(free_surface, grid, velocities) = nothing

# return the total advective velocities
@inline total_velocities(model::HydrostaticFreeSurfaceModel) = model.velocities

