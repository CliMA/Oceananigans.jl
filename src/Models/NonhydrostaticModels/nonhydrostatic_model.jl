using CUDA: has_cuda
using OrderedCollections: OrderedDict

using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Advection: CenteredSecondOrder, adapt_advection_order
using Oceananigans.BuoyancyModels: validate_buoyancy, regularize_buoyancy, SeawaterBuoyancy
using Oceananigans.Biogeochemistry: validate_biogeochemistry, AbstractBiogeochemistry, biogeochemical_auxiliary_fields
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Fields: Field, tracernames, VelocityFields, TracerFields, CenterField
using Oceananigans.Forcings: model_forcing
using Oceananigans.Grids: inflate_halo_size, with_halo, architecture
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Models: AbstractModel, NaNChecker, extract_boundary_conditions
using Oceananigans.Solvers: FFTBasedPoissonSolver
using Oceananigans.TimeSteppers: Clock, TimeStepper, update_state!, AbstractLagrangianParticles
using Oceananigans.TurbulenceClosures: validate_closure, with_tracers, DiffusivityFields, time_discretization, implicit_diffusion_solver
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: FlavorOfCATKE
using Oceananigans.Utils: tupleit
using Oceananigans.Grids: topology

import Oceananigans.Architectures: architecture
import Oceananigans.Models: total_velocities, default_nan_checker, timestepper

const ParticlesOrNothing = Union{Nothing, AbstractLagrangianParticles}
const AbstractBGCOrNothing = Union{Nothing, AbstractBiogeochemistry}
const BFOrNamedTuple = Union{BackgroundFields, NamedTuple}

# TODO: this concept may be more generally useful,
# but for now we use it only for hydrostatic pressure anomalies for now.
struct DefaultHydrostaticPressureAnomaly end

mutable struct NonhydrostaticModel{TS, E, A<:AbstractArchitecture, G, T, B, R, SD, U, C, Φ, F,
                                   V, S, K, BG, P, BGC, AF} <: AbstractModel{TS}

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
      biogeochemistry :: BGC      # Biogeochemistry for Oceananigans tracers
           velocities :: U        # Container for velocity fields `u`, `v`, and `w`
              tracers :: C        # Container for tracer fields
            pressures :: Φ        # Container for hydrostatic and nonhydrostatic pressure
   diffusivity_fields :: K        # Container for turbulent diffusivities
          timestepper :: TS       # Object containing timestepper fields and parameters
      pressure_solver :: S        # Pressure/Poisson solver
     auxiliary_fields :: AF       # User-specified auxiliary fields for forcing functions and boundary conditions
end

"""
    NonhydrostaticModel(;           grid,
                                    clock = Clock{eltype(grid)}(time = 0),
                                advection = CenteredSecondOrder(),
                                 buoyancy = nothing,
                                 coriolis = nothing,
                             stokes_drift = nothing,
                      forcing::NamedTuple = NamedTuple(),
                                  closure = nothing,
          boundary_conditions::NamedTuple = NamedTuple(),
                                  tracers = (),
                              timestepper = :RungeKutta3,
            background_fields::NamedTuple = NamedTuple(),
            particles::ParticlesOrNothing = nothing,
    biogeochemistry::AbstractBGCOrNothing = nothing,
                               velocities = nothing,
                  nonhydrostatic_pressure = CenterField(grid),
             hydrostatic_pressure_anomaly = DefaultHydrostaticPressureAnomaly(),
                       diffusivity_fields = nothing,
                          pressure_solver = nothing,
                         auxiliary_fields = NamedTuple())

Construct a model for a non-hydrostatic, incompressible fluid on `grid`, using the Boussinesq
approximation when `buoyancy != nothing`. By default, all Bounded directions are rigid and impenetrable.

Keyword arguments
=================

  - `grid`: (required) The resolution and discrete geometry on which the `model` is solved. The
            architecture (CPU/GPU) that the model is solved on is inferred from the architecture
            of the `grid`. Note that the grid needs to be regularly spaced in the horizontal
            dimensions, ``x`` and ``y``.
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
                   `:RungeKutta3` (default).
  - `background_fields`: `NamedTuple` with background fields (e.g., background flow). Default: `nothing`.
  - `particles`: Lagrangian particles to be advected with the flow. Default: `nothing`.
  - `biogeochemistry`: Biogeochemical model for `tracers`.
  - `velocities`: The model velocities. Default: `nothing`.
  - `nonhydrostatic_pressure`: The nonhydrostatic pressure field. Default: `CenterField(grid)`.
  - `hydrostatic_pressure_anomaly`: An optional field that stores the part of the nonhydrostatic pressure
                                    in hydrostatic balance with the buoyancy field. If `CenterField(grid)` (default), the anomaly is precomputed by
                                    vertically integrating the buoyancy field. In this case, the `nonhydrostatic_pressure` represents
                                    only the part of pressure that deviates from the hydrostatic anomaly. If `nothing`, the anomaly
                                    is not computed. 
  - `diffusivity_fields`: Diffusivity fields. Default: `nothing`.
  - `pressure_solver`: Pressure solver to be used in the model. If `nothing` (default), the model constructor
    chooses the default based on the `grid` provide.
  - `auxiliary_fields`: `NamedTuple` of auxiliary fields. Default: `nothing`         
"""
function NonhydrostaticModel(; grid,
                             clock = Clock{eltype(grid)}(time = 0),
                             advection = CenteredSecondOrder(),
                             buoyancy = nothing,
                             coriolis = nothing,
                             stokes_drift = nothing,
                             forcing::NamedTuple = NamedTuple(),
                             closure = nothing,
                             boundary_conditions::NamedTuple = NamedTuple(),
                             tracers = (),
                             timestepper = :RungeKutta3,
                             background_fields::BFOrNamedTuple = NamedTuple(),
                             particles::ParticlesOrNothing = nothing,
                             biogeochemistry::AbstractBGCOrNothing = nothing,
                             velocities = nothing,
                             hydrostatic_pressure_anomaly = DefaultHydrostaticPressureAnomaly(),
                             nonhydrostatic_pressure = CenterField(grid),
                             diffusivity_fields = nothing,
                             pressure_solver = nothing,
                             auxiliary_fields = NamedTuple())

    arch = architecture(grid)

    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    # Validate pressure fields
    nonhydrostatic_pressure isa Field{Center, Center, Center} ||
        throw(ArgumentError("nonhydrostatic_pressure must be CenterField(grid)."))

    if hydrostatic_pressure_anomaly isa DefaultHydrostaticPressureAnomaly
        # Manage treatment of the hydrostatic pressure anomaly:

        if !isnothing(buoyancy)
            # Separate the hydrostatic pressure anomaly
            # from the nonhydrostatic pressure contribution.
            # See https://github.com/CliMA/Oceananigans.jl/issues/3677
            # and https://github.com/CliMA/Oceananigans.jl/issues/3795.

            hydrostatic_pressure_anomaly = CenterField(grid)
        else
            # Use a single combined pressure, saving memory and computation.

            hydrostatic_pressure_anomaly = nothing
        end
    end

    # Check validity of hydrostatic_pressure_anomaly.
    isnothing(hydrostatic_pressure_anomaly) || hydrostatic_pressure_anomaly isa Field{Center, Center, Center} ||
        throw(ArgumentError("hydrostatic_pressure_anomaly must be `nothing` or `CenterField(grid)`."))

    # We don't support CAKTE for NonhydrostaticModel yet.
    closure = validate_closure(closure)
    first_closure = closure isa Tuple ? first(closure) : closure
    first_closure isa FlavorOfCATKE &&
        error("CATKEVerticalDiffusivity is not supported for NonhydrostaticModel --- yet!")

    all_auxiliary_fields = merge(auxiliary_fields, biogeochemical_auxiliary_fields(biogeochemistry))
    tracers, auxiliary_fields = validate_biogeochemistry(tracers, all_auxiliary_fields, biogeochemistry, grid, clock)
    validate_buoyancy(buoyancy, tracernames(tracers))
    buoyancy = regularize_buoyancy(buoyancy)

    # Adjust advection scheme to be valid on a particular grid size. i.e. if the grid size
    # is smaller than the advection order, reduce the order of the advection in that particular
    # direction
    advection = adapt_advection_order(advection, grid)

    # Adjust halos when the advection scheme or turbulence closure requires it.
    # Note that halos are isotropic by default; however we respect user-input here
    # by adjusting each (x, y, z) halo individually.
    grid = inflate_grid_halo_size(grid, advection, closure)

    # Collect boundary conditions for all model prognostic fields and, if specified, some model
    # auxiliary fields. Boundary conditions are "regularized" based on the _name_ of the field:
    # boundary conditions on u, v, w are regularized assuming they represent momentum at appropriate
    # staggered locations. All other fields are regularized assuming they are tracers.
    # Note that we do not regularize boundary conditions contained in *tupled* diffusivity fields right now.

    # First, we extract boundary conditions that are embedded within any _user-specified_ field tuples:
    embedded_boundary_conditions = merge(extract_boundary_conditions(velocities),
                                         extract_boundary_conditions(tracers),
                                         extract_boundary_conditions(diffusivity_fields))

    # Next, we form a list of default boundary conditions:
    prognostic_field_names = (:u, :v, :w, tracernames(tracers)..., keys(auxiliary_fields)...)
    default_boundary_conditions = NamedTuple{prognostic_field_names}(FieldBoundaryConditions()
                                                                     for name in prognostic_field_names)

    # Finally, we merge specified, embedded, and default boundary conditions. Specified boundary conditions
    # have precedence, followed by embedded, followed by default.
    boundary_conditions = merge(default_boundary_conditions, embedded_boundary_conditions, boundary_conditions)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, prognostic_field_names)

    # Ensure `closure` describes all tracers
    closure = with_tracers(tracernames(tracers), closure)

    # Either check grid-correctness, or construct tuples of fields
    velocities         = VelocityFields(velocities, grid, boundary_conditions)
    tracers            = TracerFields(tracers,      grid, boundary_conditions)
    pressures          = (pNHS=nonhydrostatic_pressure, pHY′=hydrostatic_pressure_anomaly)
    diffusivity_fields = DiffusivityFields(diffusivity_fields, grid, tracernames(tracers), boundary_conditions, closure)

    if isnothing(pressure_solver)
        pressure_solver = nonhydrostatic_pressure_solver(grid)
    end

    # Materialize background fields
    background_fields = BackgroundFields(background_fields, tracernames(tracers), grid, clock)

    # Instantiate timestepper if not already instantiated
    implicit_solver = implicit_diffusion_solver(time_discretization(closure), grid)
    timestepper = TimeStepper(timestepper, grid, tracernames(tracers), implicit_solver=implicit_solver)

    # Regularize forcing for model tracer and velocity fields.
    model_fields = merge(velocities, tracers, auxiliary_fields)
    forcing = model_forcing(model_fields; forcing...)

    model = NonhydrostaticModel(arch, grid, clock, advection, buoyancy, coriolis, stokes_drift,
                                forcing, closure, background_fields, particles, biogeochemistry, velocities, tracers,
                                pressures, diffusivity_fields, timestepper, pressure_solver, auxiliary_fields)

    update_state!(model; compute_tendencies = false)
    
    return model
end

architecture(model::NonhydrostaticModel) = model.architecture

function inflate_grid_halo_size(grid, tendency_terms...)
    user_halo = grid.Hx, grid.Hy, grid.Hz
    required_halo = Hx, Hy, Hz = inflate_halo_size(user_halo..., grid, tendency_terms...)

    if any(user_halo .< required_halo) # Replace grid
        @warn "Inflating model grid halo size to ($Hx, $Hy, $Hz) and recreating grid. " *
              "Note that an ImmersedBoundaryGrid requires an extra halo point in all non-flat directions compared to a non-immersed boundary grid."
              "The model grid will be different from the input grid. To avoid this warning, " *
              "pass halo=($Hx, $Hy, $Hz) when constructing the grid."

        grid = with_halo((Hx, Hy, Hz), grid)
    end

    return grid
end

# return the total advective velocities
@inline total_velocities(m::NonhydrostaticModel) =
    (u = SumOfArrays{2}(m.velocities.u, m.background_fields.velocities.u),
     v = SumOfArrays{2}(m.velocities.v, m.background_fields.velocities.v),
     w = SumOfArrays{2}(m.velocities.w, m.background_fields.velocities.w))

