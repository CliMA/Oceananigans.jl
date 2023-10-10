using Oceananigans: AbstractModel, AbstractOutputWriter, AbstractDiagnostic

using Oceananigans.Architectures: AbstractArchitecture, CPU
using Oceananigans.AbstractOperations: @at, KernelFunctionOperation
using Oceananigans.DistributedComputations
using Oceananigans.Advection: CenteredSecondOrder, VectorInvariant
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Fields: Field, tracernames, TracerFields, XFaceField, YFaceField, CenterField, compute!
using Oceananigans.Forcings: model_forcing
using Oceananigans.Grids: topology, Flat, architecture, RectilinearGrid, Face, Center
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Models: validate_model_halo, NaNChecker, validate_tracer_advection
using Oceananigans.TimeSteppers: Clock, TimeStepper, update_state!
using Oceananigans.TurbulenceClosures: with_tracers, DiffusivityFields
using Oceananigans.Utils: tupleit

import Oceananigans.Architectures: architecture
import Oceananigans.Models: default_nan_checker, timestepper

const RectilinearGrids = Union{RectilinearGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:RectilinearGrid}}

function ShallowWaterTendencyFields(grid, tracer_names, prognostic_names)
    u =  XFaceField(grid)
    v =  YFaceField(grid)
    h = CenterField(grid)
    tracers = TracerFields(tracer_names, grid)

    return NamedTuple{prognostic_names}((u, v, h, Tuple(tracers)...))
end

function ShallowWaterSolutionFields(grid, bcs, prognostic_names)
    u =  XFaceField(grid, boundary_conditions = getproperty(bcs, prognostic_names[1]))
    v =  YFaceField(grid, boundary_conditions = getproperty(bcs, prognostic_names[2]))
    h = CenterField(grid, boundary_conditions = getproperty(bcs, prognostic_names[3]))

    return NamedTuple{prognostic_names[1:3]}((u, v, h))
end

mutable struct ShallowWaterModel{G, A<:AbstractArchitecture, T, GR, V, U, R, F, E, B, Q, C, K, TS, FR} <: AbstractModel{TS}
                          grid :: G         # Grid of physical points on which `Model` is solved
                  architecture :: A         # Computer `Architecture` on which `Model` is run
                         clock :: Clock{T}  # Tracks iteration number and simulation time of `Model`
    gravitational_acceleration :: GR        # Gravitational acceleration, full, or reduced
                     advection :: V         # Advection scheme for velocities, mass and tracers
                    velocities :: U         # Velocities in the shallow water model
                      coriolis :: R         # Set of parameters for the background rotation rate of `Model`
                       forcing :: F         # Container for forcing functions defined by the user
                       closure :: E         # Diffusive 'turbulence closure' for all model fields
                    bathymetry :: B         # Bathymetry/Topography for the model
                      solution :: Q         # Container for transports `uh`, `vh`, and height `h`
                       tracers :: C         # Container for tracer fields
            diffusivity_fields :: K         # Container for turbulent diffusivities
                   timestepper :: TS        # Object containing timestepper fields and parameters
                   formulation :: FR        # Either conservative or vector-invariant
end

struct ConservativeFormulation end

struct VectorInvariantFormulation end

"""
    ShallowWaterModel(; grid,
                        gravitational_acceleration,
                              clock = Clock{eltype(grid)}(0, 0, 1),
                 momentum_advection = UpwindBiasedFifthOrder(),
                   tracer_advection = WENO(),
                     mass_advection = WENO(),
                           coriolis = nothing,
                forcing::NamedTuple = NamedTuple(),
                            closure = nothing,
                         bathymetry = nothing,
                            tracers = (),
                 diffusivity_fields = nothing,
    boundary_conditions::NamedTuple = NamedTuple(),
                timestepper::Symbol = :RungeKutta3,
                        formulation = ConservativeFormulation())

Construct a shallow water model on `grid` with `gravitational_acceleration` constant.

Keyword arguments
=================

  - `grid`: (required) The resolution and discrete geometry on which `model` is solved. The
            architecture (CPU/GPU) that the model is solve is inferred from the architecture
            of the grid.
  - `gravitational_acceleration`: (required) The gravitational acceleration constant.
  - `clock`: The `clock` for the model.
  - `momentum_advection`: The scheme that advects velocities. See `Oceananigans.Advection`.
    Default: `UpwindBiasedFifthOrder()`.
  - `tracer_advection`: The scheme that advects tracers. See `Oceananigans.Advection`. Default: `WENO()`.
  - `mass_advection`: The scheme that advects the mass equation. See `Oceananigans.Advection`. Default:
    `WENO()`.
  - `coriolis`: Parameters for the background rotation rate of the model.
  - `forcing`: `NamedTuple` of user-defined forcing functions that contribute to solution tendencies.
  - `closure`: The turbulence closure for `model`. See `Oceananigans.TurbulenceClosures`.
  - `bathymetry`: The bottom bathymetry.
  - `tracers`: A tuple of symbols defining the names of the modeled tracers, or a `NamedTuple` of
               preallocated `CenterField`s.
  - `diffusivity_fields`: Stores diffusivity fields when the closures require a diffusivity to be
                          calculated at each timestep.
  - `boundary_conditions`: `NamedTuple` containing field boundary conditions.
  - `timestepper`: A symbol that specifies the time-stepping method. Either `:QuasiAdamsBashforth2` or
                   `:RungeKutta3` (default).
  - `formulation`: Whether the dynamics are expressed in conservative form (`ConservativeFormulation()`;
                   default) or in non-conservative form with a vector-invariant formulation for the
                   non-linear terms (`VectorInvariantFormulation()`).

!!! warning "Formulation-grid compatibility requirements"
    The `ConservativeFormulation()` requires `RectilinearGrid`.
    Use `VectorInvariantFormulation()` with `LatitudeLongitudeGrid`.
"""
function ShallowWaterModel(;
                           grid,
                           gravitational_acceleration,
                               clock = Clock{eltype(grid)}(0, 0, 1),
                  momentum_advection = UpwindBiasedFifthOrder(),
                    tracer_advection = WENO(),
                      mass_advection = WENO(),
                            coriolis = nothing,
                 forcing::NamedTuple = NamedTuple(),
                             closure = nothing,
                          bathymetry = nothing,
                             tracers = (),
                  diffusivity_fields = nothing,
     boundary_conditions::NamedTuple = NamedTuple(),
                 timestepper::Symbol = :RungeKutta3,
                         formulation = ConservativeFormulation())

    @warn "The ShallowWaterModel is currently unvalidated, subject to change, and should not be used for scientific research without adequate validation."

    arch = architecture(grid)

    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    topology(grid, 3) === Flat ||
        throw(ArgumentError("ShallowWaterModel requires `topology(grid, 3) === Flat`. " *
                            "Use `topology = ($(topology(grid, 1)), $(topology(grid, 2)), Flat)` " *
                            "when constructing `grid`."))

    (typeof(grid) <: RectilinearGrids || formulation == VectorInvariantFormulation()) ||
        throw(ArgumentError("`ConservativeFormulation()` requires a rectilinear `grid`. \n" *
                            "Use `VectorInvariantFormulation()` or change your grid to a rectilinear one."))

    # Check halos and throw an error if the grid's halo is too small
    validate_model_halo(grid, momentum_advection, tracer_advection, closure)

    prognostic_field_names = formulation isa ConservativeFormulation ? (:uh, :vh, :h, tracers...) :  (:u, :v, :h, tracers...) 
    default_boundary_conditions = NamedTuple{prognostic_field_names}(Tuple(FieldBoundaryConditions()
                                                                           for name in prognostic_field_names))

    momentum_advection = validate_momentum_advection(momentum_advection, formulation)

    if isnothing(tracer_advection)
        tracer_advection_tuple = NamedTuple{tracernames(tracers)}(nothing for tracer in 1:length(tracers))
    else
        default_tracer_advection, tracer_advection = validate_tracer_advection(tracer_advection, grid)

        # Advection schemes
        tracer_advection_tuple = with_tracers(tracernames(tracers),
                                            tracer_advection,
                                            (name, tracer_advection) -> default_tracer_advection,
                                            with_velocities=false)
    end

    advection = merge((momentum=momentum_advection, mass=mass_advection), tracer_advection_tuple)
    
    bathymetry_field = CenterField(grid)
    if !isnothing(bathymetry)
        set!(bathymetry_field, bathymetry)
        fill_halo_regions!(bathymetry_field)
    else
        fill!(bathymetry_field, 0)
    end

    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, prognostic_field_names)

    solution           = ShallowWaterSolutionFields(grid, boundary_conditions, prognostic_field_names)
    tracers            = TracerFields(tracers, grid, boundary_conditions)
    diffusivity_fields = DiffusivityFields(diffusivity_fields, grid, tracernames(tracers), boundary_conditions, closure)

    # Instantiate timestepper if not already instantiated
    timestepper = TimeStepper(timestepper, grid, tracernames(tracers);
                              Gⁿ = ShallowWaterTendencyFields(grid, tracernames(tracers), prognostic_field_names),
                              G⁻ = ShallowWaterTendencyFields(grid, tracernames(tracers), prognostic_field_names))

    # Regularize forcing and closure for model tracer and velocity fields.
    model_fields = merge(solution, tracers)
    forcing = model_forcing(model_fields; forcing...)
    closure = with_tracers(tracernames(tracers), closure)

    model = ShallowWaterModel(grid,
                              arch,
                              clock,
                              eltype(grid)(gravitational_acceleration),
                              advection,
                              shallow_water_velocities(formulation, solution),
                              coriolis,
                              forcing,
                              closure,
                              bathymetry_field,
                              solution,
                              tracers,
                              diffusivity_fields,
                              timestepper,
                              formulation)

    update_state!(model)

    return model
end

validate_momentum_advection(momentum_advection, formulation) = momentum_advection
validate_momentum_advection(momentum_advection, ::VectorInvariantFormulation) =
    throw(ArgumentError("VectorInvariantFormulation requires a vector invariant momentum advection scheme. \n"* 
                        "Use `momentum_advection = VectorInvariant()`."))
validate_momentum_advection(momentum_advection::Union{VectorInvariant, Nothing}, ::VectorInvariantFormulation) = momentum_advection

formulation(model::ShallowWaterModel)  = model.formulation
architecture(model::ShallowWaterModel) = model.architecture

# The w velocity is needed to use generic TurbulenceClosures methods, therefore it is set to nothing
shallow_water_velocities(::VectorInvariantFormulation, solution) = (u = solution.u, v = solution.v, w = nothing)

# TODO: convert u and v into binary operations
function shallow_water_velocities(::ConservativeFormulation, solution)
    u = compute!(Field(solution.uh / solution.h))
    v = compute!(Field(solution.vh / solution.h))
    return (; u, v, w=nothing)
end

shallow_water_velocities(model::ShallowWaterModel) = shallow_water_velocities(model.formulation, model.solution)

shallow_water_fields(velocities, solution, tracers, ::ConservativeFormulation)    = merge(velocities, solution, tracers)
shallow_water_fields(velocities, solution, tracers, ::VectorInvariantFormulation) = merge(solution, (; w = velocities.w), tracers)

