using Oceananigans: AbstractModel, AbstractOutputWriter, AbstractDiagnostic

using Oceananigans.Architectures: AbstractArchitecture, CPU
using Oceananigans.Distributed
using Oceananigans.Advection: CenteredSecondOrder
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Fields: Field, tracernames, TracerFields, XFaceField, YFaceField, CenterField
using Oceananigans.Forcings: model_forcing
using Oceananigans.Grids: with_halo, topology, inflate_halo_size, halo_size, Flat, architecture
using Oceananigans.TimeSteppers: Clock, TimeStepper, update_state!
using Oceananigans.TurbulenceClosures: with_tracers, DiffusivityFields
using Oceananigans.Utils: tupleit

import Oceananigans.Architectures: architecture

function ShallowWaterTendencyFields(grid, tracer_names)

    uh = XFaceField(grid)
    vh = YFaceField(grid)
    h = CenterField(grid)

    tracers = TracerFields(tracer_names, grid)
    solution = (; uh, vh, h)

    return merge(solution, tracers)
end

function ShallowWaterSolutionFields(grid, bcs)
    uh = XFaceField(grid, boundary_conditions=bcs.uh)
    vh = YFaceField(grid, boundary_conditions=bcs.vh)
    h = CenterField(grid, boundary_conditions=bcs.h)

    return (; uh, vh, h)
end

mutable struct ShallowWaterModel{G, A<:AbstractArchitecture, T, V, R, F, E, B, Q, C, K, TS} <: AbstractModel{TS}

                          grid :: G         # Grid of physical points on which `Model` is solved
                  architecture :: A         # Computer `Architecture` on which `Model` is run
                         clock :: Clock{T}  # Tracks iteration number and simulation time of `Model`
    gravitational_acceleration :: T         # Gravitational acceleration, full, or reduced
                     advection :: V         # Advection scheme for velocities _and_ tracers
                      coriolis :: R         # Set of parameters for the background rotation rate of `Model`
                       forcing :: F         # Container for forcing functions defined by the user
                       closure :: E         # Diffusive 'turbulence closure' for all model fields
                    bathymetry :: B         # Bathymetry/Topography for the model
                      solution :: Q         # Container for transports `uh`, `vh`, and height `h`
                       tracers :: C         # Container for tracer fields
            diffusivity_fields :: K         # Container for turbulent diffusivities
                   timestepper :: TS        # Object containing timestepper fields and parameters

end

"""
    ShallowWaterModel(; grid,
                        gravitational_acceleration,
                            clock = Clock{eltype(grid)}(0, 0, 1),
                        advection = UpwindBiasedFifthOrder(),
                         coriolis = nothing,
              forcing::NamedTuple = NamedTuple(),
                          closure = nothing,
                       bathymetry = nothing,
                          tracers = (),
               diffusivity_fields = nothing,
  boundary_conditions::NamedTuple = NamedTuple(),
              timestepper::Symbol = :RungeKutta3)

Construct a shallow water `Oceananigans.jl` model on `grid` with `gravitational_acceleration` constant.

Keyword arguments
=================

  - `grid`: (required) The resolution and discrete geometry on which `model` is solved.
  - `gravitational_acceleration`: (required) The gravitational accelaration constant.
  - `clock`: The `clock` for the model
  - `advection`: The scheme that advects velocities and tracers. See `Oceananigans.Advection`.
  - `coriolis`: Parameters for the background rotation rate of the model.
  - `forcing`: `NamedTuple` of user-defined forcing functions that contribute to solution tendencies.
  - `bathymetry`: The bottom bathymetry.
  - `tracers`: A tuple of symbols defining the names of the modeled tracers, or a `NamedTuple` of
               preallocated `CenterField`s.
  - `diffusivity_fields`: Stores diffusivity fields when the closures require a diffusivity to be
                          calculated at each timestep.
  - `boundary_conditions`: `NamedTuple` containing field boundary conditions.
  - `timestepper`: A symbol that specifies the time-stepping method. Either `:QuasiAdamsBashforth2`,
                   `:RungeKutta3`.
"""
function ShallowWaterModel(;
                           grid,
                           gravitational_acceleration,
                               clock = Clock{eltype(grid)}(0, 0, 1),
                           advection = UpwindBiasedFifthOrder(),
                            coriolis = nothing,
                 forcing::NamedTuple = NamedTuple(),
                             closure = nothing,
                          bathymetry = nothing,
                             tracers = (),
                  diffusivity_fields = nothing,
     boundary_conditions::NamedTuple = NamedTuple(),
                 timestepper::Symbol = :RungeKutta3)

    arch = architecture(grid)

    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    @assert topology(grid, 3) === Flat "ShallowWaterModel requires `topology(grid, 3) === Flat`. " *
                                       "Use `topology = ($(topology(grid, 1)), $(topology(grid, 2)), Flat)` " *
                                       "when constructing `grid`."

    Hx, Hy, Hz = inflate_halo_size(grid.Hx, grid.Hy, 0, topology(grid), advection, closure)
    any((grid.Hx, grid.Hy, grid.Hz) .< (Hx, Hy, 0)) && # halos are too small, remake grid
        (grid = with_halo((Hx, Hy, 0), grid))

    prognostic_field_names = (:uh, :vh, :h, tracers...)
    default_boundary_conditions = NamedTuple{prognostic_field_names}(Tuple(FieldBoundaryConditions()
                                                                           for name in prognostic_field_names))

    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, prognostic_field_names)

    solution           = ShallowWaterSolutionFields(grid, boundary_conditions)
    tracers            = TracerFields(tracers, grid, boundary_conditions)
    diffusivity_fields = DiffusivityFields(diffusivity_fields, grid, tracernames(tracers), boundary_conditions, closure)

    # Instantiate timestepper if not already instantiated
    timestepper = TimeStepper(timestepper, grid, tracernames(tracers);
                              Gⁿ = ShallowWaterTendencyFields(grid, tracernames(tracers)),
                              G⁻ = ShallowWaterTendencyFields(grid, tracernames(tracers)))

    # Regularize forcing and closure for model tracer and velocity fields.
    model_fields = merge(solution, tracers)
    forcing = model_forcing(model_fields; forcing...)
    closure = with_tracers(tracernames(tracers), closure)

    model = ShallowWaterModel(grid,
                              arch,
                              clock,
                              eltype(grid)(gravitational_acceleration),
                              advection,
                              coriolis,
                              forcing,
                              closure,
                              bathymetry,
                              solution,
                              tracers,
                              diffusivity_fields,
                              timestepper)

    update_state!(model)

    return model
end

architecture(model::ShallowWaterModel) = model.architecture
