using CUDA: has_cuda
using OrderedCollections: OrderedDict

using Oceananigans: AbstractModel, AbstractOutputWriter, AbstractDiagnostic

using Oceananigans.Architectures: AbstractArchitecture, GPU
using Oceananigans.Advection: AbstractAdvectionScheme, CenteredSecondOrder
using Oceananigans.Buoyancy: validate_buoyancy, SeawaterBuoyancy, g_Earth
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions, TracerBoundaryConditions
using Oceananigans.Fields: Field, CenterField, tracernames, VelocityFields, TracerFields
using Oceananigans.Forcings: model_forcing
using Oceananigans.Grids: with_halo
using Oceananigans.Models.IncompressibleModels: extract_boundary_conditions
using Oceananigans.TimeSteppers: Clock, TimeStepper
using Oceananigans.TurbulenceClosures: ν₀, κ₀, with_tracers, DiffusivityFields, IsotropicDiffusivity
using Oceananigans.LagrangianParticleTracking: LagrangianParticles
using Oceananigans.Utils: inflate_halo_size, tupleit

struct VectorInvariant end

struct ExplicitFreeSurface{E, T}
    η :: E
    gravitational_acceleration :: T
end

ExplicitFreeSurface(; gravitational_acceleration=g_Earth) =
    ExplicitFreeSurface(nothing, gravitational_acceleration)

function FreeSurface(free_surface::ExplicitFreeSurface{Nothing}, arch, grid)
    η = CenterField(arch, grid, TracerBoundaryConditions(grid))
    g = convert(eltype(grid), free_surface.gravitational_acceleration)
    return ExplicitFreeSurface(η, g)
end

""" Returns a default_tracer_advection, tracer_advection `tuple`. """
validate_tracer_advection(invalid_tracer_advection, grid) = error("$invalid_tracer_advection is invalid tracer_advection!")
validate_tracer_advection(tracer_advection_tuple::NamedTuple, grid) = CenteredSecondOrder(), advection_scheme_tuple
validate_tracer_advection(tracer_advection::AbstractAdvectionScheme, grid) = tracer_advection, NamedTuple()

mutable struct HydrostaticFreeSurfaceModel{TS, E, A<:AbstractArchitecture,
                                           G, T, V, B, R, S, F, P, U, C, Φ, K} <: AbstractModel{TS}
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
       velocities :: U        # Container for velocity fields `u`, `v`, and `w`
          tracers :: C        # Container for tracer fields
         pressure :: Φ        # Container for hydrostatic pressure
    diffusivities :: K        # Container for turbulent diffusivities
      timestepper :: TS       # Object containing timestepper fields and parameters
end

"""
    HydrostaticFreeSurfaceModel(;
                   grid,
           architecture = CPU(),
                  clock = Clock{eltype(grid)}(0, 0, 1),
              advection = CenteredSecondOrder(),
               buoyancy = SeawaterBuoyancy(eltype(grid)),
               coriolis = nothing,
                forcing = NamedTuple(),
                closure = IsotropicDiffusivity(eltype(grid), ν=ν₀, κ=κ₀),
    boundary_conditions = NamedTuple(),
                tracers = (:T, :S),
              particles = nothing,
             velocities = nothing,
               pressure = nothing,
          diffusivities = nothing,
    )

Construct an hydrostatic `Oceananigans.jl` model with a free surface on `grid`.

Keyword arguments
=================

    - `grid`: (required) The resolution and discrete geometry on which `model` is solved.
    - `architecture`: `CPU()` or `GPU()`. The computer architecture used to time-step `model`.
    - `gravitational_acceleration`: The gravitational acceleration applied to the free surface
    - `advection`: The scheme that advects velocities and tracers. See `Oceananigans.Advection`.
    - `buoyancy`: The buoyancy model. See `Oceananigans.Buoyancy`.
    - `closure`: The turbulence closure for `model`. See `Oceananigans.TurbulenceClosures`.
    - `coriolis`: Parameters for the background rotation rate of the model.
    - `forcing`: `NamedTuple` of user-defined forcing functions that contribute to solution tendencies.
    - `boundary_conditions`: `NamedTuple` containing field boundary conditions.
    - `tracers`: A tuple of symbols defining the names of the modeled tracers, or a `NamedTuple` of
                 preallocated `CenterField`s.
"""
function HydrostaticFreeSurfaceModel(; grid,
                architecture::AbstractArchitecture = CPU(),
                                             clock = Clock{eltype(grid)}(0, 0, 1),
                                momentum_advection = CenteredSecondOrder(),
                                  tracer_advection = CenteredSecondOrder(),
                                          buoyancy = SeawaterBuoyancy(eltype(grid)),
                                          coriolis = nothing,
                                      free_surface = ExplicitFreeSurface(gravitational_acceleration=g_Earth),
                               forcing::NamedTuple = NamedTuple(),
                                           closure = IsotropicDiffusivity(eltype(grid), ν=ν₀, κ=κ₀),
                   boundary_conditions::NamedTuple = NamedTuple(),
                                           tracers = (:T, :S),
    particles::Union{Nothing, LagrangianParticles} = nothing,
                                        velocities = nothing,
                                          pressure = nothing,
                                     diffusivities = nothing,
    )

    if architecture == GPU() && !has_cuda()
         throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))
    end

    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)
    validate_buoyancy(buoyancy, tracernames(tracers))

    # Recursively "regularize" field-dependent boundary conditions by supplying list of tracer names.
    # We also regularize boundary conditions included in velocities, tracers, pressure, and diffusivities.
    # Note that we do not regularize boundary conditions contained in *tupled* diffusivity fields right now.
    embedded_boundary_conditions = merge(extract_boundary_conditions(velocities),
                                         extract_boundary_conditions(tracers),
                                         extract_boundary_conditions(pressure),
                                         extract_boundary_conditions(diffusivities))

    boundary_conditions = merge(embedded_boundary_conditions, boundary_conditions)

    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, tracernames(tracers), nothing)

    # Either check grid-correctness, or construct tuples of fields
    velocities    = VelocityFields(velocities, architecture, grid, boundary_conditions)
    tracers       = TracerFields(tracers,      architecture, grid, boundary_conditions)
    pressure      = (pHY′ = CenterField(architecture, grid, TracerBoundaryConditions(grid)),)
    diffusivities = DiffusivityFields(diffusivities, architecture, grid,
                                      tracernames(tracers), boundary_conditions, closure)

    # Instantiate timestepper if not already instantiated
    timestepper = TimeStepper(:QuasiAdamsBashforth2, architecture, grid, tracernames(tracers);
                              Gⁿ = HydrostaticFreeSurfaceTendencyFields(architecture, grid, tracernames(tracers)),
                              G⁻ = HydrostaticFreeSurfaceTendencyFields(architecture, grid, tracernames(tracers)))

    free_surface = FreeSurface(free_surface, architecture, grid)

    # Regularize forcing and closure for model tracer and velocity fields.
    model_fields = merge((u=velocities.u, v=velocities.v, η=free_surface.η), tracers)
    forcing = model_forcing(model_fields; forcing...)
    closure = with_tracers(tracernames(tracers), closure)

    default_tracer_advection, tracer_advection = validate_tracer_advection(tracer_advection, grid)

    # Advection schemes
    tracer_advection_tuple = with_tracers(tracernames(tracers),
                                          tracer_advection,
                                          (name, tracer_advection) -> default_tracer_advection,
                                          with_velocities=false)

    advection = merge((momentum=momentum_advection,), tracer_advection_tuple)

    return HydrostaticFreeSurfaceModel(architecture, grid, clock, advection, buoyancy, coriolis,
                                       free_surface, forcing, closure, particles, velocities, tracers,
                                       pressure, diffusivities, timestepper)
end
