using CUDA: has_cuda
using OrderedCollections: OrderedDict

using Oceananigans: AbstractModel, AbstractOutputWriter, AbstractDiagnostic

using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.Advection: CenteredSecondOrder
using Oceananigans.BuoyancyModels: validate_buoyancy, regularize_buoyancy, SeawaterBuoyancy
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Fields: BackgroundFields, Field, tracernames, VelocityFields, TracerFields, PressureFields
using Oceananigans.Forcings: model_forcing
using Oceananigans.Grids: inflate_halo_size, with_halo
using Oceananigans.Solvers: FFTBasedPoissonSolver
using Oceananigans.TimeSteppers: Clock, TimeStepper
using Oceananigans.TurbulenceClosures: with_tracers, DiffusivityFields, time_discretization
using Oceananigans.LagrangianParticleTracking: LagrangianParticles
using Oceananigans.Utils: tupleit
using Oceananigans.Grids: topology

const ParticlesOrNothing = Union{Nothing, LagrangianParticles}

mutable struct IncompressibleModel{TS, E, A<:AbstractArchitecture, G, T, B, R, SD, U, C, Φ, F,
                                   V, S, K, BG, P, I} <: AbstractModel{TS}

         architecture :: A         # Computer `Architecture` on which `Model` is run
                 grid :: G         # Grid of physical points on which `Model` is solved
                clock :: Clock{T}  # Tracks iteration number and simulation time of `Model`
            advection :: V         # Advection scheme for velocities _and_ tracers
             buoyancy :: B         # Set of parameters for buoyancy model
             coriolis :: R         # Set of parameters for the background rotation rate of `Model`
         stokes_drift :: SD        # Set of parameters for surfaces waves via the Craik-Leibovich approximation
              forcing :: F         # Container for forcing functions defined by the user
              closure :: E         # Diffusive 'turbulence closure' for all model fields
    background_fields :: BG        # Background velocity and tracer fields
            particles :: P         # Particle set for Lagrangian tracking
           velocities :: U         # Container for velocity fields `u`, `v`, and `w`
              tracers :: C         # Container for tracer fields
            pressures :: Φ         # Container for hydrostatic and nonhydrostatic pressure
        diffusivities :: K         # Container for turbulent diffusivities
          timestepper :: TS        # Object containing timestepper fields and parameters
      pressure_solver :: S         # Pressure/Poisson solver
    immersed_boundary :: I         # Models the physics of immersed boundaries within the grid
end

"""
    IncompressibleModel(;
                   grid,
           architecture = CPU(),
             float_type = Float64,
                  clock = Clock{float_type}(0, 0, 1),
              advection = CenteredSecondOrder(),
               buoyancy = Buoyancy(SeawaterBuoyancy(float_type)),
               coriolis = nothing,
           stokes_drift = nothing,
                forcing = NamedTuple(),
                closure = nothing,
    boundary_conditions = NamedTuple(),
                tracers = (:T, :S),
            timestepper = :QuasiAdamsBashforth2,
      background_fields = NamedTuple(),
              particles = nothing,
             velocities = nothing,
              pressures = nothing,
          diffusivities = nothing,
        pressure_solver = nothing,
      immersed_boundary = nothing
    )

Construct an incompressible `Oceananigans.jl` model on `grid`.

Keyword arguments
=================

    - `grid`: (required) The resolution and discrete geometry on which `model` is solved.
    - `architecture`: `CPU()` or `GPU()`. The computer architecture used to time-step `model`.
    - `float_type`: `Float32` or `Float64`. The floating point type used for `model` data.
    - `advection`: The scheme that advects velocities and tracers. See `Oceananigans.Advection`.
    - `buoyancy`: The buoyancy model. See `Oceananigans.BuoyancyModels`.
    - `closure`: The turbulence closure for `model`. See `Oceananigans.TurbulenceClosures`.
    - `coriolis`: Parameters for the background rotation rate of the model.
    - `forcing`: `NamedTuple` of user-defined forcing functions that contribute to solution tendencies.
    - `boundary_conditions`: `NamedTuple` containing field boundary conditions.
    - `tracers`: A tuple of symbols defining the names of the modeled tracers, or a `NamedTuple` of
                 preallocated `CenterField`s.
    - `timestepper`: A symbol that specifies the time-stepping method. Either `:QuasiAdamsBashforth2` or
                     `:RungeKutta3`.
"""
function IncompressibleModel(;    grid,
    architecture::AbstractArchitecture = CPU(),
                            float_type = Float64,
                                 clock = Clock{float_type}(0, 0, 1),
                             advection = CenteredSecondOrder(),
                              buoyancy = Buoyancy(model=SeawaterBuoyancy(float_type)),
                              coriolis = nothing,
                          stokes_drift = nothing,
                   forcing::NamedTuple = NamedTuple(),
                               closure = nothing,
       boundary_conditions::NamedTuple = NamedTuple(),
                               tracers = (:T, :S),
                           timestepper = :QuasiAdamsBashforth2,
         background_fields::NamedTuple = NamedTuple(),
         particles::ParticlesOrNothing = nothing,
                            velocities = nothing,
                             pressures = nothing,
                         diffusivities = nothing,
                       pressure_solver = nothing,
                     immersed_boundary = nothing
    )

    if architecture == GPU() && !has_cuda()
         throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))
    end

    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)
    validate_buoyancy(buoyancy, tracernames(tracers))

    buoyancy = regularize_buoyancy(buoyancy)

    # Adjust halos when the advection scheme or turbulence closure requires it.
    # Note that halos are isotropic by default; however we respect user-input here
    # by adjusting each (x, y, z) halo individually.
    Hx, Hy, Hz = inflate_halo_size(grid.Hx, grid.Hy, grid.Hz, topology(grid), advection, closure)
    grid = with_halo((Hx, Hy, Hz), grid)

    # Recursively "regularize" field-dependent boundary conditions by supplying list of tracer names.
    # We also regularize boundary conditions included in velocities, tracers, pressures, and diffusivities.
    # Note that we do not regularize boundary conditions contained in *tupled* diffusivity fields right now.
    embedded_boundary_conditions = merge(extract_boundary_conditions(velocities),
                                         extract_boundary_conditions(tracers),
                                         extract_boundary_conditions(pressures),
                                         extract_boundary_conditions(diffusivities))

    boundary_conditions = merge(embedded_boundary_conditions, boundary_conditions)

    model_field_names = (:u, :v, :w, tracernames(tracers)...)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, model_field_names)

    # Either check grid-correctness, or construct tuples of fields
    velocities    = VelocityFields(velocities, architecture, grid, boundary_conditions)
    tracers       = TracerFields(tracers,      architecture, grid, boundary_conditions)
    pressures     = PressureFields(pressures,  architecture, grid, boundary_conditions)
    diffusivities = DiffusivityFields(diffusivities, architecture, grid,
                                      tracernames(tracers), boundary_conditions, closure)

    if isnothing(pressure_solver)
        pressure_solver = PressureSolver(architecture, grid)
    end

    background_fields = BackgroundFields(background_fields, tracernames(tracers), grid, clock)

    # Instantiate timestepper if not already instantiated
    implicit_solver = implicit_diffusion_solver(time_discretization(closure), architecture, grid)
    timestepper = TimeStepper(timestepper, architecture, grid, tracernames(tracers), implicit_solver=implicit_solver)

    # Regularize forcing and closure for model tracer and velocity fields.
    model_fields = merge(velocities, tracers)
    forcing = model_forcing(model_fields; forcing...)
    closure = with_tracers(tracernames(tracers), closure)

    return IncompressibleModel(architecture, grid, clock, advection, buoyancy, coriolis, stokes_drift,
                               forcing, closure, background_fields, particles, velocities, tracers,
                               pressures, diffusivities, timestepper, pressure_solver, immersed_boundary)
end

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
