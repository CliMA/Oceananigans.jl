using CUDA
using OrderedCollections: OrderedDict

using Oceananigans.Advection
using Oceananigans: AbstractOutputWriter, AbstractDiagnostic, TimeStepper
using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.Buoyancy: validate_buoyancy
using Oceananigans.TurbulenceClosures: ν₀, κ₀, with_tracers

mutable struct IncompressibleModel{TS, E, A<:AbstractArchitecture, G, T, B, R, SW, U, C, Φ, F,
                                   V, S, K} <: AbstractModel
       architecture :: A         # Computer `Architecture` on which `Model` is run
               grid :: G         # Grid of physical points on which `Model` is solved
              clock :: Clock{T}  # Tracks iteration number and simulation time of `Model`
          advection :: V         # Advection scheme for velocities _and_ tracers
           buoyancy :: B         # Set of parameters for buoyancy model
           coriolis :: R         # Set of parameters for the background rotation rate of `Model`
      surface_waves :: SW        # Set of parameters for surfaces waves via the Craik-Leibovich approximation
            forcing :: F         # Container for forcing functions defined by the user
            closure :: E         # Diffusive 'turbulence closure' for all model fields
         velocities :: U         # Container for velocity fields `u`, `v`, and `w`
            tracers :: C         # Container for tracer fields
          pressures :: Φ         # Container for hydrostatic and nonhydrostatic pressure
      diffusivities :: K         # Container for turbulent diffusivities
        timestepper :: TS        # Object containing timestepper fields and parameters
    pressure_solver :: S         # Pressure/Poisson solver
end

"""
    IncompressibleModel(;
                   grid,
           architecture = CPU(),
             float_type = Float64,
                  clock = Clock{float_type}(0, 0),
              advection = CenteredSecondOrder(),
               buoyancy = SeawaterBuoyancy(float_type),
               coriolis = nothing,
          surface_waves = nothing,
                forcing = ModelForcing(),
                closure = IsotropicDiffusivity(float_type, ν=ν₀, κ=κ₀),
    boundary_conditions = (u=UVelocityBoundaryConditions(grid),
                           v=VVelocityBoundaryConditions(grid),
                           w=WVelocityBoundaryConditions(grid)),
             velocities = VelocityFields(architecture, grid, boundary_conditions),
                tracers = (:T, :S),
              pressures = PressureFields(architecture, grid, boundary_conditions),
          diffusivities = DiffusivityFields(architecture, grid, tracernames(tracers), boundary_conditions, closure),
            timestepper = :AdamsBashforth,
        pressure_solver = PressureSolver(architecture, grid, PressureBoundaryConditions(grid))
    )

Construct an incompressible `Oceananigans.jl` model on `grid`.

Keyword arguments
=================

    - `grid`: (required) The resolution and discrete geometry on which `model` is solved.
    - `architecture`: `CPU()` or `GPU()`. The computer architecture used to time-step `model`.
    - `float_type`: `Float32` or `Float64`. The floating point type used for `model` data.
    - `advection`: The scheme that advects velocities and tracers. See `Oceananigans.Advection`.
    - `buoyancy`: The buoyancy model. See `Oceananigans.Buoyancy`.
    - `closure`: The turbulence closure for `model`. See `Oceananigans.TurbulenceClosures`.
    - `coriolis`: Parameters for the background rotation rate of the model.
    - `forcing`: User-defined forcing functions that contribute to solution tendencies.
    - `tracers`: A tuple of symbols defining the names of the modeled tracers, or a `NamedTuple` of
                 preallocated `CellField`s.
    - `boundary_conditions`: `NamedTuple` containing field boundary conditions.
"""
function IncompressibleModel(;
                   grid,
           architecture = CPU(),
             float_type = Float64,
                  clock = Clock{float_type}(0, 0),
              advection = CenteredSecondOrder(),
               buoyancy = SeawaterBuoyancy(float_type),
               coriolis = nothing,
          surface_waves = nothing,
                forcing = ModelForcing(),
                closure = IsotropicDiffusivity(float_type, ν=ν₀, κ=κ₀),
    boundary_conditions = (u=UVelocityBoundaryConditions(grid),
                           v=VVelocityBoundaryConditions(grid),
                           w=WVelocityBoundaryConditions(grid)),
             velocities = VelocityFields(architecture, grid, boundary_conditions),
                tracers = (:T, :S),
              pressures = PressureFields(architecture, grid, boundary_conditions),
          diffusivities = DiffusivityFields(architecture, grid, tracernames(tracers), boundary_conditions, closure),
            timestepper = :AdamsBashforth,
        pressure_solver = PressureSolver(architecture, grid, PressureBoundaryConditions(grid))
    )

    if architecture == GPU() && !has_cuda()
         throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))
    end

    validate_buoyancy(buoyancy, tracernames(tracers))

    # Regularize forcing and closure for given tracer fields.
    forcing = ModelForcing(tracernames(tracers), forcing)
    closure = with_tracers(tracernames(tracers), closure)

    # Instantiate tracer fields if not already instantiated
    tracer_fields = TracerFields(architecture, grid, tracers, boundary_conditions)

    # Instantiate timestepper if not already instantiated
    timestepper = TimeStepper(timestepper, float_type, architecture, grid, velocities, tracernames(tracers))

    return IncompressibleModel(architecture, grid, clock, advection, buoyancy, coriolis, surface_waves,
                               forcing, closure, velocities, tracer_fields, pressures, diffusivities,
                               timestepper, pressure_solver)
end
