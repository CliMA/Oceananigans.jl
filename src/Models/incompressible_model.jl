using OrderedCollections: OrderedDict
using CUDAapi: has_cuda

using Oceananigans: AbstractOutputWriter, AbstractDiagnostic, TimeStepper
using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.Buoyancy: validate_buoyancy
using Oceananigans.TurbulenceClosures: ν₀, κ₀, with_tracers

mutable struct IncompressibleModel{TS, E, A<:AbstractArchitecture, G, T, B, R, SW, U, C, Φ, F,
                                   S, K, Θ} <: AbstractModel
       architecture :: A         # Computer `Architecture` on which `Model` is run
               grid :: G         # Grid of physical points on which `Model` is solved
              clock :: Clock{T}  # Tracks iteration number and simulation time of `Model`
           buoyancy :: B         # Set of parameters for buoyancy model
           coriolis :: R         # Set of parameters for the background rotation rate of `Model`
      surface_waves :: SW        # Set of parameters for surfaces waves via the Craik-Leibovich approximation
         velocities :: U         # Container for velocity fields `u`, `v`, and `w`
            tracers :: C         # Container for tracer fields
          pressures :: Φ         # Container for hydrostatic and nonhydrostatic pressure
            forcing :: F         # Container for forcing functions defined by the user
            closure :: E         # Diffusive 'turbulence closure' for all model fields
        timestepper :: TS        # Object containing timestepper fields and parameters
    pressure_solver :: S         # Pressure/Poisson solver
      diffusivities :: K         # Container for turbulent diffusivities
         parameters :: Θ         # Container for arbitrary user-defined parameters
end

"""
   IncompressibleModel(;
                   grid,
           architecture = CPU(),
             float_type = Float64,
                tracers = (:T, :S),
                closure = ConstantIsotropicDiffusivity(float_type, ν=ν₀, κ=κ₀),
                  clock = Clock{float_type}(0, 0),
               buoyancy = SeawaterBuoyancy(float_type),
               coriolis = nothing,
          surface_waves = nothing,
                forcing = ModelForcing(),
    boundary_conditions = (u=UVelocityBoundaryConditions(grid),
                           v=VVelocityBoundaryConditions(grid),
                           w=WVelocityBoundaryConditions(grid)),
             parameters = nothing,
             velocities = VelocityFields(architecture, grid, boundary_conditions),
          tracer_fields = TracerFields(architecture, grid, tracernames(tracers), boundary_conditions),
              pressures = PressureFields(architecture, grid, boundary_conditions),
          diffusivities = DiffusivityFields(architecture, grid, tracernames(tracers), boundary_conditions, closure),
     timestepper_method = :AdamsBashforth,
            timestepper = TimeStepper(timestepper_method, float_type, architecture, grid, tracernames(tracers)),
        pressure_solver = PressureSolver(architecture, grid, PressureBoundaryConditions(grid))
    )

Construct an incompressible `Oceananigans.jl` model on `grid`.

Keyword arguments
=================
- `grid`: (required) The resolution and discrete geometry on which `model` is solved.
- `architecture`: `CPU()` or `GPU()`. The computer architecture used to time-step `model`.
- `float_type`: `Float32` or `Float64`. The floating point type used for `model` data.
- `closure`: The turbulence closure for `model`. See `TurbulenceClosures`.
- `buoyancy`: Buoyancy model parameters.
- `coriolis`: Parameters for the background rotation rate of the model.
- `forcing`: User-defined forcing functions that contribute to solution tendencies.
- `boundary_conditions`: Named tuple containing field boundary conditions.
- `parameters`: User-defined parameters for use in user-defined forcing functions and boundary condition functions.
"""
function IncompressibleModel(;
                   grid,
           architecture = CPU(),
             float_type = Float64,
                tracers = (:T, :S),
                closure = ConstantIsotropicDiffusivity(float_type, ν=ν₀, κ=κ₀),
                  clock = Clock{float_type}(0, 0),
               buoyancy = SeawaterBuoyancy(float_type),
               coriolis = nothing,
          surface_waves = nothing,
                forcing = ModelForcing(),
    boundary_conditions = (u=UVelocityBoundaryConditions(grid),
                           v=VVelocityBoundaryConditions(grid),
                           w=WVelocityBoundaryConditions(grid)),
             parameters = nothing,
             velocities = VelocityFields(architecture, grid, boundary_conditions),
          tracer_fields = TracerFields(architecture, grid, tracernames(tracers), boundary_conditions),
              pressures = PressureFields(architecture, grid, boundary_conditions),
          diffusivities = DiffusivityFields(architecture, grid, tracernames(tracers), boundary_conditions, closure),
     timestepper_method = :AdamsBashforth,
            timestepper = TimeStepper(timestepper_method, float_type, architecture, grid, tracernames(tracers)),
        pressure_solver = PressureSolver(architecture, grid, PressureBoundaryConditions(grid))
    )

    if architecture == GPU() && !has_cuda()
         throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))
    end

    validate_buoyancy(buoyancy, tracernames(tracers))

    # Regularize forcing and closure for given tracer fields.
    forcing = ModelForcing(tracernames(tracers), forcing)
    closure = with_tracers(tracernames(tracers), closure)

    return IncompressibleModel(architecture, grid, clock, buoyancy, coriolis, surface_waves,
                               velocities, tracer_fields, pressures, forcing, closure,
                               timestepper, pressure_solver, diffusivities, parameters)
end
