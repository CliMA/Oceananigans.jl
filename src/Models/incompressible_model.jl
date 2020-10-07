using CUDA
using OrderedCollections: OrderedDict

using Oceananigans.Advection

using Oceananigans: AbstractOutputWriter, AbstractDiagnostic, TimeStepper

using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.Utils: inflate_halo_size, tupleit
using Oceananigans.Grids: with_halo
using Oceananigans.Buoyancy: validate_buoyancy
using Oceananigans.TurbulenceClosures: ν₀, κ₀, with_tracers
using Oceananigans.Forcings: model_forcing

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
                  clock = Clock{float_type}(0, 0, 1),
              advection = CenteredSecondOrder(),
               buoyancy = SeawaterBuoyancy(float_type),
               coriolis = nothing,
          surface_waves = nothing,
                forcing = NamedTuple(),
                closure = IsotropicDiffusivity(float_type, ν=ν₀, κ=κ₀),
    boundary_conditions = NamedTuple(),
                tracers = (:T, :S),
            timestepper = :QuasiAdamsBashforth2,
             velocities = nothing,
              pressures = nothing,
          diffusivities = nothing,
        pressure_solver = nothing
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
    - `forcing`: `NamedTuple` of user-defined forcing functions that contribute to solution tendencies.
    - `boundary_conditions`: `NamedTuple` containing field boundary conditions.
    - `tracers`: A tuple of symbols defining the names of the modeled tracers, or a `NamedTuple` of
                 preallocated `CellField`s.
    - `timestepper`: A symbol that specifies the time-stepping method. Either `:QuasiAdamsBashforth2` or
                     `:RungeKutta3`.
"""
function IncompressibleModel(;
                   grid,
           architecture = CPU(),
             float_type = Float64,
                  clock = Clock{float_type}(0, 0, 1),
              advection = CenteredSecondOrder(),
               buoyancy = SeawaterBuoyancy(float_type),
               coriolis = nothing,
          surface_waves = nothing,
                forcing = NamedTuple(),
                closure = IsotropicDiffusivity(float_type, ν=ν₀, κ=κ₀),
    boundary_conditions = NamedTuple(),
                tracers = (:T, :S),
            timestepper = :QuasiAdamsBashforth2,
             velocities = nothing,
              pressures = nothing,
          diffusivities = nothing,
        pressure_solver = nothing,
    )

    if architecture == GPU() && !has_cuda()
         throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))
    end

    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)
    validate_buoyancy(buoyancy, tracernames(tracers))

    # Adjust halos when the advection scheme or turbulence closure requires it.
    # Note that halos are isotropic by default; however we respect user-input here
    # by adjusting each (x, y, z) halo individually.
    Hx, Hy, Hz = inflate_halo_size(grid.Hx, grid.Hy, grid.Hz, advection, closure)
    grid = with_halo((Hx, Hy, Hz), grid)

    # Either check grid-correctness, or construct tuples of fields
    velocities    = VelocityFields(velocities, architecture, grid, boundary_conditions)
    tracers       = TracerFields(tracers,      architecture, grid, boundary_conditions)
    pressures     = PressureFields(pressures,  architecture, grid, boundary_conditions)
    diffusivities = DiffusivityFields(diffusivities, architecture, grid,
                                      tracernames(tracers), boundary_conditions, closure)
                                      
    pressure_solver = PressureSolver(pressure_solver, architecture, grid, PressureBoundaryConditions(grid))

    # Instantiate timestepper if not already instantiated
    timestepper = TimeStepper(timestepper, architecture, grid, tracernames(tracers))

    # Regularize forcing and closure for model tracer and velocity fields.
    forcing = model_forcing(tracernames(tracers); forcing...)
    closure = with_tracers(tracernames(tracers), closure)

    return IncompressibleModel(architecture, grid, clock, advection, buoyancy, coriolis, surface_waves,
                               forcing, closure, velocities, tracers, pressures, diffusivities,
                               timestepper, pressure_solver)
end
