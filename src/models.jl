using .TurbulenceClosures: ν₀, κ₀

mutable struct Model{TS, E, A<:AbstractArchitecture, G, T, B, R, U, C, Φ, F,
                     BCS, S, K, OW, DI, Θ} <: AbstractModel

           architecture :: A         # Computer `Architecture` on which `Model` is run
                   grid :: G         # Grid of physical points on which `Model` is solved
                  clock :: Clock{T}  # Tracks iteration number and simulation time of `Model`
               buoyancy :: B         # Set of parameters for buoyancy model
               coriolis :: R         # Set of parameters for the background rotation rate of `Model`
             velocities :: U         # Container for velocity fields `u`, `v`, and `w`
                tracers :: C         # Container for tracer fields
              pressures :: Φ         # Container for hydrostatic and nonhydrostatic pressure
                forcing :: F         # Container for forcing functions defined by the user
                closure :: E         # Diffusive 'turbulence closure' for all model fields
    boundary_conditions :: BCS       # Container for 3d bcs on all fields
            timestepper :: TS        # Object containing timestepper fields and parameters
         poisson_solver :: S         # Poisson Solver
          diffusivities :: K         # Container for turbulent diffusivities
         output_writers :: OW        # Objects that write data to disk
            diagnostics :: DI        # Objects that calc diagnostics on-line during simulation
             parameters :: Θ         # Container for arbitrary user-defined parameters
end

"""
    Model(; grid, kwargs...)

Construct an `Oceananigans.jl` model on `grid`.

Keyword arguments
=================
- `grid`: (required) The resolution and discrete geometry on which `model` is solved. Currently the only option is
  `RegularCartesianGrid`.
- `architecture`: `CPU()` or `GPU()`. The computer architecture used to time-step `model`.
- `float_type`: `Float32` or `Float64`. The floating point type used for `model` data.
- `closure`: The turbulence closure for `model`. See `TurbulenceClosures`.
- `buoyancy`: Buoyancy model parameters.
- `coriolis`: Parameters for the background rotation rate of the model.
- `forcing`: User-defined forcing functions that contribute to solution tendencies.
- `boundary_conditions`: User-defined boundary conditions for model fields. Can be either`SolutionBoundaryConditions`
  or `ModelBoundaryConditions`. See `BoundaryConditions`, `HorizontallyPeriodicSolutionBCs`, and `ChannelSolutionBCs`.
- `parameters`: User-defined parameters for use in user-defined forcing functions and boundary condition functions.
"""
function Model(;
                   grid,
           architecture = CPU(),
             float_type = Float64,
                tracers = (:T, :S),
                closure = ConstantIsotropicDiffusivity(float_type, ν=ν₀, κ=κ₀),
                  clock = Clock{float_type}(0, 0),
               buoyancy = SeawaterBuoyancy(float_type),
               coriolis = nothing,
                forcing = ModelForcing(),
    boundary_conditions = HorizontallyPeriodicSolutionBCs(),
         output_writers = OrderedDict{Symbol, AbstractOutputWriter}(),
            diagnostics = OrderedDict{Symbol, AbstractDiagnostic}(),
             parameters = nothing,
             velocities = VelocityFields(architecture, grid),
              pressures = PressureFields(architecture, grid),
          diffusivities = TurbulentDiffusivities(architecture, grid, tracernames(tracers), closure),
            timestepper = :AdamsBashforth,
         poisson_solver = PoissonSolver(architecture, PoissonBCs(boundary_conditions), grid)
    )

    if architecture == GPU()
        !has_cuda() && throw(ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))
        if mod(grid.Nx, 16) != 0 || mod(grid.Ny, 16) != 0
            throw(ArgumentError("For GPU models, Nx and Ny must be multiples of 16."))
        end
    end

    timestepper = TimeStepper(timestepper, float_type, architecture, grid, tracernames(tracers))

    tracers = TracerFields(architecture, grid, tracers)
    validate_buoyancy(buoyancy, tracernames(tracers))

    # Regularize forcing, boundary conditions, and closure for given tracer fields
    forcing = ModelForcing(tracernames(tracers), forcing)
    boundary_conditions = ModelBoundaryConditions(tracernames(tracers), boundary_conditions)
    closure = with_tracers(tracernames(tracers), closure)

    return Model(architecture, grid, clock, buoyancy, coriolis, velocities, tracers,
                 pressures, forcing, closure, boundary_conditions, timestepper,
                 poisson_solver, diffusivities, output_writers, diagnostics, parameters)
end

"""Show the innards of a `Model` in the REPL."""
Base.show(io::IO, model::Model) =
    print(io,
              "Oceananigans.Model on a ", typeof(model.architecture), " architecture (time = ", 
                                          prettytime(model.clock.time), ", iteration = ", 
                                          model.clock.iteration, ") \n",
              "├── grid: ", typeof(model.grid), '\n',
              "├── tracers: ", tracernames(model.tracers), '\n',
              "├── closure: ", typeof(model.closure), '\n',
              "├── buoyancy: ", typeof(model.buoyancy), '\n',
              "├── coriolis: ", typeof(model.coriolis), '\n',
              "├── output writers: ", ordered_dict_show(model.output_writers, "│"), '\n',
              "└── diagnostics: ", ordered_dict_show(model.diagnostics, " "))
              

"""
    ChannelModel(; kwargs...)

Construct a `Model` with walls in the y-direction. This is done by imposing
`FreeSlip` boundary conditions in the y-direction instead of `Periodic`.

kwargs are passed to the regular `Model` constructor.
"""
ChannelModel(; boundary_conditions=ChannelSolutionBCs(), kwargs...) =
    Model(; boundary_conditions=boundary_conditions, kwargs...)

"""
    NonDimensionalModel(; N, L, Re, Pr=0.7, Ro=Inf, float_type=Float64, kwargs...)

Construct a "Non-dimensional" `Model` with resolution `N`, domain extent `L`,
precision `float_type`, and the four non-dimensional numbers:

    * `Re = U λ / ν` (Reynolds number)
    * `Pr = U λ / κ` (Prandtl number)
    * `Ro = U / f λ` (Rossby number)

for characteristic velocity scale `U`, length-scale `λ`, viscosity `ν`,
tracer diffusivity `κ`, and Coriolis parameter `f`. Buoyancy is scaled
with `λ U²`, so that the Richardson number is `Ri=B`, where `B` is a
non-dimensional buoyancy scale set by the user via initial conditions or
forcing.

Note that `N`, `L`, and `Re` are required.

Additional `kwargs` are passed to the regular `Model` constructor.
"""
function NonDimensionalModel(; grid, float_type=Float64, Re, Pr=0.7, Ro=Inf,
    buoyancy = BuoyancyTracer(),
    coriolis = FPlane(float_type, f=1/Ro),
     closure = ConstantIsotropicDiffusivity(float_type, ν=1/Re, κ=1/(Pr*Re)),
    kwargs...)

    return Model(; float_type=float_type, grid=grid, closure=closure,
                   coriolis=coriolis, tracers=(:b,), buoyancy=buoyancy, kwargs...)
end

#####
##### Utils
#####

float_type(m::AbstractModel) = eltype(model.grid)

"""
    VelocityFields(arch, grid)

Return a NamedTuple with fields `u`, `v`, `w` initialized on
the architecture `arch` and `grid`.
"""
function VelocityFields(arch, grid)
    u = FaceFieldX(arch, grid)
    v = FaceFieldY(arch, grid)
    w = FaceFieldZ(arch, grid)
    return (u=u, v=v, w=w)
end

"""
    TracerFields(arch, grid)

Return a NamedTuple with tracer fields initialized
as `CellField`s on the architecture `arch` and `grid`.
"""
function TracerFields(arch, grid, tracernames)
    tracerfields = Tuple(CellField(arch, grid) for c in tracernames)
    return NamedTuple{tracernames}(tracerfields)
end

TracerFields(arch, grid, ::Union{Tuple{}, Nothing}) = NamedTuple{()}(())
TracerFields(arch, grid, tracer::Symbol) = TracerFields(arch, grid, tuple(tracer))
TracerFields(arch, grid, tracers::NamedTuple) = tracers

tracernames(::Nothing) = ()
tracernames(name::Symbol) = tuple(name)
tracernames(names::NTuple{N, Symbol}) where N = :u ∈ names ? names[4:end] : names
tracernames(::NamedTuple{names}) where names = tracernames(names)

"""
    PressureFields(arch, grid)

Return a NamedTuple with pressure fields `pHY′` and `pNHS`
initialized as `CellField`s on the architecture `arch` and `grid`.
"""
function PressureFields(arch, grid)
    pHY′ = CellField(arch, grid)
    pNHS = CellField(arch, grid)
    return (pHY′=pHY′, pNHS=pNHS)
end

"""
    Tendencies(arch, grid, tracernames)

Return a NamedTuple with tendencies for all solution fields
(velocity fields and tracer fields), initialized on
the architecture `arch` and `grid`.
"""
function Tendencies(arch, grid, tracernames)

    velocities = (u = FaceFieldX(arch, grid),
                  v = FaceFieldY(arch, grid),
                  w = FaceFieldZ(arch, grid))

    tracers = TracerFields(arch, grid, tracernames)

    return merge(velocities, tracers)
end
