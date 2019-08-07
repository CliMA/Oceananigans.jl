using .TurbulenceClosures

mutable struct Model{A<:Architecture, GR, T, EOS<:EquationOfState, 
                     PC<:PlanetaryConstants, 
                     VC, TR, PF, F, TC, BCS, TS, PS, D}

              arch :: A                      # Computer `Architecture` on which `Model` is run
              grid :: GR                     # Grid of physical points on which `Model` is solved
             clock :: Clock{T}               # Tracks iteration number and simulation time of `Model`
               eos :: EOS                    # Relationship between temperature, salinity, and buoyancy
         constants :: PC                     # Set of physical constants, inc. gravitational acceleration
        velocities :: VC                     # Container for velocity fields `u`, `v`, and `w`
           tracers :: TR                     # Container for tracer fields
         pressures :: PF                     # Container for hydrostatic and nonhydrostatic pressure
           forcing :: F                      # Container for forcing functions defined by the user
           closure :: TC                     # Diffusive 'turbulence closure' for all model fields
    boundary_conditions :: BCS               # Container for 3d bcs on all fields
       timestepper :: TS                     # Object containing timestepper fields and parameters
    poisson_solver :: PS                     # Poisson Solver
     diffusivities :: D                      # Container for turbulent diffusivities
    output_writers :: Array{OutputWriter, 1} # Objects that write data to disk
       diagnostics :: Array{Diagnostic, 1}   # Objects that calc diagnostics on-line during simulation

end


"""
    Model(; kwargs...)

Construct an `Oceananigans.jl` model.
"""
function Model(;
    # Model resolution and domain size
             N,
             L,
    # Model architecture and floating point precision
          arch = CPU(),
    float_type = Float64,
          grid = RegularCartesianGrid(float_type, N, L),
    # Isotropic transport coefficients (exposed to `Model` constructor for convenience)
             ν = 1.05e-6, νh=ν, νv=ν,
             κ = 1.43e-7, κh=κ, κv=κ,
       closure = ConstantAnisotropicDiffusivity(float_type, νh=νh, νv=νv, κh=κh, κv=κv),
    # Time stepping
    start_time = 0,
     iteration = 0,
         clock = Clock{float_type}(start_time, iteration),
    # Fluid and physical parameters
     constants = Earth(float_type),
           eos = LinearEquationOfState(float_type),
    # Forcing and boundary conditions for (u, v, w, T, S)
       forcing = Forcing(),
           bcs = HorizontallyPeriodicModelBCs(),
    boundary_conditions = bcs,
    # Output and diagonstics
    output_writers = OutputWriter[],
       diagnostics = Diagnostic[]
    )

    arch == GPU() && !HAVE_CUDA && throw(
        ArgumentError("Cannot create a GPU model. No CUDA-enabled GPU was detected!"))

    # Initialize fields.
       velocities = VelocityFields(arch, grid)
          tracers = TracerFields(arch, grid)
        pressures = PressureFields(arch, grid)
      timestepper = AdamsBashforthTimestepper(float_type, arch, grid, 0.125)
    diffusivities = TurbulentDiffusivities(arch, grid, closure)

    # Initialize Poisson solver.
    poisson_solver = PoissonSolver(arch, PoissonBCs(bcs), grid)

    # Set the default initial condition
    initialize_with_defaults!(eos, tracers)

    Model(arch, grid, clock, eos, constants, velocities, tracers, 
          pressures, forcing, closure, boundary_conditions, timestepper, 
          poisson_solver, diffusivities, output_writers, diagnostics)
end

"""
    ChannelModel(; kwargs...)

    Construct a `Model` with walls in the y-direction. This is done by imposing
    `FreeSlip` boundary conditions in the y-direction instead of `Periodic`.

    kwargs are passed to the regular `Model` constructor.
"""
ChannelModel(; bcs=ChannelModelBCs(), kwargs...) = 
    Model(; bcs=bcs, kwargs...)
          
#
# Model initialization utilities
#

arch(model::Model{A}) where A <: Architecture = A
float_type(m::Model) = eltype(model.grid)
add_bcs!(model::Model; kwargs...) = add_bcs(model.boundary_conditions; kwargs...)


function initialize_with_defaults!(eos::EquationOfState, tracers, sets...)
    # Default tracer initial condition is deteremined by eos.
    tracers.S.data.parent .= eos.S₀
    tracers.T.data.parent .= eos.T₀
    initialize_with_zeros!(sets...)
    return nothing
end

function initialize_with_zeros!(sets...)
    # Set all fields to 0
    for set in sets
        for fldname in propertynames(set)
            fld = getproperty(set, fldname)
            fld.data.parent .= 0 # promotes to eltype of fld.data
        end
    end
    return nothing
end

"""
    Forcing(; kwargs...)

Return a named tuple of forcing functions 
for each solution field.
"""
Forcing(; Fu=zerofunk, Fv=zerofunk, Fw=zerofunk, FT=zerofunk, FS=zerofunk) = 
    (u=Fu, v=Fv, w=Fw, T=FT, S=FS)

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
function TracerFields(arch, grid)
    T = CellField(arch, grid)  # Temperature θ to avoid conflict with type T.
    S = CellField(arch, grid)
    return (T=T, S=S)
end

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
    Tendencies(arch, grid)

Return a NamedTuple with tendencies for all solution fields
(velocity fields and tracer fields), initialized on 
the architecture `arch` and `grid`.
"""
function Tendencies(arch, grid)
    Gu = FaceFieldX(arch, grid)
    Gv = FaceFieldY(arch, grid)
    Gw = FaceFieldZ(arch, grid)
    GT = CellField(arch, grid)
    GS = CellField(arch, grid)
    return (Gu=Gu, Gv=Gv, Gw=Gw, GT=GT, GS=GS)
end

"""
    AdamsBashforthTimestepper(float_type, arch, grid, χ)

Return an AdamsBashforthTimestepper object with tendency
fields on `arch` and `grid` and AB2 parameter `χ`.
"""
struct AdamsBashforthTimestepper{T, TG}
    Gⁿ :: TG
    G⁻ :: TG
     χ :: T
end

function AdamsBashforthTimestepper(float_type, arch, grid, χ)
   Gⁿ = Tendencies(arch, grid)
   G⁻ = Tendencies(arch, grid)
   return AdamsBashforthTimestepper{float_type, typeof(Gⁿ)}(Gⁿ, G⁻, χ)
end
