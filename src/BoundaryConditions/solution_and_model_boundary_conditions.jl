using Oceananigans.Utils: with_tracers
using Oceananigans.TurbulenceClosures: with_tracers

#####
##### Default boundary conditions on tracers are periodic or no flux and
##### can be derived from boundary conditions on any field
#####

DefaultTracerBC(::BC)  = BoundaryCondition(Flux, nothing)
DefaultTracerBC(::PBC) = PeriodicBC()

DefaultTracerCoordinateBCs(bcs) =
    CoordinateBoundaryConditions(DefaultTracerBC(bcs.left), DefaultTracerBC(bcs.right))

DefaultTracerBoundaryConditions(field_bcs) =
    FieldBoundaryConditions(Tuple(DefaultTracerCoordinateBCs(bcs) for bcs in field_bcs))

"Returns default tracer boundary conditions. For use in `with_tracers`."
default_tracer_bcs(tracer_names, solution_bcs) = DefaultTracerBoundaryConditions(solution_bcs.u)

#####
##### Boundary conditions on tendency terms are derived from the boundary
##### conditions on their repsective fields.
#####

TendencyBC(::BC)   = BoundaryCondition(Flux, nothing)
TendencyBC(::PBC)  = PeriodicBC()
TendencyBC(::NPBC) = NoPenetrationBC()

TendencyCoordinateBCs(bcs) =
    CoordinateBoundaryConditions(TendencyBC(bcs.left), TendencyBC(bcs.right))

TendencyFieldBCs(field_bcs) =
    FieldBoundaryConditions(Tuple(TendencyCoordinateBCs(bcs) for bcs in field_bcs))

TendenciesBoundaryConditions(solution_bcs) =
    NamedTuple{propertynames(solution_bcs)}(Tuple(TendencyFieldBCs(bcs) for bcs in solution_bcs))

#####
##### Boundary conditions on pressure are derived from boundary conditions
##### on the north-south horizontal velocity, v.
#####

# Pressure boundary conditions are either zero flux (Neumann) or Periodic.
PressureBC(::BC)  = BoundaryCondition(Flux, nothing)
PressureBC(::PBC) = PeriodicBC()

function PressureBoundaryConditions(vbcs)
    x = CoordinateBoundaryConditions(PressureBC(vbcs.x.left), PressureBC(vbcs.x.right))
    y = CoordinateBoundaryConditions(PressureBC(vbcs.y.left), PressureBC(vbcs.y.right))
    z = CoordinateBoundaryConditions(PressureBC(vbcs.z.left), PressureBC(vbcs.z.right))
    return (x=x, y=y, z=z)
end

#####
##### Boundary conditions on diffusivities are derived from boundary conditions
##### on the north-south horizontal velocity, v.
#####

# Diffusivity boundary conditions are either zero flux (Neumann) or Periodic.
DiffusivityBC(::BC)  = BoundaryCondition(Flux, nothing)
DiffusivityBC(::PBC) = PeriodicBC()

function DiffusivityBoundaryConditions(vbcs)
    x = CoordinateBoundaryConditions(DiffusivityBC(vbcs.x.left), DiffusivityBC(vbcs.x.right))
    y = CoordinateBoundaryConditions(DiffusivityBC(vbcs.y.left), DiffusivityBC(vbcs.y.right))
    z = CoordinateBoundaryConditions(DiffusivityBC(vbcs.z.left), DiffusivityBC(vbcs.z.right))
    return (x=x, y=y, z=z)
end

DiffusivitiesBoundaryConditions(::Nothing, args...) = nothing
DiffusivitiesBoundaryConditions(::AbstractField, vbcs) = DiffusivityBoundaryConditions(vbcs)

function DiffusivitiesBoundaryConditions(diffusivities, vbcs)
    κbcs = Dict()

    for κ in propertynames(diffusivities)
        κbcs[κ] = DiffusivitiesBoundaryConditions(diffusivities[κ], vbcs)
    end

    return (; κbcs...)
end

#####
##### SolutionBoundaryConditions on the tuple of velocity fields and tracer fields
##### called the model "solution"
#####

"""
    SolutionBoundaryConditions(tracers, proposal_bcs)

Construct a `NamedTuple` of `FieldBoundaryConditions` for a model with
fields `u`, `v`, `w`, and `tracers` from the proposal boundary conditions
`proposal_bcs`, which must contain the boundary conditions on `u`, `v`, and `w`
and may contain some or all of the boundary conditions on `tracers`.
"""
SolutionBoundaryConditions(tracer_names, proposal_bcs) =
    with_tracers(tracer_names, proposal_bcs, default_tracer_bcs, with_velocities=true)

"""
    HorizontallyPeriodicSolutionBCs(u=HorizontallyPeriodicBCs(), ...)

Construct `SolutionBoundaryConditions` for a horizontally-periodic model
configuration with solution fields `u`, `v`, `w`, `T`, and `S` specified by keyword arguments.

By default `HorizontallyPeriodicBCs` are applied to `u`, `v`, `T`, and `S`
and `HorizontallyPeriodicBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC())` is applied to `w`.

Use `HorizontallyPeriodicBCs` when constructing non-default boundary conditions for `u`, `v`, `w`, `T`, `S`.
"""
function HorizontallyPeriodicSolutionBCs(;
    u = HorizontallyPeriodicBCs(),
    v = HorizontallyPeriodicBCs(),
    w = HorizontallyPeriodicBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC()),
    tracers_boundary_conditions...)

    return merge((u=u, v=v, w=w), tracers_boundary_conditions)
end

"""
    ChannelSolutionBCs(u=ChannelBCs(), ...)

Construct `SolutionBoundaryConditions` for a reentrant channel model
configuration with solution fields `u`, `v`, `w`, `T`, and `S` specified by keyword arguments.

By default `ChannelBCs` are applied to `u`, `v`, `T`, and `S`
and `ChannelBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC())` is applied to `w`.

Use `ChannelBCs` when constructing non-default boundary conditions for `u`, `v`, `w`, `T`, `S`.
"""
function ChannelSolutionBCs(;
    u = ChannelBCs(),
    v = ChannelBCs(north=NoPenetrationBC(), south=NoPenetrationBC()),
    w = ChannelBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC()),
    tracers_boundary_conditions...)

    return merge((u=u, v=v, w=w), tracers_boundary_conditions)
end

#####
##### ModelBoundaryConditions, which include boundary conditions on the solution,
##### tendencies, pressure, and diffusivities 
#####

const ModelBoundaryConditions = NamedTuple{(:solution, :tendency, :pressure, :diffusivities)}

"""
    ModelBoundaryConditions(tracer_names, diffusivity_fields, proposal_bcs)

Construct `ModelBoundaryConditions` with defaults for the tendency boundary conditions, pressure boundary
conditions, and diffusivity boundary conditions given a tuple of `tracer_names`, a `NamedTuple` of 
`diffusivity_fields`, and a set of `proposal_boundary_conditions` that includes boundary conditions for
`u`, `v`, and `w` and any of the tracers in `tracer_names`. The boundary conditions on `v` 
are used to determine the topology of the grid for setting default boundary conditions.
"""
function ModelBoundaryConditions(tracer_names, diffusivity_fields, proposal_bcs;
         tendency = TendenciesBoundaryConditions(SolutionBoundaryConditions(tracer_names, proposal_bcs)),
         pressure = PressureBoundaryConditions(proposal_bcs.v),
    diffusivities = DiffusivitiesBoundaryConditions(diffusivity_fields, proposal_bcs.v)
   )

    return (     solution = SolutionBoundaryConditions(tracer_names, proposal_bcs),  
                 tendency = tendency, 
                 pressure = pressure, 
            diffusivities = diffusivities)
end

ModelBoundaryConditions(::Tuple, diffusivities, model_boundary_conditions::ModelBoundaryConditions) =
    model_boundary_conditions
