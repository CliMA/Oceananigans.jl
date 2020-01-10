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

#####
##### Boundary conditions for model solutions
#####

default_tracer_bcs(tracers, solution_bcs) = DefaultTracerBoundaryConditions(solution_bcs[1])
