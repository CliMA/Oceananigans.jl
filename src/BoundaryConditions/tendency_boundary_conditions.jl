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
