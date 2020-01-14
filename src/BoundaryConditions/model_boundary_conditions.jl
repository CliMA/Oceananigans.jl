const ModelBoundaryConditions = NamedTuple{(:solution, :tendency, :pressure)}

function ModelBoundaryConditions(tracers, proposal_bcs::NamedTuple)
    solution_boundary_conditions = SolutionBoundaryConditions(tracers, proposal_bcs)

    model_boundary_conditions = (solution = solution_boundary_conditions,
                                 tendency = TendenciesBoundaryConditions(solution_boundary_conditions),
                                 pressure = PressureBoundaryConditions(solution_boundary_conditions.v))
    return model_boundary_conditions
end

ModelBoundaryConditions(tracers, model_boundary_conditions::ModelBoundaryConditions) =
    model_boundary_conditions
