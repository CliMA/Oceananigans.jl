module BoundaryConditions

export
    BCType, Periodic, Flux, Gradient, Value, NoPenetration, Dirchlet, Neumann,
    BoundaryCondition, bctype, getbc, setbc!,
    CoordinateBoundaryConditions,
    FieldBoundaryConditions, HorizontallyPeriodicBCs, ChannelBCs,
    SolutionBoundaryConditions, HorizontallyPeriodicSolutionBCs, ChannelSolutionBCs,
    ModelBoundaryConditions, BoundaryFunction,
    apply_z_bcs!, apply_y_bcs!,
    fill_halo_regions!, zero_halo_regions!

using CUDAnative

using Oceananigans.Architectures
using Oceananigans.Fields

include("boundary_condition_types.jl")
include("boundary_condition.jl")
include("coordinate_boundary_conditions.jl")
include("field_boundary_conditions.jl")
include("solution_boundary_conditions.jl")
include("tracer_boundary_conditions.jl")
include("tendency_boundary_conditions.jl")
include("pressure_boundary_conditions.jl")
include("model_boundary_conditions.jl")
include("boundary_function.jl")
include("show_boundary_conditions.jl")

include("fill_halo_regions.jl")
include("zero_halo_regions.jl")

include("apply_flux_bcs.jl")
include("apply_flux_periodic_no_penetration_bcs.jl")
include("apply_value_gradient_bcs.jl")

end
