module BoundaryConditions

export
    BCType, Flux, Gradient, Value, NormalFlow,

    BoundaryCondition, bctype, getbc, setbc!,

    PeriodicBoundaryCondition, NormalFlowBoundaryCondition, NoFluxBoundaryCondition,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition,

    CoordinateBoundaryConditions,

    FieldBoundaryConditions, UVelocityBoundaryConditions, VVelocityBoundaryConditions,
    WVelocityBoundaryConditions, TracerBoundaryConditions, PressureBoundaryConditions,

    DiffusivityBoundaryConditions,

    BoundaryFunction, TracerBoundaryCondition,
    UVelocityBoundaryCondition, VVelocityBoundaryCondition, WVelocityBoundaryCondition,

    ParameterizedBoundaryCondition, ParameterizedBoundaryConditionFunction,

    apply_x_bcs!, apply_y_bcs!, apply_z_bcs!,

    fill_halo_regions!, zero_halo_regions!

using CUDA
using KernelAbstractions

using Oceananigans.Architectures: device
using Oceananigans.Utils: work_layout, launch!
using Oceananigans.Grids

include("boundary_condition_types.jl")
include("boundary_condition.jl")
include("coordinate_boundary_conditions.jl")
include("field_boundary_conditions.jl")
include("boundary_function.jl")
include("parameterized_boundary_condition.jl")
include("show_boundary_conditions.jl")

include("fill_halo_regions.jl")
include("zero_halo_regions.jl")

include("apply_flux_bcs.jl")
include("apply_value_gradient_bcs.jl")
include("apply_normal_flow_bcs.jl")

end
