module BoundaryConditions

export
    BCType, Flux, Gradient, Value, NoPenetration,
    BoundaryCondition, bctype, getbc, setbc!,
    PeriodicBoundaryCondition, NoPenetrationBoundaryCondition, NoFluxBoundaryCondition,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition,
    CoordinateBoundaryConditions,
    FieldBoundaryConditions, UVelocityBoundaryConditions, VVelocityBoundaryConditions,
    WVelocityBoundaryConditions, TracerBoundaryConditions, PressureBoundaryConditions,
    DiffusivityBoundaryConditions,
    BoundaryFunction,
    apply_z_bcs!, apply_y_bcs!,
    fill_halo_regions!, zero_halo_regions!

using CUDAnative

using Oceananigans: Cell, Face
using Oceananigans.Architectures
using Oceananigans.Grids

include("boundary_condition_types.jl")
include("boundary_condition.jl")
include("coordinate_boundary_conditions.jl")
include("field_boundary_conditions.jl")
include("boundary_function.jl")
include("show_boundary_conditions.jl")

include("fill_halo_regions.jl")
include("zero_halo_regions.jl")

include("apply_flux_bcs.jl")
include("apply_value_gradient_bcs.jl")

end
