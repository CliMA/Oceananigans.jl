module BoundaryConditions

export
    BCType, Flux, Gradient, Value, Open,
    BoundaryCondition, getbc, setbc!,
    PeriodicBoundaryCondition, OpenBoundaryCondition, NoFluxBoundaryCondition, MultiRegionCommunicationBoundaryCondition, 
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition, DistributedCommunicationBoundaryCondition,
    validate_boundary_condition_topology, validate_boundary_condition_architecture,
    FieldBoundaryConditions,
    apply_x_bcs!, apply_y_bcs!, apply_z_bcs!,
    fill_halo_regions!

using CUDA, Adapt
using KernelAbstractions: @index, @kernel

using Oceananigans.Architectures: CPU, GPU, device
using Oceananigans.Utils: work_layout, launch!
using Oceananigans.Operators: Ax, Ay, Az, volume
using Oceananigans.Grids

import Adapt: adapt_structure

include("boundary_condition_classifications.jl")
include("boundary_condition.jl")
include("discrete_boundary_function.jl")
include("continuous_boundary_function.jl")
include("field_boundary_conditions.jl")
include("show_boundary_conditions.jl")

include("fill_halo_regions.jl")
include("fill_halo_regions_value_gradient.jl")
include("fill_halo_regions_open.jl")
include("fill_halo_regions_periodic.jl")
include("fill_halo_regions_flux.jl")
include("fill_halo_regions_nothing.jl")

include("apply_flux_bcs.jl")

include("update_boundary_conditions.jl")
include("polar_boundary_condition.jl")

include("flat_extrapolation_open_boundary_matching_scheme.jl")
include("perturbation_advection_open_boundary_matching_scheme.jl")
end # module
