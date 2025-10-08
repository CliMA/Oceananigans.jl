module BoundaryConditions

export
    BCType, Flux, Gradient, Value, Open,
    BoundaryCondition, getbc, setbc!,
    PeriodicBoundaryCondition, OpenBoundaryCondition, NoFluxBoundaryCondition, MultiRegionCommunicationBoundaryCondition,
    FluxBoundaryCondition, ValueBoundaryCondition, GradientBoundaryCondition, DistributedCommunicationBoundaryCondition,
    PerturbationAdvection,
    validate_boundary_condition_topology, validate_boundary_condition_architecture,
    FieldBoundaryConditions,
    compute_x_bcs!, compute_y_bcs!, compute_z_bcs!,
    fill_halo_regions!,
    WestAndEast, SouthAndNorth, BottomAndTop,
    West, East, South, North, Bottom, Top,
    DistributedFillHalo

using Adapt
using KernelAbstractions: @index, @kernel

using Oceananigans.Architectures: CPU, GPU, device
using Oceananigans.Utils: work_layout, launch!
using Oceananigans.Operators: Ax, Ay, Az, volume
using Oceananigans.Grids

import Adapt: adapt_structure

# All possible fill_halo! kernels
struct WestAndEast end
struct SouthAndNorth end
struct BottomAndTop end
struct West end
struct East end
struct South end
struct North end
struct Bottom end
struct Top end

include("boundary_condition_classifications.jl")
include("boundary_condition.jl")
include("discrete_boundary_function.jl")
include("continuous_boundary_function.jl")
include("boundary_condition_ordering.jl")
include("field_boundary_conditions.jl")
include("show_boundary_conditions.jl")

include("fill_halo_regions.jl")
include("fill_halo_regions_value_gradient.jl")
include("fill_halo_regions_open.jl")
include("fill_halo_regions_periodic.jl")
include("fill_halo_regions_flux.jl")
include("fill_halo_regions_zipper.jl")
include("fill_halo_kernels.jl")

include("compute_flux_bcs.jl")

include("update_boundary_conditions.jl")
include("polar_boundary_condition.jl")

include("perturbation_advection.jl")
end # module
