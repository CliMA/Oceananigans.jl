module MultiRegion

export MultiRegionGrid, MultiRegionField
export XPartition, YPartition, Connectivity
export CubedSpherePartition, ConformalCubedSphereGrid, CubedSphereField

using Oceananigans: Oceananigans
using Oceananigans.Architectures: AbstractArchitecture, CPU, GPU, architecture
using Oceananigans.BoundaryConditions: BoundaryConditions, East, North, South, SouthAndNorth, West, WestAndEast,
    fill_halo_regions!
using Oceananigans.DistributedComputations: DistributedComputations
using Oceananigans.Fields: Field, location
using Oceananigans.Grids: Grids, AbstractGrid, AbstractUnderlyingGrid, Bounded, Center, Face, Flat,
    FullyConnected, LeftConnected, OrthogonalSphericalShellGrid, Periodic, RightConnected, on_architecture
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Utils: KernelParameters, Iterate, MultiRegionObject, Reference, Utils,
    @apply_regionally, apply_regionally!, construct_regionally, isregional, getregion, _getregion, regions

using DocStringExtensions: TYPEDFIELDS
using OffsetArrays: OffsetArray

using KernelAbstractions: KernelAbstractions as KA
using KernelAbstractions: @kernel, @index

abstract type AbstractMultiRegionGrid{FT, TX, TY, TZ, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch} end

abstract type AbstractPartition end

abstract type AbstractConnectivity end

struct XPartition{N} <: AbstractPartition
    div :: N

    function XPartition(sizes)
        div = length(sizes) > 1 && allequal(sizes) ? length(sizes) : sizes

        return new{typeof(div)}(div)
    end
end

struct YPartition{N} <: AbstractPartition
    div :: N

    function YPartition(sizes)
        div = length(sizes) > 1 && allequal(sizes) ? length(sizes) : sizes

        return new{typeof(div)}(div)
    end
end

include("multi_region_utils.jl")
include("multi_region_connectivity.jl")
include("x_partitions.jl")
include("y_partitions.jl")
include("cubed_sphere_partitions.jl")
include("cubed_sphere_connectivity.jl")
include("multi_region_grid.jl")
include("cubed_sphere_grid.jl")
include("cubed_sphere_field.jl")
include("cubed_sphere_boundary_conditions.jl")
include("multi_region_field.jl")
include("multi_region_abstract_operations.jl")
include("multi_region_boundary_conditions.jl")
include("multi_region_reductions.jl")
include("multi_region_models.jl")
include("multi_region_output_writers.jl")

end #module
