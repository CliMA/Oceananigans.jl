module MultiRegion

export MultiRegionGrid, MultiRegionField
export XPartition, YPartition

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Models
using Oceananigans.Architectures
using Oceananigans.BoundaryConditions
using Oceananigans.Utils
using CUDA
using Adapt
using OffsetArrays

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Utils: Reference, Iterate

using KernelAbstractions: Event, NoneEvent, @kernel, @index

import Base: show, length, size

import Oceananigans.Utils:
                getdevice,
                switch_device!,
                devices,
                isregional,
                getregion,
                _getregion,
                sync_all_devices!

abstract type AbstractMultiRegionGrid{FT, TX, TY, TZ, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch} end

abstract type AbstractPartition end

getname(type) = typeof(type).name.wrapper

include("multi_region_utils.jl")
include("x_partitions.jl")
include("y_partitions.jl")
include("multi_region_grid.jl")
include("multi_region_field.jl")
include("multi_region_abstract_operations.jl")
include("multi_region_boundary_conditions.jl")
include("multi_region_reductions.jl")
include("unified_heptadiagonal_iterative_solver.jl")
include("unified_implicit_free_surface_solver.jl")
include("multi_region_models.jl")
include("multi_region_output_writers.jl")

end #module
