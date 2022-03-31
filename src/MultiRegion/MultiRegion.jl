module MultiRegion

export MultiRegionGrid, MultiRegionField
export XPartition

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

using KernelAbstractions: Event, NoneEvent

import Base: show, length, size

import Oceananigans.Utils:
                getdevice,
                switch_device!,
                devices,
                isregional,
                getregion

abstract type AbstractMultiGrid{FT, TX, TY, TZ, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch} end

abstract type AbstractPartition end

getname(type) = typeof(type).name.wrapper

include("multi_region_utils.jl")
include("x_partitions.jl")
include("multi_region_grid.jl")
include("multi_region_field.jl")
include("multi_region_boundary_conditions.jl")
include("multi_region_reductions.jl")
# include("matrix_implicit_free_surface_distributed_solver.jl")
include("delete_me.jl")

end #module