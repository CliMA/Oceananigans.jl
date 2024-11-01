module SplitExplicitFreeSurfaces

export SplitExplicitFreeSurface, ForwardBackwardScheme, AdamsBashforth3Scheme
export FixedSubstepNumber, FixedTimeStepSize

using Oceananigans
using Oceananigans.Architectures
using Oceananigans.Fields
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Models.HydrostaticFreeSurfaceModels: AbstractFreeSurface

using Adapt
using Base
using KernelAbstractions: @index, @kernel

import Oceananigans.Models.HydrostaticFreeSurfaceModels: initialize_free_surface!,
                                                         setup_free_surface!,
                                                         materialize_free_surface,
                                                         ab2_step_free_surface!

include("split_explicit_timesteppers.jl")
include("split_explicit_free_surface.jl")
include("distributed_split_explicit_free_surface.jl")
include("setup_split_explicit.jl")
include("substepping_kernels.jl")
include("barotropic_correction.jl")

# extend
@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = zero(grid)
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = zero(grid)

end