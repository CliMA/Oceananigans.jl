module SplitExplicitFreeSurfaces

export SplitExplicitFreeSurface, ForwardBackwardScheme, AdamsBashforth3Scheme
export FixedSubstepNumber, FixedTimeStepSize

using Oceananigans
using Oceananigans.Architectures
using Oceananigans.Architectures: convert_args
using Oceananigans.Fields
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.BoundaryConditions
using Oceananigans.Grids: AbstractGrid, topology
using Oceananigans.ImmersedBoundaries: active_linear_index_to_tuple, mask_immersed_field!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: AbstractFreeSurface,
                                                        free_surface_displacement_field

using Adapt
using Base
using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll

import Oceananigans.Models.HydrostaticFreeSurfaceModels: initialize_free_surface!,
                                                         materialize_free_surface,
                                                         ab2_step_free_surface!,
                                                         compute_free_surface_tendency!,
                                                         explicit_barotropic_pressure_x_gradient,
                                                         explicit_barotropic_pressure_y_gradient

include("split_explicit_timesteppers.jl")
include("split_explicit_free_surface.jl")
include("distributed_split_explicit_free_surface.jl")
include("initialize_split_explicit_substepping.jl")
include("compute_slow_tendencies.jl")
include("step_split_explicit_free_surface.jl")
include("barotropic_split_explicit_corrector.jl")

# extend
@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = zero(grid)
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = zero(grid)

end
