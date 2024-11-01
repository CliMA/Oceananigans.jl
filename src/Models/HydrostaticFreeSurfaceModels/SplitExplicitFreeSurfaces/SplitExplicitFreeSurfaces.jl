module SplitExplicitFreeSurfaces

export SplitExplicitFreeSurface, ForwardBackwardScheme, AdamsBashforth3Scheme

using Oceananigans
using Oceananigans.Architectures
using Oceananigans.Fields
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.Grids: AbstractGrid

using Adapt
using Base
using KernelAbstractions: @index, @kernel

import Oceananigans.Models.HydrostaticFreeSurfaceModels: initialize_free_surface!,
                                                         setup_free_surface!,
                                                         materialize_free_surface,
                                                         ab2_step_free_surface!



end