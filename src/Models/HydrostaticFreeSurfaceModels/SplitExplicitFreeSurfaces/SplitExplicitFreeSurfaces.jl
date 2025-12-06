module SplitExplicitFreeSurfaces

export SplitExplicitFreeSurface, ForwardBackwardScheme, AdamsBashforth3Scheme
export FixedSubstepNumber, FixedTimeStepSize

using Oceananigans.Architectures: convert_to_device, architecture
using Oceananigans.Utils: KernelParameters, configure_kernel, launch!, @apply_regionally
using Oceananigans.Operators: Az⁻¹ᶜᶜᶠ, Δx_qᶜᶠᶠ, Δy_qᶠᶜᶠ, Δzᶜᶠᶜ, Δzᶠᶜᶜ, δxTᶜᵃᵃ, δyTᵃᶜᵃ, ∂xTᶠᶜᶠ, ∂yTᶜᶠᶠ
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.Fields: Field
using Oceananigans.Grids: Center, Face, get_active_column_map, topology
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: AbstractFreeSurface,
                                                        free_surface_displacement_field

using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Grids: column_depthᶜᶠᵃ,
                          column_depthᶠᶜᵃ

import Oceananigans.Models.HydrostaticFreeSurfaceModels: initialize_free_surface!,
                                                         materialize_free_surface,
                                                         step_free_surface!,
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
