module SplitExplicitFreeSurfaces

export SplitExplicitFreeSurface, ForwardBackwardScheme, CoriolisForwardBackwardScheme
export FixedSubstepNumber, FixedTimeStepSize

using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @index, @kernel, @localmem, @synchronize, StaticSize

using Oceananigans.Architectures: convert_to_device, architecture, device
using Oceananigans.Utils: Utils, KernelParameters, configure_kernel, launch!, @apply_regionally, add_time_interval
using Oceananigans.Operators: Az⁻¹ᶜᶜᶠ, Δx_qᶜᶠᶠ, Δy_qᶠᶜᶠ, Δzᶜᶠᶜ, Δzᶠᶜᶜ, Δxᶜᶠᵃ, Δyᶠᶜᵃ
using Oceananigans.ImmersedBoundaries: column_depthTᶠᶜᵃ, column_depthTᶜᶠᵃ, column_depthᶠᶜᵃ, column_depthᶜᶠᵃ
using Oceananigans.Operators: ∂xᵣTᶠᶜᶠ, ∂xᵣᶠᶜᶠ, ∂yᵣTᶜᶠᶠ, ∂yᵣᶜᶠᶠ, δxTᶜᵃᵃ, δyTᵃᶜᵃ, δxᶜᶜᶜ, δyᶜᶜᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!, FieldBoundaryConditions, SurfaceWaveRadiationBoundaryCondition, gravity_wave_boundary_condition,
                                       cache_boundary_conditions!, step_boundary_conditions!, regularize_field_boundary_conditions
using Oceananigans.Fields: Field, instantiated_location
using Oceananigans.Grids: Center, Face, Periodic, topology, column_depthᶜᶠᵃ, column_depthᶠᶜᵃ, Connected, Open, TwoSided
using Oceananigans.DistributedComputations: DistributedGrid, all_reduce, @root
using Oceananigans.Coriolis: fᶠᶠᵃ
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, Δxᶜᶠᶠ, Δyᶠᶜᶠ, Δx⁻¹ᶠᶜᶠ, Δy⁻¹ᶜᶠᶠ
using Oceananigans.ImmersedBoundaries: mask_immersed_field!,
                                       column_depthTᶠᶜᵃ, column_depthTᶜᶠᵃ,
                                       column_depthᶠᶜᵃ, column_depthᶜᶠᵃ
using Oceananigans.Models.HydrostaticFreeSurfaceModels: AbstractFreeSurface,
                                                        free_surface_displacement_field,
                                                        update_vertical_velocities!
using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.Units: Time

import Oceananigans.Models.HydrostaticFreeSurfaceModels: reconcile_free_surface!,
                                                         materialize_free_surface,
                                                         default_free_surface_boundary_conditions,
                                                         validate_free_surface_boundary_conditions,
                                                         step_free_surface!,
                                                         compute_free_surface_tendency!,
                                                         compute_transport_velocities!,
                                                         explicit_barotropic_pressure_x_gradient,
                                                         explicit_barotropic_pressure_y_gradient

include("split_explicit_timesteppers.jl")
include("split_explicit_free_surface.jl")
include("distributed_split_explicit_free_surface.jl")
include("initialize_split_explicit_substepping.jl")
include("compute_slow_tendencies.jl")
include("barotropic_targeted_transport.jl")
include("step_split_explicit_free_surface.jl")
include("substep_coriolis.jl")
include("barotropic_split_explicit_corrector.jl")

# extend
@inline explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = zero(grid)
@inline explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = zero(grid)

end
