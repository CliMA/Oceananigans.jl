module SplitExplicitFreeSurfaces

export SplitExplicitFreeSurface, ForwardBackwardScheme, CoriolisForwardBackwardScheme
export FixedSubstepNumber, FixedTimeStepSize

using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @index, @kernel, @localmem, @synchronize, StaticSize

using Oceananigans: Oceananigans
using Oceananigans.Architectures: architecture, convert_to_device, device
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, SurfaceWaveRadiationBoundaryCondition,
                                       cache_boundary_conditions!, fill_halo_regions!, gravity_wave_boundary_condition,
                                       regularize_field_boundary_conditions, step_boundary_conditions!
using Oceananigans.DistributedComputations: DistributedComputations, @root, DistributedGrid, all_reduce
using Oceananigans.Fields: Field, instantiated_location
using Oceananigans.Grids: Center, Connected, Face, Open, Periodic, TwoSided, column_depthᶜᶠᵃ, column_depthᶠᶜᵃ, topology
using Oceananigans.Coriolis: fᶠᶠᵃ
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, Δxᶜᶠᶠ, Δyᶠᶜᶠ, Δx⁻¹ᶠᶜᶠ, Δy⁻¹ᶜᶠᶠ
using Oceananigans.ImmersedBoundaries: column_depthTᶜᶠᵃ, column_depthTᶠᶜᵃ, column_depthᶜᶠᵃ, column_depthᶠᶜᵃ,
                                       mask_immersed_field!
using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModels,
                                                        AbstractFreeSurface,
                                                        free_surface_displacement_field,
                                                        update_vertical_velocities!
using Oceananigans.Operators: Az⁻¹ᶜᶜᶠ, Δx_qᶜᶠᶠ, Δxᶜᶠᵃ, Δy_qᶠᶜᶠ, Δyᶠᶜᵃ, Δzᶜᶠᶜ, Δzᶠᶜᶜ, δxTᶜᵃᵃ, δxᶜᶜᶜ, δyTᵃᶜᵃ, δyᶜᶜᶜ,
                              ∂xᵣTᶠᶜᶠ, ∂xᵣᶠᶜᶠ, ∂yᵣTᶜᶠᶠ, ∂yᵣᶜᶠᶠ
using Oceananigans.Units: Time
using Oceananigans.Utils: Utils, @apply_regionally, KernelParameters, add_time_interval, configure_kernel, launch!

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
@inline HydrostaticFreeSurfaceModels.explicit_barotropic_pressure_x_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = zero(grid)
@inline HydrostaticFreeSurfaceModels.explicit_barotropic_pressure_y_gradient(i, j, k, grid, ::SplitExplicitFreeSurface) = zero(grid)

end
