module Advection

export
    div_𝐯u, div_𝐯v, div_𝐯w, div_Uc,

    U_dot_∇u_hydrostatic_metric, U_dot_∇v_hydrostatic_metric,
    U_dot_∇u_nonhydrostatic_metric, U_dot_∇v_nonhydrostatic_metric,
    U_dot_∇u_metric, U_dot_∇v_metric, U_dot_∇w_metric,

    advective_tracer_flux_x,
    advective_tracer_flux_y,
    advective_tracer_flux_z,

    Centered, UpwindBiased, WENO,
    VectorInvariant, WENOVectorInvariant,
    FluxFormAdvection,
    EnergyConserving,
    EnstrophyConserving

using Adapt: Adapt
using OffsetArrays: OffsetArray
using MuladdMacro: @muladd

using Oceananigans: Oceananigans, fully_supported_float_types
using Oceananigans.Architectures: Architectures, architecture, on_architecture, CPU
using Oceananigans.Grids: Grids, AbstractGrid, Center, Face, Flat, XFlatGrid, YFlatGrid, ZFlatGrid, with_halo,
    SphericalShellGrid, required_halo_size_x, required_halo_size_y, required_halo_size_z
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, ∂t_σ, Ax_qᶠᶜᶜ, Axᶠᶜᶜ, Ay_qᶜᶠᶜ, Ayᶜᶠᶜ, Az_qᶜᶜᶠ,
    Azᶜᶜᶜ, Azᶜᶜᶠ, Az⁻¹ᶜᶠᶜ, Az⁻¹ᶠᶜᶜ, V⁻¹ᶜᶜᶠ, V⁻¹ᶜᶠᶜ, V⁻¹ᶠᶜᶜ, Δx_qᶜᶠᶜ, Δx⁻¹ᶠᶜᶜ, Δy_qᶠᶜᶜ,
    Δy⁻¹ᶜᶠᶜ, covariant_to_volume_flux_uᶠᶜᶜ, covariant_to_volume_flux_vᶜᶠᶜ,
    δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ
using Base: Callable

abstract type AbstractAdvectionScheme{B, FT} end
abstract type AbstractCenteredAdvectionScheme{B, FT} <: AbstractAdvectionScheme{B, FT} end
abstract type AbstractUpwindBiasedAdvectionScheme{B, FT} <: AbstractAdvectionScheme{B, FT} end

# `advection_buffers` specifies the list of buffers for which advection schemes
# are constructed via metaprogramming. (The `advection_buffer` is the width of
# the halo region required for an advection scheme on a non-immersed-boundary grid.)
# An upper limit of `advection_buffer = 6` means we can build advection schemes up to
# `Centered(order=12`) and `UpwindBiased(order=11)`. The list can be extended in order to
# compile schemes with higher orders; for example `advection_buffers = [1, 2, 3, 4, 5, 6, 8]`
# will compile schemes for `advection_buffer=8` and thus `Centered(order=16)` and `UpwindBiased(order=15)`.
# Note that it is not possible to compile schemes for `advection_buffer = 41` or higher.
const advection_buffers = [1, 2, 3, 4, 5, 6]

@inline Base.eltype(::AbstractAdvectionScheme{<:Any, FT}) where FT = FT

@inline Grids.required_halo_size_x(::AbstractAdvectionScheme{B}) where B = B
@inline Grids.required_halo_size_y(::AbstractAdvectionScheme{B}) where B = B
@inline Grids.required_halo_size_z(::AbstractAdvectionScheme{B}) where B = B

@inline spherical_shell_horizontal_tracer_flux_u(U, i, j, k) = @inbounds u_velocity(U)[i, j, k]
@inline spherical_shell_horizontal_tracer_flux_v(U, i, j, k) = @inbounds v_velocity(U)[i, j, k]

struct VolumeFluxField{Loc, G, T}
    grid :: G
    velocities :: T
end

VolumeFluxField(::Val{Loc}, grid, velocities) where Loc =
    VolumeFluxField{Val{Loc}, typeof(grid), typeof(velocities)}(grid, velocities)

@inline u_velocity(velocities) = velocities.u
@inline v_velocity(velocities) = velocities.v
@inline w_velocity(velocities) = velocities.w
@inline u_velocity(velocities::Tuple) = @inbounds velocities[1]
@inline v_velocity(velocities::Tuple) = @inbounds velocities[2]
@inline w_velocity(velocities::Tuple) = @inbounds velocities[3]

@inline spherical_shell_horizontal_volume_flux_velocities(grid::SphericalShellGrid, velocities) =
    (u = VolumeFluxField(Val(:u), grid, velocities),
     v = VolumeFluxField(Val(:v), grid, velocities))

@inline spherical_shell_volume_flux_velocities(grid::SphericalShellGrid, velocities) =
    (u = VolumeFluxField(Val(:u), grid, velocities),
     v = VolumeFluxField(Val(:v), grid, velocities),
     w = w_velocity(velocities))

@inline function Base.getindex(F::VolumeFluxField{Val{:u}}, i, j, k)
    u = u_velocity(F.velocities)
    v = v_velocity(F.velocities)
    return covariant_to_volume_flux_uᶠᶜᶜ(i, j, k, F.grid, u, v)
end

@inline function Base.getindex(F::VolumeFluxField{Val{:v}}, i, j, k)
    u = u_velocity(F.velocities)
    v = v_velocity(F.velocities)
    return covariant_to_volume_flux_vᶜᶠᶜ(i, j, k, F.grid, u, v)
end

struct DecreasingOrderAdvectionScheme end

include("centered_advective_fluxes.jl")
include("upwind_biased_advective_fluxes.jl")

include("reconstruction_coefficients.jl")
include("centered_reconstruction.jl")
include("upwind_biased_reconstruction.jl")
include("weno_reconstruction.jl")
include("weno_interpolants.jl")
include("stretched_weno_smoothness.jl")
include("multi_dimensional_reconstruction.jl")
include("vector_invariant_upwinding.jl")
include("vector_invariant_advection.jl")
include("vector_invariant_self_upwinding.jl")
include("vector_invariant_cross_upwinding.jl")
include("flux_form_advection.jl")

include("topologically_conditional_interpolation.jl")
include("flat_advective_fluxes.jl")
include("immersed_advective_fluxes.jl")
include("momentum_advection_operators.jl")
include("curvature_metric_terms.jl")
include("tracer_advection_operators.jl")
include("bounds_preserving_tracer_advection_operators.jl")
include("cell_advection_timescale.jl")
include("adapt_advection_order.jl")
include("materialize_advection.jl")

end # module
