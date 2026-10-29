#####
##### ObliqueRadiation (Raymond & Kuo 1984) open boundary scheme
#####

"""
    ObliqueRadiation(; inflow_timescale = 0,
                       outflow_timescale = Inf,
                       use_boundary_velocity = false,
                       target_transport = nothing)

[Raymond & Kuo (1984)](@cite raymond.kuo1984) two-dimensional radiation condition with adaptive nudging
([Marchesiello et al., 2001](@cite marchesiello.ea2001)):

    ∂φ/∂t + cₙ ∂φ/∂n + cₜ ∂φ/∂t̂ = - (φ - φᵉˣᵗ) / τ

with

    cₙ = -(∂φ/∂t)(∂φ/∂n) / |∇φ|²,   cₜ = -(∂φ/∂t)(∂φ/∂t̂) / |∇φ|²,   |∇φ|² = (∂φ/∂n)² + (∂φ/∂t̂)²

where `n` is the boundary-normal and `t̂` the horizontal boundary-tangential direction. The
tangential derivative is upwinded and `cₜ` is limited to a tangential Courant number of one.
For `∂φ/∂t̂ = 0` the scheme reduces to `NormalRadiation`. The tangential term is applied on the
lateral boundaries only.

Inflow versus outflow is decided from the boundary-normal velocity: on inflow `cₙ = cₜ = 0` and
`τ = inflow_timescale`; on outflow `τ = outflow_timescale`. For `Value` boundary conditions,
`use_boundary_velocity = true` takes that velocity at the boundary face rather than one cell into
the interior. `target_transport` pins the net transport through a `NormalFlowBoundaryCondition`, as for
[`NormalRadiation`](@ref).

References
==========
* Raymond, W. H., & Kuo, H. L. (1984). "A radiation boundary condition for multi-dimensional
  flows." Quarterly Journal of the Royal Meteorological Society, 110(464), 535-551.
* Marchesiello, P., McWilliams, J. C., & Shchepetkin, A. (2001). "Open boundary conditions
  for long-term integration of regional oceanic models." Ocean Modelling, 3(1-2), 1-20.

```jldoctest
using Oceananigans
using Oceananigans.BoundaryConditions: ObliqueRadiation

ObliqueRadiation()

# output
ObliqueRadiation{Float64}
├── inflow_timescale: 0.0
├── outflow_timescale: Inf
├── use_boundary_velocity: false
└── target_transport: nothing
```
"""
struct ObliqueRadiation{FT, S, B, TF} <: AbstractRadiationScheme{FT}
    outflow_timescale :: FT
    inflow_timescale  :: FT
    use_boundary_velocity :: Bool
    φᵇ  :: S
    φ₁  :: S
    φ₁ˡ :: S
    previous_boundary :: B # boundary values written during the previous iteration, double-buffered by iteration parity
    previous_interior :: B # first-interior values, likewise
    target_transport :: TF # prescribed net transport through the boundary, or nothing
end

function ObliqueRadiation(FT = defaults.FloatType;
                          inflow_timescale = 0,
                          outflow_timescale = Inf,
                          use_boundary_velocity = false,
                          target_transport = nothing)

    outflow_timescale = convert(FT, outflow_timescale)
    inflow_timescale = convert(FT, inflow_timescale)
    target_transport = convert_target_transport(FT, target_transport)
    return ObliqueRadiation(outflow_timescale, inflow_timescale, use_boundary_velocity,
                            nothing, nothing, nothing, nothing, nothing, target_transport)
end

Adapt.adapt_structure(to, r::ObliqueRadiation) =
    ObliqueRadiation(adapt(to, r.outflow_timescale),
                     adapt(to, r.inflow_timescale),
                     r.use_boundary_velocity,
                     adapt(to, r.φᵇ),
                     adapt(to, r.φ₁),
                     adapt(to, r.φ₁ˡ),
                     adapt(to, r.previous_boundary),
                     adapt(to, r.previous_interior),
                     adapt(to, r.target_transport))

has_target_transport(::ObliqueRadiation{<:Any, <:Any, <:Any, <:Nothing}) = false
has_target_transport(::ObliqueRadiation) = true

radiation_buffers(radiation::ObliqueRadiation, arch, FT, tangential_ranges) =
    (ntuple(_ -> radiation_buffer(arch, FT, tangential_ranges...), 3)...,
     radiation_buffer(arch, FT, tangential_ranges..., 1:2),
     radiation_buffer(arch, FT, tangential_ranges..., 1:2))

radiation_buffers(radiation::ObliqueRadiation) =
    (radiation.φᵇ, radiation.φ₁, radiation.φ₁ˡ, radiation.previous_boundary, radiation.previous_interior)

radiation_storage(radiation::ObliqueRadiation, (φᵇ, φ₁, φ₁ˡ, previous_boundary, previous_interior)) =
    ObliqueRadiation(radiation.outflow_timescale, radiation.inflow_timescale, radiation.use_boundary_velocity,
                     φᵇ, φ₁, φ₁ˡ, previous_boundary, previous_interior, radiation.target_transport)

# Fills read the buffer written during the previous iteration and write the other one.
@inline written_buffer(clock) = clock.iteration % 2 + 1
@inline written_buffer(::Nothing) = 1

# Backward and forward differences along the boundary face, zero beyond its ends.
@inline function tangential_differences(φ, t, k, b)
    @inbounds begin
        φ₀ = φ[t, k, b]
        φ₋ = φ[max(t - 1, firstindex(φ, 1)), k, b]
        φ₊ = φ[min(t + 1, lastindex(φ, 1)),  k, b]
    end
    return φ₀ - φ₋, φ₊ - φ₀
end

@inline function oblique_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, δᵇ₋, δᵇ₊, δ₁₋, δ₁₊, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)
    ∂t_φ = φ₁ⁿ⁺¹ - φ₁ⁿ
    ∂ξ_φ = φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹
    ∂η_φ = ifelse(-∂t_φ * (δ₁₋ + δ₁₊) > 0, δ₁₋, δ₁₊)

    FT = typeof(∂ξ_φ)
    D  = max(∂ξ_φ^2 + ∂η_φ^2, eps(FT))
    Cᶜ = - ∂t_φ * ∂ξ_φ / D
    Cₙ = ifelse(outflow, max(zero(FT), min(one(FT), max(Cᶜ, Cᵃ))), zero(FT))
    Cₜ = ifelse(outflow, clamp(- ∂t_φ * ∂η_φ / D, -one(FT), one(FT)), zero(FT))

    τ = ifelse(outflow, radiation.outflow_timescale, radiation.inflow_timescale)
    τ̃ = Δt / τ

    φᵇⁿ⁺¹ = (φᵇⁿ + Cₙ * φ₁ⁿ⁺¹ - max(Cₜ, zero(FT)) * δᵇ₋ - min(Cₜ, zero(FT)) * δᵇ₊ + τ̃ * φᵉˣᵗ) / (1 + Cₙ + τ̃)

    return ifelse(τ == 0, φᵉˣᵗ, φᵇⁿ⁺¹)
end

@inline function radiation_update(radiation::ObliqueRadiation, t, k, clock, φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, outflow, Cᵃ, closed)
    w = written_buffer(clock)
    r = 3 - w
    δᵇ₋, δᵇ₊ = tangential_differences(radiation.previous_boundary, t, k, r)
    δ₁₋, δ₁₊ = tangential_differences(radiation.previous_interior, t, k, r)
    φᵇⁿ⁺¹ = oblique_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, δᵇ₋, δᵇ₊, δ₁₋, δ₁₊, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)

    @inbounds begin
        radiation.previous_boundary[t, k, w] = ifelse(closed, zero(φᵇⁿ⁺¹), φᵇⁿ⁺¹)
        radiation.previous_interior[t, k, w] = φ₁ⁿ⁺¹
    end

    return φᵇⁿ⁺¹
end

#####
##### Tangential halos of the buffers on distributed grids
#####

# The fill writes tangential halo entries before the exchange; afterwards `c` holds the neighbors' values there
const ObliqueRadiationBC = BoundaryCondition{<:Union{Value{<:ObliqueRadiation}, NormalFlow{<:ObliqueRadiation}}}

"""
$(TYPEDSIGNATURES)

Copy the boundary and first-interior values of `c` into the tangential halo entries of the
`ObliqueRadiation` buffers written with parity `w` on the lateral sides of `boundary_conditions`.
Sides that `fill_normal_flow_bcs = false` left unfilled are skipped.
"""
function refresh_tangential_halos!(c, boundary_conditions::FieldBoundaryConditions, loc, grid, w, fill_normal_flow_bcs)
    refresh_tangential_halos!(c, boundary_conditions.west,  West(),  loc, grid, w, fill_normal_flow_bcs)
    refresh_tangential_halos!(c, boundary_conditions.east,  East(),  loc, grid, w, fill_normal_flow_bcs)
    refresh_tangential_halos!(c, boundary_conditions.south, South(), loc, grid, w, fill_normal_flow_bcs)
    refresh_tangential_halos!(c, boundary_conditions.north, North(), loc, grid, w, fill_normal_flow_bcs)
    return nothing
end

refresh_tangential_halos!(c, bc, side, loc, grid, w, fill_normal_flow_bcs) = nothing

function refresh_tangential_halos!(c, bc::ObliqueRadiationBC, side, loc, grid, w, fill_normal_flow_bcs)
    fills_halo(bc, fill_normal_flow_bcs) || return nothing

    radiation = bc.classification.scheme
    previous_boundary = radiation.previous_boundary
    previous_interior = radiation.previous_interior

    nᵇ, n₁ = boundary_indices(side, loc, grid)
    St = tangential_size(side, loc, grid)
    Nk = size(previous_boundary, 2)

    for halo in (firstindex(previous_boundary, 1):0, St+1:lastindex(previous_boundary, 1))
        isempty(halo) && continue
        launch!(architecture(grid), grid, KernelParameters(halo, 1:Nk), _refresh_tangential_halos!,
                previous_boundary, previous_interior, c, side, nᵇ, n₁, w)
    end

    return nothing
end

@kernel function _refresh_tangential_halos!(previous_boundary, previous_interior, c, side, nᵇ, n₁, w)
    t, k = @index(Global, NTuple)
    @inbounds begin
        previous_boundary[t, k, w] = normal_line_value(c, side, nᵇ, t, k)
        previous_interior[t, k, w] = normal_line_value(c, side, n₁, t, k)
    end
end

@inline normal_line_value(c, ::Union{West, East},   n, t, k) = @inbounds c[n, t, k]
@inline normal_line_value(c, ::Union{South, North}, n, t, k) = @inbounds c[t, n, k]

# Radiated and first-interior indices, as in the `radiate_*_halo!` kernels
@inline left_boundary_index(::Face)   = 1
@inline left_boundary_index(::Center) = 0

@inline boundary_indices(::West,  loc, grid) = (left_boundary_index(loc[1]), left_boundary_index(loc[1]) + 1)
@inline boundary_indices(::South, loc, grid) = (left_boundary_index(loc[2]), left_boundary_index(loc[2]) + 1)
@inline boundary_indices(::East,  loc, grid) = (grid.Nx + 1, grid.Nx)
@inline boundary_indices(::North, loc, grid) = (grid.Ny + 1, grid.Ny)

@inline tangential_size(::Union{West, East},   loc, grid) = size(grid, loc)[2]
@inline tangential_size(::Union{South, North}, loc, grid) = size(grid, loc)[1]
