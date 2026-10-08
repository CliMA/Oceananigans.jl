#####
##### ObliqueRadiation (Raymond & Kuo 1984) open boundary scheme
#####

"""
    ObliqueRadiation(; inflow_timescale = 0,
                       outflow_timescale = Inf,
                       use_boundary_velocity = false,
                       target_transport = nothing)

Raymond & Kuo (1984) two-dimensional radiation condition with adaptive nudging
(Marchesiello et al. 2001):

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
└── target_transport: Nothing
```
"""
struct ObliqueRadiation{FT, S, B, TF, TB, E} <: AbstractRadiationScheme{FT}
    outflow_timescale :: FT
    inflow_timescale  :: FT
    use_boundary_velocity :: Bool
    φᵇ  :: S
    φ₁  :: S
    φ₁ˡ :: S
    previous_boundary :: B # boundary values written during the previous iteration, one array per iteration parity,
                           # with halos along the boundary that hold the values of neighbouring ranks
    previous_interior :: B # first-interior values, likewise
    target_transport :: TF # prescribed net transport through the boundary, or nothing
    tangential_bounds :: TB # first and last index along the boundary that the tangential differences read
    state_fields :: E      # the fields holding previous_boundary and previous_interior, whose halos are filled
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
                            nothing, nothing, nothing, nothing, nothing, target_transport, nothing, nothing)
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
                     adapt(to, r.target_transport),
                     r.tangential_bounds,
                     nothing)

has_target_transport(::ObliqueRadiation{<:Any, <:Any, <:Any, <:Nothing}) = false
has_target_transport(::ObliqueRadiation) = true

# The previous boundary and first-interior values are kept in fields reduced normal to the boundary, whose halos along
# it hold the values of a neighbouring rank or of the other end of a periodic boundary.
function materialize_radiation_storage(radiation::ObliqueRadiation, grid, loc, dim)
    FT = eltype(grid)
    arch = architecture(grid)
    Sx, Sy, Sz = size(grid, loc)

    tangential_size = dim == 1 ? (Sy, Sz) :
                      dim == 2 ? (Sx, Sz) :
                                 (Sx, Sy)

    φᵇ, φ₁, φ₁ˡ = ntuple(_ -> zeros(arch, FT, tangential_size...), 3)

    state_fields = ntuple(_ -> boundary_state_field(grid, loc, dim), 4)
    previous_boundary = map(f -> along_boundary(f, dim), state_fields[1:2])
    previous_interior = map(f -> along_boundary(f, dim), state_fields[3:4])

    T = topology(grid, dim == 1 ? 2 : 1)
    N = tangential_size[1]
    tangential_bounds = (ifelse(neighbour_on_left(T), 0, 1), ifelse(neighbour_on_right(T), N + 1, N))

    return ObliqueRadiation(radiation.outflow_timescale, radiation.inflow_timescale, radiation.use_boundary_velocity,
                            φᵇ, φ₁, φ₁ˡ, previous_boundary, previous_interior, radiation.target_transport,
                            tangential_bounds, state_fields)
end

radiation_buffers(radiation::ObliqueRadiation) =
    (radiation.φᵇ, radiation.φ₁, radiation.φ₁ˡ, map(f -> parent(f.data), radiation.state_fields)...)

const OBC = BoundaryCondition{<:Union{Value{<:ObliqueRadiation}, NormalFlow{<:ObliqueRadiation}}}

# The halos of the previous values take those of the neighbouring ranks
function fill_boundary_state_halos!(radiation::ObliqueRadiation)
    isnothing(radiation.state_fields) || foreach(fill_halo_regions!, radiation.state_fields)
    return nothing
end

update_boundary_condition!(bc::OBC, side, field, model) = fill_boundary_state_halos!(bc.classification.scheme)

# Fills read the buffer written during the previous iteration and write the other one.
@inline written_buffer(clock) = clock.iteration % 2 + 1
@inline written_buffer(::Nothing) = 1

# Backward and forward differences along the boundary face, zero beyond its ends. Next to a rank edge the neighbour is
# in the halo of φ, which holds the neighbouring rank's value.
@inline function tangential_differences(φ, t, k, (lower, upper))
    @inbounds begin
        φ₀ = φ[t, k]
        φ₋ = φ[max(t - 1, lower), k]
        φ₊ = φ[min(t + 1, upper), k]
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

@inline function radiation_update(radiation::ObliqueRadiation, t, k, clock, φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, outflow, Cᵃ)
    w = written_buffer(clock)
    bounds = radiation.tangential_bounds
    δᵇ₋, δᵇ₊ = tangential_differences(radiation.previous_boundary[3 - w], t, k, bounds)
    δ₁₋, δ₁₊ = tangential_differences(radiation.previous_interior[3 - w], t, k, bounds)
    φᵇⁿ⁺¹ = oblique_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, δᵇ₋, δᵇ₊, δ₁₋, δ₁₊, φᵉˣᵗ, Δt, radiation, outflow, Cᵃ)

    @inbounds begin
        radiation.previous_boundary[w][t, k] = φᵇⁿ⁺¹
        radiation.previous_interior[w][t, k] = φ₁ⁿ⁺¹
    end

    return φᵇⁿ⁺¹
end
