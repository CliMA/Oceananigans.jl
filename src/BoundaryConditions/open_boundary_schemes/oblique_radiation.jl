#####
##### ObliqueRadiation (Raymond & Kuo 1984) open boundary scheme
#####

"""
    ObliqueRadiation(; inflow_timescale = 0,
                       outflow_timescale = Inf,
                       phase_speed_weight = 0.3,
                       target_transport = nothing)

Raymond & Kuo (1984) two-dimensional radiation condition with nudging to the exterior value
(Marchesiello et al. 2001). With `φ₁`, `φ₂` the first and second interior values next to the
boundary, the normal and tangential phase-speed coefficients are estimated from the interior as

    rₙ = min(∂ₜφ ∂ₙφ, c),   rₜ = clamp(∂ₜφ ∂ₜ̂φ, -c, c),   c = max((∂ₙφ)² + (∂ₜ̂φ)², ε)

with `∂ₜφ = φ₁ⁿ - φ₁ⁿ⁺¹`, `∂ₙφ = φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹` pointing out of the domain, and `∂ₜ̂φ` the upwinded
difference along the boundary. A phase speed pointing into the domain (`∂ₜφ ∂ₙφ < 0`) is set to
zero: waves are only radiated out. The coefficients `rₙ`, `rₜ` and `c` are averaged in time,

    r̄ = (1 - w) r̄ + w r,    w = phase_speed_weight,

once per time step (`phase_speed_weight = 1` turns the averaging off), and the boundary value is

    φᵇ = (c̄ φᵇ + r̄ₙ φ₁ⁿ⁺¹ - max(r̄ₜ, 0) δ₋φᵇ - min(r̄ₜ, 0) δ₊φᵇ) / (c̄ + r̄ₙ)

where `δ∓φᵇ` are the differences of the boundary values along the boundary. It is then nudged
toward the exterior value `φᵉˣᵗ`,

    φᵇ = (1 - Δt / (τ + Δt)) φᵇ + Δt / (τ + Δt) φᵉˣᵗ,

with `τ = inflow_timescale` when the phase speed points into the domain or vanishes, and
`τ = outflow_timescale` otherwise; `inflow_timescale = 0` imposes `φᵉˣᵗ` on inflow. The tangential
term is applied on the lateral boundaries only. `target_transport` pins the net transport through a
`NormalFlowBoundaryCondition`, as for [`NormalRadiation`](@ref).

On the velocity component tangential to a lateral boundary, the coefficients are those of the
boundary-normal velocity at the two adjacent normal-velocity points, averaged, and the tangential
velocity is nudged with `inflow_timescale` only while the averaged normal coefficient `r̄ₙ` is zero. With `phase_speed_weight = 1` the tangential
velocity uses its own phase speed, like any other field.

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
├── phase_speed_weight: 0.3
└── target_transport: Nothing
```
"""
struct ObliqueRadiation{FT, S, B, TF} <: AbstractRadiationScheme{FT}
    outflow_timescale  :: FT
    inflow_timescale   :: FT
    phase_speed_weight :: FT
    φᵇ  :: S
    φ₁  :: S
    φ₁ˡ :: S
    previous_boundary :: B # boundary values written during the previous iteration, double-buffered by iteration parity
    previous_interior :: B # first-interior values, likewise
    rₙ :: B                # time-averaged coefficients: [.., .., 1] at the start of the step, [.., .., 2] latest
    rₜ :: B
    c  :: B
    normal_interior :: B   # tangential velocity only: the two adjacent normal-velocity interior values,
                           # [.., .., 1:2] start-of-step and latest on one side, [.., .., 3:4] on the other
    target_transport :: TF # prescribed net transport through the boundary, or nothing
end

function ObliqueRadiation(FT = defaults.FloatType;
                          inflow_timescale = 0,
                          outflow_timescale = Inf,
                          phase_speed_weight = 0.3,
                          target_transport = nothing)

    0 < phase_speed_weight ≤ 1 || throw(ArgumentError("phase_speed_weight must be in (0, 1], got $phase_speed_weight"))
    outflow_timescale = convert(FT, outflow_timescale)
    inflow_timescale = convert(FT, inflow_timescale)
    phase_speed_weight = convert(FT, phase_speed_weight)
    target_transport = convert_target_transport(FT, target_transport)
    return ObliqueRadiation(outflow_timescale, inflow_timescale, phase_speed_weight,
                            nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, target_transport)
end

Adapt.adapt_structure(to, r::ObliqueRadiation) =
    ObliqueRadiation(adapt(to, r.outflow_timescale),
                     adapt(to, r.inflow_timescale),
                     adapt(to, r.phase_speed_weight),
                     adapt(to, r.φᵇ),
                     adapt(to, r.φ₁),
                     adapt(to, r.φ₁ˡ),
                     adapt(to, r.previous_boundary),
                     adapt(to, r.previous_interior),
                     adapt(to, r.rₙ),
                     adapt(to, r.rₜ),
                     adapt(to, r.c),
                     adapt(to, r.normal_interior),
                     adapt(to, r.target_transport))

function Base.show(io::IO, r::ObliqueRadiation)
    print(io, summary(r), '\n')
    print(io, "├── inflow_timescale: ",  prettysummary(r.inflow_timescale), '\n')
    print(io, "├── outflow_timescale: ", prettysummary(r.outflow_timescale), '\n')
    print(io, "├── phase_speed_weight: ", prettysummary(r.phase_speed_weight), '\n')
    print(io, "└── target_transport: ", prettysummary(r.target_transport))
end

# Inflow and outflow are distinguished by the direction of the phase speed, not by the velocity.
@inline uses_boundary_velocity(::ObliqueRadiation) = false

has_target_transport(::ObliqueRadiation{<:Any, <:Any, <:Any, <:Nothing}) = false
has_target_transport(::ObliqueRadiation) = true

radiation_buffers(radiation::ObliqueRadiation, arch, FT, tangential_size) =
    (ntuple(_ -> zeros(arch, FT, tangential_size...), 3)...,
     ntuple(_ -> zeros(arch, FT, tangential_size..., 2), 5)...,
     zeros(arch, FT, tangential_size..., 4))

radiation_buffers(radiation::ObliqueRadiation) =
    (radiation.φᵇ, radiation.φ₁, radiation.φ₁ˡ, radiation.previous_boundary, radiation.previous_interior,
     radiation.rₙ, radiation.rₜ, radiation.c, radiation.normal_interior)

radiation_storage(radiation::ObliqueRadiation, (φᵇ, φ₁, φ₁ˡ, previous_boundary, previous_interior, rₙ, rₜ, c, normal_interior)) =
    ObliqueRadiation(radiation.outflow_timescale, radiation.inflow_timescale, radiation.phase_speed_weight,
                     φᵇ, φ₁, φ₁ˡ, previous_boundary, previous_interior, rₙ, rₜ, c, normal_interior, radiation.target_transport)

# Fills read the buffer written during the previous iteration and write the other one.
@inline written_buffer(clock) = clock.iteration % 2 + 1
@inline written_buffer(::Nothing) = 1

# Backward and forward differences along the boundary face, zero beyond its ends.
@inline function tangential_differences(φ, t, k, b)
    T = size(φ, 1)
    @inbounds begin
        φ₀ = φ[t, k, b]
        φ₋ = φ[max(t - 1, 1), k, b]
        φ₊ = φ[min(t + 1, T), k, b]
    end
    return φ₀ - φ₋, φ₊ - φ₀
end

# The phase-speed coefficients (rₙ, rₜ, c) estimated from the interior at this fill, and whether the phase
# speed points out of the domain.
@inline function oblique_phase_speeds(φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, δ₁₋, δ₁₊)
    ∂t_φ = φ₁ⁿ - φ₁ⁿ⁺¹
    ∂n_φ = φ₁ⁿ⁺¹ - φ₂ⁿ⁺¹
    s = ∂t_φ * (δ₁₋ + δ₁₊)
    FT = typeof(∂n_φ)
    ∂t̂_φ = ifelse(s > 0, δ₁₋, ifelse(s == 0, zero(FT), δ₁₊))
    radiating = ∂t_φ * ∂n_φ > 0
    ∂t_φ = ifelse(∂t_φ * ∂n_φ < 0, zero(FT), ∂t_φ)

    c  = max(∂n_φ^2 + ∂t̂_φ^2, eps(FT))
    rₙ = min(∂t_φ * ∂n_φ, c)
    rₜ = clamp(∂t_φ * ∂t̂_φ, -c, c)
    return rₙ, rₜ, c, radiating
end

# Radiate the boundary value with the time-averaged coefficients, then nudge it toward the exterior value.
@inline function oblique_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, δᵇ₋, δᵇ₊, r̄ₙ, r̄ₜ, c̄, radiating, φᵉˣᵗ, Δt, radiation)
    FT = typeof(φᵇⁿ)
    φʳ = (c̄ * φᵇⁿ + r̄ₙ * φ₁ⁿ⁺¹ - max(r̄ₜ, zero(FT)) * δᵇ₋ - min(r̄ₜ, zero(FT)) * δᵇ₊) / (c̄ + r̄ₙ)

    τ = ifelse(radiating, radiation.outflow_timescale, radiation.inflow_timescale)
    γ = Δt / (τ + Δt)
    φᵇⁿ⁺¹ = (1 - γ) * φʳ + γ * φᵉˣᵗ

    return ifelse(τ == 0, φᵉˣᵗ, φᵇⁿ⁺¹)
end

# `outflow` and `Cᵃ` are unused: inflow and outflow are distinguished by the direction of the phase speed.
@inline function radiation_update(radiation::ObliqueRadiation, t, k, clock, φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, outflow, Cᵃ)
    w = written_buffer(clock)
    r = 3 - w
    δᵇ₋, δᵇ₊ = tangential_differences(radiation.previous_boundary, t, k, r)
    δ₁₋, δ₁₊ = tangential_differences(radiation.previous_interior, t, k, r)
    rₙ, rₜ, c, radiating = oblique_phase_speeds(φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, δ₁₋, δ₁₊)

    r̄ₙ, r̄ₜ, c̄ = average_phase_speeds!(radiation, t, k, clock, rₙ, rₜ, c)
    φᵇⁿ⁺¹ = oblique_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, δᵇ₋, δᵇ₊, r̄ₙ, r̄ₜ, c̄, radiating, φᵉˣᵗ, Δt, radiation)

    @inbounds begin
        radiation.previous_boundary[t, k, w] = φᵇⁿ⁺¹
        radiation.previous_interior[t, k, w] = φ₁ⁿ⁺¹
    end

    return φᵇⁿ⁺¹
end

# The averages advance once per time step: the first fill of a step promotes the latest average to the
# start-of-step value, and every fill of the step averages from that start-of-step value.
@inline function average_phase_speeds!(radiation, t, k, clock, rₙ, rₜ, c)
    anchored = anchored_fill(clock)
    ω = radiation.phase_speed_weight
    @inbounds begin
        r̄ₙ⁰ = ifelse(anchored, radiation.rₙ[t, k, 2], radiation.rₙ[t, k, 1])
        r̄ₜ⁰ = ifelse(anchored, radiation.rₜ[t, k, 2], radiation.rₜ[t, k, 1])
        c̄⁰  = ifelse(anchored, radiation.c[t, k, 2],  radiation.c[t, k, 1])
    end
    r̄ₙ = (1 - ω) * r̄ₙ⁰ + ω * rₙ
    r̄ₜ = (1 - ω) * r̄ₜ⁰ + ω * rₜ
    c̄  = (1 - ω) * c̄⁰  + ω * c
    @inbounds begin
        radiation.rₙ[t, k, 1] = r̄ₙ⁰
        radiation.rₜ[t, k, 1] = r̄ₜ⁰
        radiation.c[t, k, 1]  = c̄⁰
        radiation.rₙ[t, k, 2] = r̄ₙ
        radiation.rₜ[t, k, 2] = r̄ₜ
        radiation.c[t, k, 2]  = c̄
    end
    return r̄ₙ, r̄ₜ, c̄
end

#####
##### Tangential velocity: phase speeds of the boundary-normal velocity
#####

# The start-of-step value of a normal-velocity interior point, kept in slots (s, s + 1) of `normal_interior`.
@inline function previous_normal_interior!(radiation, t, k, s, uⁿ⁺¹, first_call, anchored)
    @inbounds begin
        uᵃ = ifelse(anchored, radiation.normal_interior[t, k, s + 1], radiation.normal_interior[t, k, s])
        uⁿ = ifelse(first_call, uⁿ⁺¹, uᵃ)
        radiation.normal_interior[t, k, s]     = uⁿ
        radiation.normal_interior[t, k, s + 1] = uⁿ⁺¹
    end
    return uⁿ
end

# Radiate a tangential velocity with the averaged coefficients of the normal velocity at its two neighbours,
# `(u₁, u₂, δ₋, δ₊)` on each side: first and second interior values, and differences along the boundary.
@inline function tangential_radiation_update(radiation, t, k, clock, φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt,
                                             (u₁₋, u₂₋, δ₋₋, δ₋₊), (u₁₊, u₂₊, δ₊₋, δ₊₊))
    radiation.phase_speed_weight == 1 &&
        return radiation_update(radiation, t, k, clock, φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, nothing, nothing)

    first_call = isinf(stage_Δt(clock))
    anchored = anchored_fill(clock)
    u₁₋ⁿ = previous_normal_interior!(radiation, t, k, 1, u₁₋, first_call, anchored)
    u₁₊ⁿ = previous_normal_interior!(radiation, t, k, 3, u₁₊, first_call, anchored)
    rₙ₋, rₜ₋, c₋, _ = oblique_phase_speeds(u₁₋, u₂₋, u₁₋ⁿ, δ₋₋, δ₋₊)
    rₙ₊, rₜ₊, c₊, _ = oblique_phase_speeds(u₁₊, u₂₊, u₁₊ⁿ, δ₊₋, δ₊₊)
    r̄ₙ, r̄ₜ, c̄ = average_phase_speeds!(radiation, t, k, clock, (rₙ₋ + rₙ₊) / 2, (rₜ₋ + rₜ₊) / 2, (c₋ + c₊) / 2)

    w = written_buffer(clock)
    δᵇ₋, δᵇ₊ = tangential_differences(radiation.previous_boundary, t, k, 3 - w)
    φᵇⁿ⁺¹ = oblique_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, δᵇ₋, δᵇ₊, r̄ₙ, r̄ₜ, c̄, r̄ₙ > 0, φᵉˣᵗ, Δt, radiation)

    @inbounds begin
        radiation.previous_boundary[t, k, w] = φᵇⁿ⁺¹
        radiation.previous_interior[t, k, w] = φ₁ⁿ⁺¹
    end

    return φᵇⁿ⁺¹
end

# First and second interior values of a normal velocity next to the boundary, and its differences along the
# boundary (zero beyond the ends).
@inline function normal_neighbour_x(u, iᵢ, iᵢᵢ, j, k, Ny)
    j = clamp(j, 1, Ny)
    @inbounds begin
        u₁ = u[iᵢ, j, k]
        δ₋ = ifelse(j > 1,  u₁ - u[iᵢ, j - 1, k], zero(u₁))
        δ₊ = ifelse(j < Ny, u[iᵢ, j + 1, k] - u₁, zero(u₁))
        return u₁, u[iᵢᵢ, j, k], δ₋, δ₊
    end
end

@inline function normal_neighbour_y(v, jᵢ, jᵢᵢ, i, k, Nx)
    i = clamp(i, 1, Nx)
    @inbounds begin
        v₁ = v[i, jᵢ, k]
        δ₋ = ifelse(i > 1,  v₁ - v[i - 1, jᵢ, k], zero(v₁))
        δ₊ = ifelse(i < Nx, v[i + 1, jᵢ, k] - v₁, zero(v₁))
        return v₁, v[i, jᵢᵢ, k], δ₋, δ₊
    end
end

@inline function radiate_tangential!(c, bᵢ, b₁, b₂, t, k, grid, bc, clock, model_fields, neighbours, closed)
    Δτ = stage_Δt(clock)
    first_call = isinf(Δτ)
    Δt = ifelse(first_call, zero(Δτ), Δτ)
    anchored = anchored_fill(clock)
    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, t, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[b₁...]
        φ₂ⁿ⁺¹ = c[b₂...]
        φᵇᵃ = ifelse(anchored, c[bᵢ...], radiation.φᵇ[t, k])
        φ₁ᵃ = ifelse(anchored, radiation.φ₁ˡ[t, k], radiation.φ₁[t, k])
        φᵇⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φᵇᵃ)
        φ₁ⁿ = ifelse(first_call, φ₁ⁿ⁺¹, φ₁ᵃ)

        φᵇⁿ⁺¹ = tangential_radiation_update(radiation, t, k, clock, φᵇⁿ, φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, φᵉˣᵗ, Δt, neighbours...)
        c[bᵢ...] = ifelse(closed, zero(grid), φᵇⁿ⁺¹)
        radiation.φᵇ[t, k]  = φᵇⁿ
        radiation.φ₁[t, k]  = φ₁ⁿ
        radiation.φ₁ˡ[t, k] = φ₁ⁿ⁺¹
    end

    return nothing
end

const OVBC = BoundaryCondition{<:Value{<:ObliqueRadiation}}
const TangentialToX = Tuple{Center, Face, Center} # v on an east or west boundary
const TangentialToY = Tuple{Face, Center, Center} # u on a north or south boundary

# A face-located tangential velocity point J lies between the normal-velocity points J - 1 and J.
@inline function _fill_east_halo!(j, k, grid, c, bc::OVBC, loc::TangentialToX, clock, model_fields)
    Nx, Ny = grid.Nx, grid.Ny
    u = model_fields.u
    neighbours = (normal_neighbour_x(u, Nx, Nx - 1, j - 1, k, Ny), normal_neighbour_x(u, Nx, Nx - 1, j, k, Ny))
    closed = immersed_peripheral_node(Nx, j, k, grid, Center(), Face(), Center())
    return radiate_tangential!(c, (Nx + 1, j, k), (Nx, j, k), (Nx - 1, j, k), j, k, grid, bc, clock, model_fields, neighbours, closed)
end

@inline function _fill_west_halo!(j, k, grid, c, bc::OVBC, loc::TangentialToX, clock, model_fields)
    Ny = grid.Ny
    u = model_fields.u
    neighbours = (normal_neighbour_x(u, 2, 3, j - 1, k, Ny), normal_neighbour_x(u, 2, 3, j, k, Ny))
    closed = immersed_peripheral_node(1, j, k, grid, Center(), Face(), Center())
    return radiate_tangential!(c, (0, j, k), (1, j, k), (2, j, k), j, k, grid, bc, clock, model_fields, neighbours, closed)
end

@inline function _fill_north_halo!(i, k, grid, c, bc::OVBC, loc::TangentialToY, clock, model_fields)
    Nx, Ny = grid.Nx, grid.Ny
    v = model_fields.v
    neighbours = (normal_neighbour_y(v, Ny, Ny - 1, i - 1, k, Nx), normal_neighbour_y(v, Ny, Ny - 1, i, k, Nx))
    closed = immersed_peripheral_node(i, Ny, k, grid, Face(), Center(), Center())
    return radiate_tangential!(c, (i, Ny + 1, k), (i, Ny, k), (i, Ny - 1, k), i, k, grid, bc, clock, model_fields, neighbours, closed)
end

@inline function _fill_south_halo!(i, k, grid, c, bc::OVBC, loc::TangentialToY, clock, model_fields)
    Nx = grid.Nx
    v = model_fields.v
    neighbours = (normal_neighbour_y(v, 2, 3, i - 1, k, Nx), normal_neighbour_y(v, 2, 3, i, k, Nx))
    closed = immersed_peripheral_node(i, 1, k, grid, Face(), Center(), Center())
    return radiate_tangential!(c, (i, 0, k), (i, 1, k), (i, 2, k), i, k, grid, bc, clock, model_fields, neighbours, closed)
end
