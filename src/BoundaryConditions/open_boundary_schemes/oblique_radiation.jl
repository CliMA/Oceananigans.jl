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
└── target_transport: nothing
```
"""
struct ObliqueRadiation{FT, S, P, N, C, F, B, TF} <: AbstractRadiationScheme{FT}
    outflow_timescale  :: FT
    inflow_timescale   :: FT
    phase_speed_weight :: FT
    φ⁻ :: S # boundary value at the base of the stage (2D array or nothing)
    φ₁ :: S # interior value at the base of the stage (2D array or nothing)
    phase_speeds :: P # time-averaged coefficients (rₙ, rₜ, c): at the start of the step, then the latest of each
    normal_interior :: N # tangential velocity only: the first interior values of the normal velocity at the two
                         # adjacent points, at the base of the stage
    normal_current :: C  # tangential velocity only: the first and second interior values of the normal velocity along
                         # the boundary, with halos that hold those of the neighbouring ranks
    normal_fields :: F   # the fields holding normal_current, whose halos are filled
    tangential_bounds :: B # first and last index along the boundary that the normal velocity is read at
    target_transport :: TF # prescribed net transport through the boundary, or nothing
    periodic :: Bool       # whether the direction along the boundary is periodic on this grid
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
                            nothing, nothing, nothing, nothing, nothing, nothing, nothing, target_transport, false)
end

Adapt.adapt_structure(to, r::ObliqueRadiation) =
    ObliqueRadiation(adapt(to, r.outflow_timescale),
                     adapt(to, r.inflow_timescale),
                     adapt(to, r.phase_speed_weight),
                     adapt(to, r.φ⁻),
                     adapt(to, r.φ₁),
                     adapt(to, r.phase_speeds),
                     adapt(to, r.normal_interior),
                     adapt(to, r.normal_current),
                     nothing,
                     r.tangential_bounds,
                     adapt(to, r.target_transport),
                     r.periodic)

function Base.show(io::IO, r::ObliqueRadiation)
    print(io, summary(r), '\n')
    print(io, "├── inflow_timescale: ",  prettysummary(r.inflow_timescale), '\n')
    print(io, "├── outflow_timescale: ", prettysummary(r.outflow_timescale), '\n')
    print(io, "├── phase_speed_weight: ", prettysummary(r.phase_speed_weight), '\n')
    print(io, "└── target_transport: ", prettysummary(r.target_transport))
end

has_target_transport(radiation::ObliqueRadiation) = !isnothing(radiation.target_transport)

radiation_storage(radiation::ObliqueRadiation, buffer) =
    ObliqueRadiation(radiation.outflow_timescale, radiation.inflow_timescale, radiation.phase_speed_weight,
                     buffer(), buffer(), ntuple(_ -> buffer(), 6), ntuple(_ -> buffer(), 2),
                     radiation.normal_current, radiation.normal_fields, radiation.tangential_bounds,
                     radiation.target_transport, radiation.periodic)

# The buffers span the Connected halos along the boundary, but a periodic direction that is not partitioned wraps
# onto itself.
#
# A tangential velocity also reads the normal velocity along the boundary at its two adjacent points. The current
# values are kept in fields reduced normal to the boundary, whose halos along it hold the values of a neighbouring
# rank or of the other end of a periodic boundary.
function materialize_radiation_storage(radiation::ObliqueRadiation, grid, loc, dim)
    scheme = invoke(materialize_radiation_storage, Tuple{AbstractRadiationScheme, Any, Any, Any}, radiation, grid, loc, dim)
    d = ifelse(dim == 1, 2, 1)
    T = topology(grid, d)

    tangential = loc[dim] isa Center && loc[d] isa Face
    normal_fields = nothing
    normal_current = nothing
    bounds = nothing

    if tangential
        normal_location = dim == 1 ? (Face(), Center(), Center()) : (Center(), Face(), Center())
        normal_fields = ntuple(_ -> boundary_state_field(grid, normal_location, dim), 2)
        normal_current = map(f -> along_boundary(f, dim), normal_fields)

        N = ifelse(d == 1, grid.Nx, grid.Ny)
        H = halo_size(grid, d)
        bounds = (ifelse(has_left_neighbour(T), 1 - H, 1), ifelse(has_right_neighbour(T), N + H, N))
    end

    return ObliqueRadiation(scheme.outflow_timescale, scheme.inflow_timescale, scheme.phase_speed_weight,
                            scheme.φ⁻, scheme.φ₁, scheme.phase_speeds, scheme.normal_interior,
                            normal_current, normal_fields, bounds, scheme.target_transport, T === Grids.Periodic)
end

const OblBC = Union{BoundaryCondition{<:Value{<:ObliqueRadiation}}, BoundaryCondition{<:NormalFlow{<:ObliqueRadiation}}}

# The averaged coefficients of the latest step are the only state that outlives it
latest_phase_speeds(radiation::ObliqueRadiation) = radiation.phase_speeds[4:6]

Oceananigans.prognostic_state(bc::OblBC) = map(b -> on_architecture(CPU(), b), latest_phase_speeds(bc.classification.scheme))

function Oceananigans.restore_prognostic_state!(bc::OblBC, from)
    latest = latest_phase_speeds(bc.classification.scheme)
    length(from) == length(latest) && foreach(copyto!, latest, from)
    return bc
end

Oceananigans.restore_prognostic_state!(::OblBC, ::Nothing) = nothing

# Backward and forward differences along the boundary face. Beyond its ends they are zero, or take the value from the
# other end if the direction is periodic.
@inline function tangential_differences(φ, i, j, periodic)
    lower = firstindex(φ, 1)
    upper = lastindex(φ, 1)
    i₋ = ifelse(periodic, ifelse(i == lower, upper, i - 1), max(i - 1, lower))
    i₊ = ifelse(periodic, ifelse(i == upper, lower, i + 1), min(i + 1, upper))
    @inbounds begin
        φ₀ = φ[i, j]
        φ₋ = φ[i₋, j]
        φ₊ = φ[i₊, j]
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

# The averages advance once per time step: the first stage of a step promotes the latest averages to the start-of-step
# values, at the base of the stage, and every stage of the step averages from the start-of-step values.
@inline first_stage(clock) = clock.stage ≤ 1
@inline first_stage(::Nothing) = true

@inline function cache_exterior_state!(side, i, j, grid, c, bc::OblBC, loc, clock, model_fields)
    iᵇ, jᵇ, kᵇ = boundary_point(side, loc, grid, 0, i, j)
    i₁, j₁, k₁ = boundary_point(side, loc, grid, 1, i, j)
    radiation = bc.classification.scheme
    rₙ⁰, rₜ⁰, c⁰, rₙ, rₜ, cˡ = radiation.phase_speeds
    promote = first_stage(clock)

    @inbounds begin
        radiation.φ⁻[i, j] = c[iᵇ, jᵇ, kᵇ]
        radiation.φ₁[i, j] = c[i₁, j₁, k₁]
        rₙ⁰[i, j] = ifelse(promote, rₙ[i, j], rₙ⁰[i, j])
        rₜ⁰[i, j] = ifelse(promote, rₜ[i, j], rₜ⁰[i, j])
        c⁰[i, j]  = ifelse(promote, cˡ[i, j], c⁰[i, j])
    end

    cache_normal_interior!(side, i, j, grid, radiation, loc, model_fields)

    return nothing
end

@inline cache_normal_interior!(side, i, j, grid, radiation, loc, model_fields) = nothing

# Time-average the coefficients from the start-of-step values, keeping the result as the latest
@inline function average_phase_speeds!(radiation, i, j, rₙ, rₜ, c)
    ω = radiation.phase_speed_weight
    rₙ⁰, rₜ⁰, c⁰, rₙˡ, rₜˡ, cˡ = radiation.phase_speeds

    @inbounds begin
        r̄ₙ = (1 - ω) * rₙ⁰[i, j] + ω * rₙ
        r̄ₜ = (1 - ω) * rₜ⁰[i, j] + ω * rₜ
        c̄  = (1 - ω) * c⁰[i, j]  + ω * c
        rₙˡ[i, j] = r̄ₙ
        rₜˡ[i, j] = r̄ₜ
        cˡ[i, j]  = c̄
    end

    return r̄ₙ, r̄ₜ, c̄
end

@inline step_exterior_state!(side, i, j, grid, c, bc::OblBC, loc, clock, model_fields, Δt) =
    step_oblique_exterior_state!(side, i, j, grid, c, bc, loc, clock, model_fields, Δt)

@inline function step_oblique_exterior_state!(side, i, j, grid, c, bc, loc, clock, model_fields, Δt)
    iᵇ, jᵇ, kᵇ = boundary_point(side, loc, grid, 0, i, j)
    i₁, j₁, k₁ = boundary_point(side, loc, grid, 1, i, j)
    i₂, j₂, k₂ = boundary_point(side, loc, grid, 2, i, j)
    radiation = bc.classification.scheme

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, i, j, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i₁, j₁, k₁]
        φ₂ⁿ⁺¹ = c[i₂, j₂, k₂]

        φᵇⁿ = radiation.φ⁻[i, j]
        φ₁ⁿ = radiation.φ₁[i, j]
        δᵇ₋, δᵇ₊ = tangential_differences(radiation.φ⁻, i, j, radiation.periodic)
        δ₁₋, δ₁₊ = tangential_differences(radiation.φ₁, i, j, radiation.periodic)

        rₙ, rₜ, cᵖ, radiating = oblique_phase_speeds(φ₁ⁿ⁺¹, φ₂ⁿ⁺¹, φ₁ⁿ, δ₁₋, δ₁₊)
        r̄ₙ, r̄ₜ, c̄ = average_phase_speeds!(radiation, i, j, rₙ, rₜ, cᵖ)

        φᵇⁿ⁺¹ = oblique_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, δᵇ₋, δᵇ₊, r̄ₙ, r̄ₜ, c̄, radiating, φᵉˣᵗ, Δt, radiation)
        c[iᵇ, jᵇ, kᵇ] = ifelse(immersed_boundary_point(side, loc, grid, i, j), zero(grid), φᵇⁿ⁺¹)
    end

    return nothing
end

#####
##### Tangential velocity: phase speeds of the boundary-normal velocity
#####

const TangentialToX = Tuple{Center, Face, Center} # v on a west or east boundary
const TangentialToY = Tuple{Face, Center, Center} # u on a south or north boundary

# A face-located tangential velocity point `t` lies between the normal-velocity points `t - 1` and `t`. Beyond the ends
# of the boundary the neighbour is the point itself.
@inline normal_neighbours(t, (lower, upper)) = (clamp(t - 1, lower, upper), clamp(t, lower, upper))

# The normal velocity and the length of the boundary it varies along
@inline normal_component(::Union{West, East},   model_fields) = model_fields.u
@inline normal_component(::Union{South, North}, model_fields) = model_fields.v
@inline boundary_length(::Union{West, East},   grid) = grid.Ny
@inline boundary_length(::Union{South, North}, grid) = grid.Nx

# Index of the normal velocity at plane point `t`, `n` faces into the interior
@inline normal_velocity_index(side::Union{West, East},   grid, n, t, k) = (normal_index(side, Face(), grid.Nx, n), t, k)
@inline normal_velocity_index(side::Union{South, North}, grid, n, t, k) = (t, normal_index(side, Face(), grid.Ny, n), k)

const WestEast = Union{West, East}
const SouthNorth = Union{South, North}

@inline cache_normal_interior!(side::WestEast,   t, k, grid, radiation, loc::TangentialToX, model_fields) =
    cache_tangential_normal_interior!(side, t, k, grid, radiation, model_fields)

@inline cache_normal_interior!(side::SouthNorth, t, k, grid, radiation, loc::TangentialToY, model_fields) =
    cache_tangential_normal_interior!(side, t, k, grid, radiation, model_fields)

@inline function cache_tangential_normal_interior!(side, t, k, grid, radiation, model_fields)
    uₙ = normal_component(side, model_fields)
    t₋, t₊ = normal_neighbours(t, radiation.tangential_bounds)
    @inbounds begin
        radiation.normal_interior[1][t, k] = uₙ[normal_velocity_index(side, grid, 1, t₋, k)...]
        radiation.normal_interior[2][t, k] = uₙ[normal_velocity_index(side, grid, 1, t₊, k)...]
    end
    return nothing
end

# First and second interior values of the normal velocity at plane point `t`, and its differences along the
# boundary (zero beyond the ends)
@inline function normal_neighbour(radiation, t, k)
    f₁, f₂ = radiation.normal_current
    lower, upper = radiation.tangential_bounds
    @inbounds begin
        u₁ = f₁[t, k]
        u₂ = f₂[t, k]
        u₋ = f₁[clamp(t - 1, lower, upper), k]
        u₊ = f₁[clamp(t + 1, lower, upper), k]
    end
    return u₁, u₂, u₁ - u₋, u₊ - u₁
end

# The current normal-velocity values along the boundary, with the halos filled, are read when the tangential velocity
# is stepped
@kernel function _copy_normal_interior!(f₁, f₂, uₙ, side, grid)
    t, k = @index(Global, NTuple)
    @inbounds begin
        f₁[t, k] = uₙ[normal_velocity_index(side, grid, 1, t, k)...]
        f₂[t, k] = uₙ[normal_velocity_index(side, grid, 2, t, k)...]
    end
end

step_boundary_condition!(bc::OblBC, side, field, clock, model_fields, Δt) =
    (prepare_normal_interior!(side, bc.classification.scheme, field.grid, model_fields);
     update_exterior_state!(step_exterior_state!, bc, side, field, clock, model_fields, Δt))

prepare_normal_interior!(side, radiation, grid, model_fields) =
    prepare_normal_interior!(side, radiation, radiation.normal_fields, grid, model_fields)

prepare_normal_interior!(side, radiation, ::Nothing, grid, model_fields) = nothing

function prepare_normal_interior!(side, radiation, fields, grid, model_fields)
    f₁, f₂ = radiation.normal_current
    uₙ = normal_component(side, model_fields)
    parameters = kernel_parameters((boundary_length(side, grid), grid.Nz), (0, 0))
    launch!(architecture(grid), grid, parameters, _copy_normal_interior!, f₁, f₂, uₙ, side, grid)
    foreach(fill_halo_regions!, fields)
    return nothing
end

# Radiate a tangential velocity with the averaged coefficients of the normal velocity at its two neighbours.
# With `phase_speed_weight = 1` it uses its own phase speed, like any other field.
@inline step_exterior_state!(side::WestEast,   t, k, grid, c, bc::OblBC, loc::TangentialToX, clock, model_fields, Δt) =
    step_tangential_exterior_state!(side, t, k, grid, c, bc, loc, clock, model_fields, Δt)

@inline step_exterior_state!(side::SouthNorth, t, k, grid, c, bc::OblBC, loc::TangentialToY, clock, model_fields, Δt) =
    step_tangential_exterior_state!(side, t, k, grid, c, bc, loc, clock, model_fields, Δt)

@inline function step_tangential_exterior_state!(side, t, k, grid, c, bc, loc, clock, model_fields, Δt)
    radiation = bc.classification.scheme

    radiation.phase_speed_weight == 1 &&
        return step_oblique_exterior_state!(side, t, k, grid, c, bc, loc, clock, model_fields, Δt)

    iᵇ, jᵇ, kᵇ = boundary_point(side, loc, grid, 0, t, k)
    i₁, j₁, k₁ = boundary_point(side, loc, grid, 1, t, k)
    t₋, t₊ = normal_neighbours(t, radiation.tangential_bounds)

    @inbounds begin
        φᵉˣᵗ  = getbc(bc, t, k, grid, clock, model_fields)
        φ₁ⁿ⁺¹ = c[i₁, j₁, k₁]
        φᵇⁿ = radiation.φ⁻[t, k]
        δᵇ₋, δᵇ₊ = tangential_differences(radiation.φ⁻, t, k, radiation.periodic)

        u₁₋, u₂₋, δ₋₋, δ₋₊ = normal_neighbour(radiation, t₋, k)
        u₁₊, u₂₊, δ₊₋, δ₊₊ = normal_neighbour(radiation, t₊, k)
        u₁₋ⁿ = radiation.normal_interior[1][t, k]
        u₁₊ⁿ = radiation.normal_interior[2][t, k]

        rₙ₋, rₜ₋, c₋, _ = oblique_phase_speeds(u₁₋, u₂₋, u₁₋ⁿ, δ₋₋, δ₋₊)
        rₙ₊, rₜ₊, c₊, _ = oblique_phase_speeds(u₁₊, u₂₊, u₁₊ⁿ, δ₊₋, δ₊₊)
        r̄ₙ, r̄ₜ, c̄ = average_phase_speeds!(radiation, t, k, (rₙ₋ + rₙ₊) / 2, (rₜ₋ + rₜ₊) / 2, (c₋ + c₊) / 2)

        φᵇⁿ⁺¹ = oblique_radiation_update(φᵇⁿ, φ₁ⁿ⁺¹, δᵇ₋, δᵇ₊, r̄ₙ, r̄ₜ, c̄, r̄ₙ > 0, φᵉˣᵗ, Δt, radiation)
        c[iᵇ, jᵇ, kᵇ] = ifelse(immersed_boundary_point(side, loc, grid, t, k), zero(grid), φᵇⁿ⁺¹)
    end

    return nothing
end
