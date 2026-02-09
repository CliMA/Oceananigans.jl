using Oceananigans: defaults
using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ

"""
    StevensAdvection{FT, U, T}

Open boundary condition scheme based on Stevens (1990), implementing
radiation conditions for both velocities and tracers.

For velocities, the normal component at the boundary is set to the prescribed
(barotropic) value plus a baroclinic anomaly from the previous timestep
(when storage is provided).

For tracers, the boundary cell is updated via forward Euler using:
    T_boundary = T_ref + Δt * (-(u_adv + c_phase) * ∂T/∂x - gFac * γ * (T_ref - T_prescribed))
where `T_ref` is the value one cell interior, `c_phase` is the Orlanski phase velocity,
`gFac = 1` on inflow and `0` on outflow, and `γ = 1/τ_relax`.
"""
struct StevensAdvection{FT, U, T}
    "Relaxation timescale for tracer nudging on inflow (seconds)"
    relaxation_timescale :: FT
    "Whether to use the Orlanski phase velocity estimation"
    use_phase_velocity :: Bool
    "Whether to use advection by the normal velocity"
    use_advection :: Bool
    "Previous velocity at boundary-adjacent cell (2D array along boundary)"
    previous_velocity :: U
    "Previous tracer values at boundary-adjacent cell (2D array along boundary)"
    previous_tracers :: T
end

"""
    StevensAdvection(FT = defaults.FloatType;
                     relaxation_timescale = Inf,
                     use_phase_velocity = true,
                     use_advection = true)

Create a `StevensAdvection` scheme for use with `OpenBoundaryCondition`.

Implements Stevens (1990) open boundary conditions following the MITgcm
implementation. For velocities, it prescribes the barotropic value and
(optionally) adds a baroclinic anomaly. For tracers, it uses Orlanski
radiation with relaxation toward a prescribed value on inflow.

Keyword Arguments
=================

- `relaxation_timescale`: timescale (seconds) for nudging tracers toward the
  prescribed value on inflow. Default: `Inf` (no relaxation).
- `use_phase_velocity`: whether to estimate Orlanski phase velocity. Default: `true`.
- `use_advection`: whether to include boundary-normal advection. Default: `true`.
"""
function StevensAdvection(FT = defaults.FloatType;
                          relaxation_timescale = Inf,
                          use_phase_velocity = true,
                          use_advection = true)

    relaxation_timescale = convert(FT, relaxation_timescale)
    return StevensAdvection(relaxation_timescale, use_phase_velocity, use_advection,
                            nothing, nothing)
end

Adapt.adapt_structure(to, sa::StevensAdvection) =
    StevensAdvection(adapt(to, sa.relaxation_timescale),
                     sa.use_phase_velocity,
                     sa.use_advection,
                     adapt(to, sa.previous_velocity),
                     adapt(to, sa.previous_tracers))

const SOBC = BoundaryCondition{<:Open{<:StevensAdvection}}

#####
##### update_boundary_condition!: save previous values before halo filling
#####
##### When previous_velocity / previous_tracers arrays are allocated,
##### these hooks save interior values for use in the next timestep.
##### Currently a no-op when storage is `nothing`.
#####

@inline update_boundary_condition!(::SOBC, ::Val{:east},   field, model) = nothing
@inline update_boundary_condition!(::SOBC, ::Val{:west},   field, model) = nothing
@inline update_boundary_condition!(::SOBC, ::Val{:north},  field, model) = nothing
@inline update_boundary_condition!(::SOBC, ::Val{:south},  field, model) = nothing

#####
##### Velocity halo filling (Face-located fields)
#####
##### Sets the boundary velocity to the prescribed value.
##### When previous_velocity storage is available, adds the
##### baroclinic anomaly: u_boundary(k) = ū_prescribed + u'(k).
#####

@inline function _fill_east_halo!(j, k, grid, u, bc::SOBC, ::Tuple{Face, Any, Any}, clock, model_fields)
    i = grid.Nx + 1
    sa = bc.classification.scheme
    ū = getbc(bc, j, k, grid, clock, model_fields)

    if sa.previous_velocity === nothing
        @inbounds u[i, j, k] = ū
    else
        @inbounds u[i, j, k] = ū + sa.previous_velocity[j, k]
    end

    return nothing
end

@inline function _fill_west_halo!(j, k, grid, u, bc::SOBC, ::Tuple{Face, Any, Any}, clock, model_fields)
    sa = bc.classification.scheme
    ū = getbc(bc, j, k, grid, clock, model_fields)

    if sa.previous_velocity === nothing
        @inbounds u[1, j, k] = ū
    else
        @inbounds u[1, j, k] = ū + sa.previous_velocity[j, k]
    end

    return nothing
end

@inline function _fill_north_halo!(i, k, grid, v, bc::SOBC, ::Tuple{Any, Face, Any}, clock, model_fields)
    j = grid.Ny + 1
    sa = bc.classification.scheme
    v̄ = getbc(bc, i, k, grid, clock, model_fields)

    if sa.previous_velocity === nothing
        @inbounds v[i, j, k] = v̄
    else
        @inbounds v[i, j, k] = v̄ + sa.previous_velocity[i, k]
    end

    return nothing
end

@inline function _fill_south_halo!(i, k, grid, v, bc::SOBC, ::Tuple{Any, Face, Any}, clock, model_fields)
    sa = bc.classification.scheme
    v̄ = getbc(bc, i, k, grid, clock, model_fields)

    if sa.previous_velocity === nothing
        @inbounds v[i, 1, k] = v̄
    else
        @inbounds v[i, 1, k] = v̄ + sa.previous_velocity[i, k]
    end

    return nothing
end

#####
##### Tracer halo filling (Center-located fields)
#####
##### Following MITgcm obcs_calc_stevens.F:
#####   T_boundary = T_ref + Δt * (
#####       -(u_adv + c_phase) * ΔT_space / Δx     [east/north: outward radiation]
#####        (u_adv + c_phase) * ΔT_space / Δx      [west/south: outward radiation]
#####       - gFac * γ * (T_ref - T_prescribed)      [inflow relaxation]
#####   )
#####
##### where T_ref is one cell interior from boundary, ΔT_space is the
##### spatial gradient at the interior, and c_phase is the Orlanski phase speed.
#####

# East boundary: boundary cell = Nx, halo = Nx+1
# Reference cell = Nx-1, interior cell = Nx-2
# Face velocity between ref and boundary = u[Nx]
@inline function _fill_east_halo!(j, k, grid, c, bc::SOBC, ::Tuple{Center, Any, Any}, clock, model_fields)
    Nx = grid.Nx
    sa = bc.classification.scheme

    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), zero(Δt), Δt)

    Δx = Δxᶠᶜᶜ(Nx, j, k, grid)
    FT = eltype(grid)
    cfl = ifelse(Δt > 0, Δx / (2 * Δt), zero(FT))

    # Spatial gradient (unnormalized, from deeper interior toward reference)
    ΔT_space = @inbounds c[Nx - 1, j, k] - c[Nx - 2, j, k]

    # Temporal gradient for Orlanski phase velocity
    if sa.previous_tracers === nothing
        ΔT_time = zero(grid)
    else
        ΔT_time = @inbounds c[Nx - 1, j, k] - sa.previous_tracers[j, k]
    end

    # Phase velocity: min(cfl, max(0, -cfl * ΔT_time / ΔT_space))
    raw_phase = ifelse(ΔT_space == 0, cfl, -cfl * ΔT_time / ΔT_space)
    c_phase = ifelse(sa.use_phase_velocity,
                     min(cfl, max(zero(cfl), raw_phase)),
                     zero(cfl))

    # Normal velocity at face between reference and boundary cell
    u_normal = @inbounds model_fields.u[Nx, j, k]

    # Inflow: u < 0 at east boundary (flow into domain)
    gFac = ifelse(u_normal < 0, one(u_normal), zero(u_normal))

    # Relaxation rate
    γ = ifelse(sa.relaxation_timescale > 0, 1 / sa.relaxation_timescale, zero(FT))

    # Advective velocity (outflow only: max(0, u))
    u_adv = ifelse(sa.use_advection, max(zero(u_normal), u_normal), zero(u_normal))

    # Prescribed boundary value
    c_prescribed = getbc(bc, j, k, grid, clock, model_fields)

    # Tendency (note minus sign for east/north = right boundaries)
    c_ref = @inbounds c[Nx - 1, j, k]
    tendency = -(u_adv + c_phase) * ΔT_space / Δx - gFac * γ * (c_ref - c_prescribed)

    c_new = c_ref + Δt * tendency

    # Set the boundary cell and halo (Stevens overwrites AB2 at the boundary cell)
    @inbounds c[Nx, j, k] = c_new
    @inbounds c[Nx + 1, j, k] = c_new

    return nothing
end

# West boundary: boundary cell = 1, halo = 0
# Reference cell = 2, interior cell = 3
# Face velocity between boundary and ref = u[2]
@inline function _fill_west_halo!(j, k, grid, c, bc::SOBC, ::Tuple{Center, Any, Any}, clock, model_fields)
    sa = bc.classification.scheme

    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), zero(Δt), Δt)

    Δx = Δxᶠᶜᶜ(2, j, k, grid)
    FT = eltype(grid)
    cfl = ifelse(Δt > 0, Δx / (2 * Δt), zero(FT))

    # Spatial gradient (from deeper interior toward reference, pointing in -x)
    ΔT_space = @inbounds c[2, j, k] - c[3, j, k]

    # Temporal gradient
    if sa.previous_tracers === nothing
        ΔT_time = zero(grid)
    else
        ΔT_time = @inbounds c[2, j, k] - sa.previous_tracers[j, k]
    end

    # Phase velocity (positive sign for west boundary)
    raw_phase = ifelse(ΔT_space == 0, cfl, cfl * ΔT_time / ΔT_space)
    c_phase = ifelse(sa.use_phase_velocity,
                     min(cfl, max(zero(cfl), raw_phase)),
                     zero(cfl))

    # Normal velocity at face between boundary and reference cell
    u_normal = @inbounds model_fields.u[2, j, k]

    # Inflow: u > 0 at west boundary (flow into domain)
    gFac = ifelse(u_normal > 0, one(u_normal), zero(u_normal))

    # Relaxation rate
    γ = ifelse(sa.relaxation_timescale > 0, 1 / sa.relaxation_timescale, zero(FT))

    # Advective velocity (outflow only: min(0, u) for west)
    u_adv = ifelse(sa.use_advection, min(zero(u_normal), u_normal), zero(u_normal))

    # Prescribed boundary value
    c_prescribed = getbc(bc, j, k, grid, clock, model_fields)

    # Tendency (positive sign for west/south = left boundaries)
    c_ref = @inbounds c[2, j, k]
    tendency = (u_adv + c_phase) * ΔT_space / Δx - gFac * γ * (c_ref - c_prescribed)

    c_new = c_ref + Δt * tendency

    # Set the boundary cell and halo (Stevens overwrites AB2 at the boundary cell)
    @inbounds c[1, j, k] = c_new
    @inbounds c[0, j, k] = c_new

    return nothing
end

# North boundary: boundary cell = Ny, halo = Ny+1
# Reference cell = Ny-1, interior cell = Ny-2
# Face velocity between ref and boundary = v[i, Ny]
@inline function _fill_north_halo!(i, k, grid, c, bc::SOBC, ::Tuple{Any, Center, Any}, clock, model_fields)
    Ny = grid.Ny
    sa = bc.classification.scheme

    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), zero(Δt), Δt)

    Δy = Δyᶜᶠᶜ(i, Ny, k, grid)
    FT = eltype(grid)
    cfl = ifelse(Δt > 0, Δy / (2 * Δt), zero(FT))

    ΔT_space = @inbounds c[i, Ny - 1, k] - c[i, Ny - 2, k]

    if sa.previous_tracers === nothing
        ΔT_time = zero(grid)
    else
        ΔT_time = @inbounds c[i, Ny - 1, k] - sa.previous_tracers[i, k]
    end

    raw_phase = ifelse(ΔT_space == 0, cfl, -cfl * ΔT_time / ΔT_space)
    c_phase = ifelse(sa.use_phase_velocity,
                     min(cfl, max(zero(cfl), raw_phase)),
                     zero(cfl))

    v_normal = @inbounds model_fields.v[i, Ny, k]

    # Inflow: v < 0 at north boundary
    gFac = ifelse(v_normal < 0, one(v_normal), zero(v_normal))

    γ = ifelse(sa.relaxation_timescale > 0, 1 / sa.relaxation_timescale, zero(FT))

    v_adv = ifelse(sa.use_advection, max(zero(v_normal), v_normal), zero(v_normal))

    c_prescribed = getbc(bc, i, k, grid, clock, model_fields)

    c_ref = @inbounds c[i, Ny - 1, k]
    tendency = -(v_adv + c_phase) * ΔT_space / Δy - gFac * γ * (c_ref - c_prescribed)

    c_new = c_ref + Δt * tendency

    # Set the boundary cell and halo (Stevens overwrites AB2 at the boundary cell)
    @inbounds c[i, Ny, k] = c_new
    @inbounds c[i, Ny + 1, k] = c_new

    return nothing
end

# South boundary: boundary cell = 1, halo = 0
# Reference cell = 2, interior cell = 3
# Face velocity between boundary and ref = v[i, 2]
@inline function _fill_south_halo!(i, k, grid, c, bc::SOBC, ::Tuple{Any, Center, Any}, clock, model_fields)
    sa = bc.classification.scheme

    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), zero(Δt), Δt)

    Δy = Δyᶜᶠᶜ(i, 2, k, grid)
    FT = eltype(grid)
    cfl = ifelse(Δt > 0, Δy / (2 * Δt), zero(FT))

    ΔT_space = @inbounds c[i, 2, k] - c[i, 3, k]

    if sa.previous_tracers === nothing
        ΔT_time = zero(grid)
    else
        ΔT_time = @inbounds c[i, 2, k] - sa.previous_tracers[i, k]
    end

    raw_phase = ifelse(ΔT_space == 0, cfl, cfl * ΔT_time / ΔT_space)
    c_phase = ifelse(sa.use_phase_velocity,
                     min(cfl, max(zero(cfl), raw_phase)),
                     zero(cfl))

    v_normal = @inbounds model_fields.v[i, 2, k]

    # Inflow: v > 0 at south boundary
    gFac = ifelse(v_normal > 0, one(v_normal), zero(v_normal))

    γ = ifelse(sa.relaxation_timescale > 0, 1 / sa.relaxation_timescale, zero(FT))

    v_adv = ifelse(sa.use_advection, min(zero(v_normal), v_normal), zero(v_normal))

    c_prescribed = getbc(bc, i, k, grid, clock, model_fields)

    c_ref = @inbounds c[i, 2, k]
    tendency = (v_adv + c_phase) * ΔT_space / Δy - gFac * γ * (c_ref - c_prescribed)

    c_new = c_ref + Δt * tendency

    # Set the boundary cell and halo (Stevens overwrites AB2 at the boundary cell)
    @inbounds c[i, 1, k] = c_new
    @inbounds c[i, 0, k] = c_new

    return nothing
end
