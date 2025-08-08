using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶠ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Az_qᶜᶜᶠ
using Oceananigans: defaults

"""
    PerturbationAdvection

For cases where we assume that the internal flow is a small perturbation from
an external prescribed or coarser flow, we can split the velocity into background
and perturbation components.
"""
struct PerturbationAdvection{FT}
    inflow_timescale :: FT
    outflow_timescale :: FT
end

Adapt.adapt_structure(to, pe::PerturbationAdvection) =
    PerturbationAdvection(adapt(to, pe.inflow_timescale),
                          adapt(to, pe.outflow_timescale))

function PerturbationAdvectionOpenBoundaryCondition(val, FT = defaults.FloatType;
                                                    outflow_timescale = Inf,
                                                    inflow_timescale = 0, kwargs...)
    inflow_timescale = convert(FT, inflow_timescale)
    outflow_timescale = convert(FT, outflow_timescale)
    classification = Open(PerturbationAdvection(inflow_timescale, outflow_timescale))

    return BoundaryCondition(classification, val; kwargs...)
end

const PAOBC = BoundaryCondition{<:Open{<:PerturbationAdvection}}

@inline function step_right_boundary!(bc::PAOBC, l, m, boundary_indices, boundary_adjacent_indices,
                                      grid, u, clock, model_fields, ΔX)
    iᴮ, jᴮ, kᴮ = boundary_indices
    iᴬ, jᴬ, kᴬ = boundary_adjacent_indices
    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), 0, Δt)

    ūⁿ⁺¹    = getbc(bc, l, m, grid, clock, model_fields)
    uᵢⁿ     = @inbounds getindex(u, iᴮ, jᴮ, kᴮ)
    uᵢ₋₁ⁿ⁺¹ = @inbounds getindex(u, iᴬ, jᴬ, kᴬ)
    U = max(0, min(1, Δt / ΔX * ūⁿ⁺¹))

    pa = bc.classification.scheme
    τ = ifelse(ūⁿ⁺¹ >= 0, pa.outflow_timescale, pa.inflow_timescale)
    τ̃ = Δt / τ

    relaxed_uᵢⁿ⁺¹ = (uᵢⁿ + U * uᵢ₋₁ⁿ⁺¹ + ūⁿ⁺¹ * τ̃) / (1 + τ̃ + U)
    uᵢⁿ⁺¹         = ifelse(τ == 0, ūⁿ⁺¹, relaxed_uᵢⁿ⁺¹)

    @inbounds setindex!(u, uᵢⁿ⁺¹, iᴮ, jᴮ, kᴮ)

    return nothing
end

@inline function step_left_boundary!(bc::PAOBC, l, m, boundary_indices, boundary_adjacent_indices,
                                     grid, u, clock, model_fields, ΔX)
    iᴮ, jᴮ, kᴮ = boundary_indices
    iᴬ, jᴬ, kᴬ = boundary_adjacent_indices
    Δt = clock.last_stage_Δt
    Δt = ifelse(isinf(Δt), 0, Δt)

    ūⁿ⁺¹    = getbc(bc, l, m, grid, clock, model_fields)
    uᵢⁿ     = @inbounds getindex(u, iᴮ, jᴮ, kᴮ)
    uᵢ₋₁ⁿ⁺¹ = @inbounds getindex(u, iᴬ, jᴬ, kᴬ)
    U = min(0, max(-1, Δt / ΔX * ūⁿ⁺¹))

    pa = bc.classification.scheme
    τ = ifelse(ūⁿ⁺¹ <= 0, pa.outflow_timescale, pa.inflow_timescale)
    τ̃ = Δt / τ

    relaxed_u₁ⁿ⁺¹ = (uᵢⁿ - U * uᵢ₋₁ⁿ⁺¹ + ūⁿ⁺¹ * τ̃) / (1 + τ̃ - U)
    u₁ⁿ⁺¹         = ifelse(τ == 0, ūⁿ⁺¹, relaxed_u₁ⁿ⁺¹)

    @inbounds setindex!(u, u₁ⁿ⁺¹, iᴮ, jᴮ, kᴮ)

    return nothing
end

@inline function _fill_east_halo!(j, k, grid, u, bc::PAOBC, ::Tuple{Face, Any, Any}, clock, model_fields)
    i = grid.Nx + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i-1, j, k)

    Δx = Δxᶠᶜᶜ(i, j, k, grid)

    step_right_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices, grid, u, clock, model_fields, Δx)

    return nothing
end

@inline function _fill_west_halo!(j, k, grid, u, bc::PAOBC, ::Tuple{Face, Any, Any}, clock, model_fields)
    boundary_indices = (1, j, k)
    boundary_adjacent_indices = (2, j, k)
    Δx = Δxᶠᶜᶜ(1, j, k, grid)
    step_left_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices, grid, u, clock, model_fields, Δx)

    return nothing
end

@inline function _fill_north_halo!(i, k, grid, u, bc::PAOBC, ::Tuple{Any, Face, Any}, clock, model_fields)
    j = grid.Ny + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j-1, k)

    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    step_right_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices, grid, u, clock, model_fields, Δy)

    return nothing
end

@inline function _fill_south_halo!(i, k, grid, u, bc::PAOBC, ::Tuple{Any, Face, Any}, clock, model_fields)
    boundary_indices = (i, 1, k)
    boundary_adjacent_indices = (i, 2, k)

    Δy = Δyᶜᶠᶜ(i, 1, k, grid)
    step_left_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices, grid, u, clock, model_fields, Δy)

    return nothing
end

@inline function _fill_top_halo!(i, j, grid, u, bc::PAOBC, ::Tuple{Any, Any, Face}, clock, model_fields)
    k = grid.Nz + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j, k-1)

    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    step_right_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices, grid, u, clock, model_fields, Δz)

    return nothing
end

@inline function _fill_bottom_halo!(i, j, grid, u, bc::PAOBC, ::Tuple{Any, Any, Face}, clock, model_fields)
    boundary_indices = (i, j, 1)
    boundary_adjacent_indices = (i, j, 2)

    Δz = Δzᶜᶜᶠ(i, j, 1, grid)
    step_left_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices, grid, u, clock, model_fields, Δz)

    return nothing
end


