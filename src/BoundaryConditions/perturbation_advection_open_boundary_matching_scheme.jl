using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶠ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Az_qᶜᶜᶠ

"""
    PerturbationAdvection

For cases where we assume that the internal flow is a small perturbation from 
an external prescribed or coarser flow, we can split the velocity into background
and perturbation components:
...
see latex document for now

TODO: check what the coriolis is doing, and check what happens if U is the mean velocity
"""
struct PerturbationAdvection{VT, FT}
       backward_step :: VT
    inflow_timescale :: FT
   outflow_timescale :: FT
end

Adapt.adapt_structure(to, pe::PerturbationAdvection) = 
    PerturbationAdvection(adapt(to, pe.backward_step),
                          adapt(to, pe.outflow_timescale),
                          adapt(to, pe.inflow_timescale))

function PerturbationAdvectionOpenBoundaryCondition(val, FT = Float64; 
                                                    backward_step = true,
                                                    outflow_timescale = Inf, 
                                                    inflow_timescale = 300.0, kwargs...)

    classification = Open(PerturbationAdvection(Val(backward_step), inflow_timescale, outflow_timescale))

    @warn "`PerturbationAdvection` open boundaries matching scheme is experimental and un-tested/validated"
    
    return BoundaryCondition(classification, val; kwargs...)
end

const PAOBC = BoundaryCondition{<:Open{<:PerturbationAdvection}}

const BPAOBC = BoundaryCondition{<:Open{<:PerturbationAdvection{Val{true}}}}
const FPAOBC = BoundaryCondition{<:Open{<:PerturbationAdvection{Val{false}}}}

@inline function step_right_boundary!(bc::BPAOBC, l, m, boundary_indices, boundary_adjacent_indices, 
                                      grid, u, clock, model_fields, ΔX)
    Δt = clock.last_stage_Δt

    Δt = ifelse(isinf(Δt), 0, Δt)

    ūⁿ⁺¹ = getbc(bc, l, m, grid, clock, model_fields)

    uᵢⁿ     = @inbounds getindex(u, boundary_indices...)
    uᵢ₋₁ⁿ⁺¹ = @inbounds getindex(u, boundary_adjacent_indices...)

    U = max(0, min(1, Δt / ΔX * ūⁿ⁺¹))

    pa = bc.classification.matching_scheme

    τ = ifelse(ūⁿ⁺¹ >= 0, pa.outflow_timescale, pa.inflow_timescale)

    τ̃ = Δt / τ

    uᵢⁿ⁺¹ = (uᵢⁿ + U * uᵢ₋₁ⁿ⁺¹ + ūⁿ⁺¹ * τ̃) / (1 + τ̃ + U)

    @inbounds setindex!(u, uᵢⁿ⁺¹, boundary_indices...)

    return nothing
end

@inline function step_left_boundary!(bc::BPAOBC, l, m, boundary_indices, boundary_adjacent_indices, boundary_secret_storage_indices, 
                                     grid, u, clock, model_fields, ΔX)
    Δt = clock.last_stage_Δt

    Δt = ifelse(isinf(Δt), 0, Δt)

    ūⁿ⁺¹ = getbc(bc, l, m, grid, clock, model_fields)

    uᵢⁿ     = @inbounds getindex(u, boundary_secret_storage_indices...)
    uᵢ₋₁ⁿ⁺¹ = @inbounds getindex(u, boundary_adjacent_indices...)

    U = min(0, max(-1, Δt / ΔX * ūⁿ⁺¹))

    pa = bc.classification.matching_scheme

    τ = ifelse(ūⁿ⁺¹ <= 0, pa.outflow_timescale, pa.inflow_timescale)

    τ̃ = Δt / τ

    u₁ⁿ⁺¹ = (uᵢⁿ - U * uᵢ₋₁ⁿ⁺¹ + ūⁿ⁺¹ * τ̃) / (1 + τ̃ - U)

    @inbounds setindex!(u, u₁ⁿ⁺¹, boundary_indices...)
    @inbounds setindex!(u, u₁ⁿ⁺¹, boundary_secret_storage_indices...)

    return nothing
end


@inline function step_right_boundary!(bc::FPAOBC, l, m, boundary_indices, boundary_adjacent_indices, 
                                      grid, u, clock, model_fields, ΔX)
    Δt = clock.last_stage_Δt

    Δt = ifelse(isinf(Δt), 0, Δt)

    ūⁿ⁺¹ = getbc(bc, l, m, grid, clock, model_fields)

    uᵢⁿ     = @inbounds getindex(u, boundary_indices...)
    uᵢ₋₁ⁿ⁺¹ = @inbounds getindex(u, boundary_adjacent_indices...)

    U = max(0, min(1, Δt / ΔX * ūⁿ⁺¹))

    pa = bc.classification.matching_scheme

    τ = ifelse(ūⁿ⁺¹ >= 0, pa.outflow_timescale, pa.inflow_timescale)

    τ̃ = Δt / τ

    uᵢⁿ⁺¹ = uᵢⁿ + U * (uᵢ₋₁ⁿ⁺¹ - ūⁿ⁺¹) + (ūⁿ⁺¹ - uᵢⁿ) * τ̃

    @inbounds setindex!(u, uᵢⁿ⁺¹, boundary_indices...)

    return nothing
end

@inline function step_left_boundary!(bc::FPAOBC, l, m, boundary_indices, boundary_adjacent_indices, boundary_secret_storage_indices, 
                                     grid, u, clock, model_fields, ΔX)
    Δt = clock.last_stage_Δt

    Δt = ifelse(isinf(Δt), 0, Δt)

    ūⁿ⁺¹ = getbc(bc, l, m, grid, clock, model_fields)

    uᵢⁿ     = @inbounds getindex(u, boundary_secret_storage_indices...)
    uᵢ₋₁ⁿ⁺¹ = @inbounds getindex(u, boundary_adjacent_indices...)

    U = min(0, max(-1, Δt / ΔX * ūⁿ⁺¹))

    pa = bc.classification.matching_scheme

    τ = ifelse(ūⁿ⁺¹ <= 0, pa.outflow_timescale, pa.inflow_timescale)

    τ̃ = Δt / τ

    u₁ⁿ⁺¹ = uᵢⁿ - U * (uᵢ₋₁ⁿ⁺¹ - ūⁿ⁺¹) + (ūⁿ⁺¹ - uᵢⁿ) * τ̃

    @inbounds setindex!(u, u₁ⁿ⁺¹, boundary_indices...)
    @inbounds setindex!(u, u₁ⁿ⁺¹, boundary_secret_storage_indices...)

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
    boundary_secret_storage_indices = (0, j, k)

    Δx = Δxᶠᶜᶜ(1, j, k, grid)

    step_left_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices, boundary_secret_storage_indices, grid, u, clock, model_fields, Δx)

    return nothing
end
