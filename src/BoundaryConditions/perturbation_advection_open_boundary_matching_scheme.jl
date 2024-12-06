using Oceananigans.Grids: xspacing
# Immersed boundaries are defined later but we probably need todo this for when a boundary intersects the bathymetry
#using Oceananigans.ImmersedBoundaries: active_cell

"""
    PerturbationAdvection

For cases where we assume that the internal flow is a small perturbation from 
an external prescribed or coarser flow, we can split the velocity into background
and perturbation components:
...
see latex document for now

TODO: check what the coriolis is doing, and check what happens if U is the mean velocity
"""
struct PerturbationAdvection{FT}
    inflow_timescale :: FT
   outflow_timescale :: FT
end

Adapt.adapt_structure(to, pe::PerturbationAdvection) = 
    PerturbationAdvection(adapt(to, pe.outflow_timescale),
                          adapt(to, pe.inflow_timescale))

function PerturbationAdvectionOpenBoundaryCondition(val, FT = Float64; 
                                                    outflow_timescale = Inf, 
                                                    inflow_timescale = 300.0, kwargs...)

    classification = Open(PerturbationAdvection(inflow_timescale, outflow_timescale))

    @warn "`PerturbationAdvection` open boundaries matching scheme is experimental and un-tested/validated"
    
    return BoundaryCondition(classification, val; kwargs...)
end

const PAOBC = BoundaryCondition{<:Open{<:PerturbationAdvection}}

@inline function _fill_east_halo!(j, k, grid, u, bc::PAOBC, loc::Tuple{Face, Any, Any}, clock, model_fields)
    i = grid.Nx + 1

    Δt = clock.last_stage_Δt

    Δt = ifelse(isinf(Δt), 0, Δt)

    Δx = xspacing(i, j, k, grid, loc...)

    ūⁿ⁺¹ = getbc(bc, j, k, grid, clock, model_fields)

    uᵢⁿ     = @inbounds u[i, j, k]
    uᵢ₋₁ⁿ⁺¹ = @inbounds u[i - 1, j, k]

    U = max(0, min(1, Δt / Δx * ūⁿ⁺¹))

    τ = ifelse(ūⁿ⁺¹ >= 0, 
               bc.classification.matching_scheme.outflow_timescale, 
               bc.classification.matching_scheme.inflow_timescale)


    τ̃ = Δt / τ

    uᵢⁿ⁺¹ = (uᵢⁿ + U * uᵢ₋₁ⁿ⁺¹ + ūⁿ⁺¹ * τ̃) / (1 + τ̃ + U)

    @inbounds u[i, j, k] = uᵢⁿ⁺¹#ifelse(active_cell(i, j, k, grid), uᵢⁿ⁺¹, zero(grid))
end

@inline function _fill_west_halo!(j, k, grid, u, bc::PAOBC, loc::Tuple{Face, Any, Any}, clock, model_fields)
    Δt = clock.last_stage_Δt

    Δt = ifelse(isinf(Δt), 0, Δt)

    Δx = xspacing(1, j, k, grid, loc...)

    ūⁿ⁺¹ = getbc(bc, j, k, grid, clock, model_fields)

    uᵢⁿ     = @inbounds u[2, j, k]
    uᵢ₋₁ⁿ⁺¹ = @inbounds u[0, j, k] 

    U = min(0, max(-1, Δt / Δx * ūⁿ⁺¹))

    τ = ifelse(ūⁿ⁺¹ <= 0, 
               bc.classification.matching_scheme.outflow_timescale, 
               bc.classification.matching_scheme.inflow_timescale)

    τ̃ = Δt / τ

    u₁ⁿ⁺¹ = (uᵢⁿ - U * uᵢ₋₁ⁿ⁺¹ + ūⁿ⁺¹ * τ̃) / (1 + τ̃ - U)

    @inbounds u[1, j, k] = u₁ⁿ⁺¹#ifelse(active_cell(i, j, k, grid), uᵢⁿ⁺¹, zero(grid))
    @inbounds u[0, j, k] = u₁ⁿ⁺¹#ifelse(active_cell(i, j, k, grid), uᵢⁿ⁺¹, zero(grid))
end
