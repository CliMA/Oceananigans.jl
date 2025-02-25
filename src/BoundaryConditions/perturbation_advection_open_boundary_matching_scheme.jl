using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶠ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Az_qᶜᶜᶠ

"""
    PerturbationAdvection

For cases where we assume that the internal flow is a small perturbation from 
an external prescribed or coarser flow, we can split the velocity into background
and perturbation components.

We begin with the equation governing the fluid in the interior:
    ∂ₜu + u⋅∇u = −∇P + F,
and note that on the boundary the pressure gradient is zero.
We can then assume that the flow composes of mean (U⃗) and pertubation (u⃗′) components,
and considering the x-component of velocity, we can rewrite the equation as
    ∂ₜu₁ = -u₁∂₁u - u₂∂₂u₁ - u₃∂₃u₁ + F₁ ≈ - U₁∂₁u₁′ - U₂∂₂u₁′ - U₃∂₃u₁′ + F.

Simplify by assuming that U⃗ = Ux̂, an then take a numerical step to find u₁.

When the boundaries are filled the interior is at time tₙ₊₁ so we can take
a backwards euler step (in the case that the mean flow is boundary normal) on a right boundary:
    (Uⁿ⁺¹ - Uⁿ) / Δt + (u′ⁿ⁺¹ - u′ⁿ) / Δt = - Uⁿ⁺¹ (u′ⁿ⁺¹ᵢ - u′ⁿ⁺¹ᵢ₋₁) / Δx + Fᵤ.

This can not be solved for general forcing, but if we assume the dominant forcing is
relaxation to the mean velocity (i.e. u′→0) then Fᵤ = -u′ / τ then we can find u′ⁿ⁺¹:
    u′ⁿ⁺¹ = (uⁿ + Ũu′ⁿ⁺¹ᵢ₋₁ - Uⁿ⁺¹) / (1 + Ũ + Δt/τ),

where Ũ = U Δt / Δx, then uⁿ⁺¹ is:
    uⁿ⁺¹ = (uᵢⁿ + Ũuᵢ₋₁ⁿ⁺¹ + Uⁿ⁺¹τ̃) / (1 + τ̃ + U)
    
where τ̃ = Δt/τ.

The same operation can be repeated for left boundaries.
"""
struct PerturbationAdvection{VT, FT}
       backward_step :: VT
    inflow_timescale :: FT
   outflow_timescale :: FT
end

Adapt.adapt_structure(to, pe::PerturbationAdvection) = 
    PerturbationAdvection(adapt(to, pe.backward_step),
                          adapt(to, pe.inflow_timescale),
                          adapt(to, pe.outflow_timescale))

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
    boundary_secret_storage_indices = (i, 0, k)

    Δy = Δyᶜᶠᶜ(i, 1, k, grid)
    
    step_left_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices, boundary_secret_storage_indices, grid, u, clock, model_fields, Δy)

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
    boundary_secret_storage_indices = (i, j, 0)

    Δz = Δzᶜᶜᶠ(i, j, 1, grid)

    step_left_boundary!(bc, i, j, boundary_indices, boundary_adjacent_indices, boundary_secret_storage_indices, grid, u, clock, model_fields, Δz)

    return nothing
end
