using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.Fields: VelocityFields, location
using KernelAbstractions: @kernel, @index
using Oceananigans.Utils

struct MPData{FT, I, A} <: AbstractUpwindBiasedAdvectionScheme{1, FT} 
    velocities :: A
    iterations :: I
end

function MPData(grid; iterations = nothing)
    velocities = VelocityFields(grid)
    return MPData{eltype(grid), typeof(iterations), typeof(velocities)}(velocities, iterations)
end

# Optimal MPData scheme from "Antidiffusive Velocities for Multipass Donor Cell Advection"
# which has only two passes
const OptimalMPData = MPData{<:Any, <:Nothing}

# Basically just first order upwind (also called the "donor cell" scheme)
@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::MPData, c, args...) = ℑxᶠᵃᵃ(i, j, k, grid, c, args...)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::MPData, c, args...) = ℑyᵃᶠᵃ(i, j, k, grid, c, args...)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::MPData, c, args...) = ℑzᵃᵃᶠ(i, j, k, grid, c, args...)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, ::MPData, u, args...) = ℑxᶜᵃᵃ(i, j, k, grid, u, args...)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, ::MPData, v, args...) = ℑyᵃᶜᵃ(i, j, k, grid, v, args...)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, ::MPData, w, args...) = ℑzᵃᵃᶜ(i, j, k, grid, w, args...)

@inline inner_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::MPData, ψ, idx, loc, args...) = @inbounds ψ[i-1, j, k]
@inline inner_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::MPData, ψ, idx, loc, args...) = @inbounds ψ[i, j-1, k]
@inline inner_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::MPData, ψ, idx, loc, args...) = @inbounds ψ[i, j, k-1]

@inline inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::MPData, ψ, idx, loc, args...) = @inbounds ψ[i, j, k]
@inline inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::MPData, ψ, idx, loc, args...) = @inbounds ψ[i, j, k]
@inline inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::MPData, ψ, idx, loc, args...) = @inbounds ψ[i, j, k]

@inline inner_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::MPData, ψ::Function, idx, loc, args...) = @inbounds ψ(i-1, j, k, grid, args...)
@inline inner_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::MPData, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j-1, k, grid, args...)
@inline inner_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::MPData, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k-1, grid, args...)

@inline inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::MPData, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)
@inline inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::MPData, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)
@inline inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::MPData, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)

function correct_advection!(model, Δt)
    grid = model.grid
    velocities = model.velocities
    
    for tracer_name in propertynames(model.tracers)
        @inbounds tracer = model.tracers[tracer_name]
        @inbounds scheme = model.advection[tracer_name]
        correct_mpdata_advection!(tracer, grid, Δt, velocities, scheme)
    end
end

correct_mpdata_advection!(field, grid, Δt, velocities, scheme) = nothing 

function correct_mpdata_advection!(field, grid, Δt, velocities, scheme::MPData) 
    pseudo_velocities = scheme.velocities
    loc = location(field)

    set!(pseudo_velocities.u, velocities.u)
    set!(pseudo_velocities.v, velocities.v)

    divUc = # "Extractor function
          loc === (Center, Center, Center) ? div_Uc :
          loc === (Face, Center, Center)   ? div_𝐯u :
          loc === (Center, Face, Center)   ? div_𝐯v :
          loc === (Center, Center, Face)   ? div_𝐯w :
          error("Cannot MPData-correct for a field at $location")

    mpdata_iterate!(field, grid, scheme, pseudo_velocities, Δt, divUc)

    return nothing
end

function mpdata_iterate!(field, grid, scheme::OptimalMPData, pseudo_velocities, Δt, divUc)

    fill_halo_regions!(field)
    launch!(architecture(grid), grid, :xyz, _calculate_optimal_mpdata_velocities!, 
            pseudo_velocities, grid, field, Δt)

    fill_halo_regions!(pseudo_velocities)
    launch!(architecture(grid), grid, :xyz, _update_tracer!, field, 
            scheme, pseudo_velocities, grid, divUc, Δt) 

    return nothing
end

function mpdata_iterate!(field, grid, scheme, pseudo_velocities, Δt, divUc)

    for iter in 1:scheme.iterations
        fill_halo_regions!(field)
        launch!(architecture(grid), grid, :xyz, _calculate_mpdata_velocities!, 
                pseudo_velocities, grid, field, Δt)

        fill_halo_regions!(pseudo_velocities)
        launch!(architecture(grid), grid, :xyz, _update_tracer!, field, scheme, 
                pseudo_velocities, grid, divUc, Δt) 
    end

    return nothing
end

""" 
Pseudo-velocities are calculated as:

uᵖ = abs(u)(1 - abs(u)) A - u v B
vᵖ = abs(v)(1 - abs(v)) A - u v A

where A = Δx / 2ψ ∂x(ψ)
and   B = Δy / 2ψ ∂y(ψ)
"""
@kernel function _calculate_mpdata_velocities!(velocities, grid, ψ, Δt)
    i, j, k = @index(Global, NTuple)

    Aᶠᶜᶜ, Bᶠᶜᶜ, Aᶜᶠᶜ, Bᶜᶠᶜ = mpdata_auxiliaries(i, j, k, grid, ψ)
    uᵖ, vᵖ, wᵖ = velocities

    ξ, η = mpdata_pseudo_velocities(i, j, k, grid, velocities, Aᶠᶜᶜ, Bᶠᶜᶜ, Aᶜᶠᶜ, Bᶜᶠᶜ)

    @inbounds begin
        uᵖ[i, j, k] = min(u_abs, abs(ξ)) * sign(ξ)
        vᵖ[i, j, k] = min(v_abs, abs(η)) * sign(η)
    end 
end

""" 
Pseudo-velocities are calculated as:

uᵖ = ∑₁∞ abs(uᴾ)(1 - abs(uᴾ)) A - uᴾ vᴾ B
vᵖ = ∑₁∞ abs(vᴾ)(1 - abs(vᴾ)) A - uᴾ vᴾ A

where A = Δx / 2ψ ∂x(ψ) stays fixed
and   B = Δy / 2ψ ∂y(ψ) stays fixed
"""
@kernel function _calculate_optimal_mpdata_velocities!(velocities, grid, ψ, Δt)
    i, j, k = @index(Global, NTuple)

    uᵖ, vᵖ, wᵖ = velocities
    Aᶠᶜᶜ, Bᶠᶜᶜ, Aᶜᶠᶜ, Bᶜᶠᶜ = mpdata_auxiliaries(i, j, k, grid, ψ)

    @inbounds begin
        u_abs = abs(uᵖ[i, j, k])
        v_abs = abs(vᵖ[i, j, k])
        
        Aₐᶠᶜᶜ = abs(Aᶠᶜᶜ)
        Bₐᶠᶜᶜ = abs(Bᶠᶜᶜ)
        Aₐᶜᶠᶜ = abs(Aᶜᶠᶜ)
        Bₐᶜᶠᶜ = abs(Bᶜᶠᶜ)

        ξ, η = mpdata_pseudo_velocities(i, j, k, grid, velocities, Aᶠᶜᶜ, Bᶠᶜᶜ, Aᶜᶠᶜ, Bᶜᶠᶜ)

        ξ *= Δt / Δxᶠᶜᶜ(i, j, k, grid)
        η *= Δt / Δyᶜᶠᶜ(i, j, k, grid)  

        dᵃ₁ = (1 - Aₐᶠᶜᶜ)
        dᵃ₂ = (1 - Aₐᶠᶜᶜ^2)
        dᵃ₃ = (1 - Aₐᶠᶜᶜ^3)

        cΣᵅ = abs(dᵃ₁) > 0
        cΣᵝ = cΣᵅ & (abs(dᵃ₂) > 0)
        cΣᵞ = cΣᵝ & (abs(dᵃ₃) > 0)
        Σˣᵅ = ifelse(cΣᵅ, 1 / dᵃ₁,                         0)
        Σˣᵝ = ifelse(cΣᵝ, - Aᶠᶜᶜ / (dᵃ₁ * dᵃ₂),            0)
        Σˣᵞ = ifelse(cΣᵞ, 2 * Aₐᶠᶜᶜ^3 / (dᵃ₁ * dᵃ₂ * dᵃ₃), 0)

        dᵇ₁ = (1 - Bₐᶜᶠᶜ)
        dᵇ₂ = (1 - Bₐᶜᶠᶜ^2)
        dᵇ₃ = (1 - Bₐᶜᶠᶜ^3)

        cΣᵅ = abs(dᵇ₁) > 0
        cΣᵝ = cΣᵅ & (abs(dᵇ₂) > 0)
        cΣᵞ = cΣᵝ & (abs(dᵇ₃) > 0)
        Σʸᵅ = ifelse(cΣᵅ, 1 / dᵇ₁,                         0)
        Σʸᵝ = ifelse(cΣᵝ, - Bᶜᶠᶜ / (dᵇ₁ * dᵇ₂),            0)
        Σʸᵞ = ifelse(cΣᵞ, 2 * Bₐᶜᶠᶜ^3 / (dᵇ₁ * dᵇ₂ * dᵇ₃), 0)

        dᵃᵇ₁ = (1 - abs(Aᶠᶜᶜ   * Bᶠᶜᶜ))
        dᵃᵇ₂ = (1 - abs(Aᶠᶜᶜ^2 * Bᶠᶜᶜ))
        dᵃᵇ₃ = (1 - abs(Aᶠᶜᶜ   * Bᶠᶜᶜ^2))

        cΣᵃ = (abs(dᵃ₁) > 0) & (abs(dᵃᵇ₁) > 1)
        cΣᵇ = cΣᵃ & (abs(dᵃᵇ₂) > 0) & (abs(dᵃ₂) > 0)
        cΣᶜ = cΣᵃ & (abs(dᵃᵇ₃) > 0) 
        Σˣᵃ = ifelse(cΣᵃ, - Bᶠᶜᶜ / (dᵃ₁ * dᵃᵇ₁), 0)
        Σˣᵇ = ifelse(cΣᵇ, Bᶠᶜᶜ * Aᶠᶜᶜ / (dᵃ₁ * dᵃᵇ₂) * (Bₐᶠᶜᶜ / dᵃᵇ₁ + 2Aᶠᶜᶜ / dᵃ₂), 0)
        Σˣᶜ = ifelse(cΣᶜ, Aₐᶠᶜᶜ * Bᶠᶜᶜ^2 / (dᵃ₁ * dᵃᵇ₃ * dᵃᵇ₁), 0)

        dᵃᵇ₁ = (1 - abs(Bᶜᶠᶜ   * Aᶜᶠᶜ))
        dᵃᵇ₂ = (1 - abs(Bᶜᶠᶜ^2 * Aᶜᶠᶜ))
        dᵃᵇ₃ = (1 - abs(Bᶜᶠᶜ   * Aᶜᶠᶜ^2))

        cΣᵃ = (abs(dᵇ₁) > 0) & (abs(dᵃᵇ₁) > 1)
        cΣᵇ = cΣᵃ & (abs(dᵃᵇ₂) > 0) & (abs(dᵇ₂) > 0)
        cΣᶜ = cΣᵃ & (abs(dᵃᵇ₃) > 0) 
        Σʸᵃ = ifelse(cΣᵃ, - Aᶜᶠᶜ / (dᵇ₁ * dᵃᵇ₁), 0)
        Σʸᵇ = ifelse(cΣᵇ, Aᶜᶠᶜ * Bᶜᶠᶜ / (dᵇ₁ * dᵃᵇ₂) * (Aₐᶜᶠᶜ / dᵃᵇ₁ + 2Bᶜᶠᶜ / dᵇ₂), 0)
        Σʸᶜ = ifelse(cΣᶜ, Bₐᶜᶠᶜ * Aᶜᶠᶜ^2 / (dᵇ₁ * dᵃᵇ₃ * dᵃᵇ₁), 0)

        uᵖ[i, j, k] = (Σˣᵅ * ξ + Σˣᵝ * ξ^2 + Σˣᵞ * ξ^3 + Σˣᵃ * ξ * η + Σˣᵇ * ξ^2 * η + Σˣᶜ * ξ * η^2) * Δxᶠᶜᶜ(i, j, k, grid) / Δt
        vᵖ[i, j, k] = (Σʸᵅ * η + Σʸᵝ * η^2 + Σʸᵞ * η^3 + Σʸᵃ * ξ * η + Σʸᵇ * η^2 * ξ + Σʸᶜ * η * ξ^2) * Δyᶜᶠᶜ(i, j, k, grid) / Δt
        
        uᵖ[i, j, k] = min(u_abs, abs(uᵖ[i, j, k])) * sign(uᵖ[i, j, k])
        vᵖ[i, j, k] = min(v_abs, abs(vᵖ[i, j, k])) * sign(vᵖ[i, j, k])
    end 
end

@inline function mpdata_auxiliaries(i, j, k, grid, ψ)

    ψ₁ᶠᶜᶜ = 2 * ℑxᶠᵃᵃ(i, j, k, grid, ψ)
    ψ₁ᶜᶠᶜ = 2 * ℑyᵃᶠᵃ(i, j, k, grid, ψ)
    Δψ₁ᶠᶜᶜ = δxᶠᵃᵃ(i, j, k, grid, ψ)
    Δψ₁ᶜᶠᶜ = δyᵃᶠᵃ(i, j, k, grid, ψ)

    # Calculating A and B
    @inbounds begin
        ψ₂ᶠᶜᶜ = (ψ[i, j+1, k] + ψ[i-1, j+1, k] + ψ[i, j-1, k] + ψ[i-1, j-1, k])
        ψ₂ᶜᶠᶜ = (ψ[i+1, j, k] + ψ[i+1, j-1, k] + ψ[i-1, j, k] + ψ[i-1, j-1, k])

        Δψ₂ᶠᶜᶜ = (ψ[i, j+1, k] + ψ[i-1, j+1, k] - ψ[i, j-1, k] - ψ[i-1, j-1, k])
        Δψ₂ᶜᶠᶜ = (ψ[i+1, j, k] + ψ[i+1, j-1, k] - ψ[i-1, j, k] - ψ[i-1, j-1, k])

        Aᶠᶜᶜ = ifelse(abs(ψ₁ᶠᶜᶜ) > 0, Δψ₁ᶠᶜᶜ / ψ₁ᶠᶜᶜ, 0)
        Bᶠᶜᶜ = ifelse(abs(ψ₂ᶠᶜᶜ) > 0, Δψ₂ᶠᶜᶜ / ψ₂ᶠᶜᶜ, 0)
        Aᶜᶠᶜ = ifelse(abs(ψ₂ᶜᶠᶜ) > 0, Δψ₂ᶜᶠᶜ / ψ₂ᶜᶠᶜ, 0)
        Bᶜᶠᶜ = ifelse(abs(ψ₁ᶜᶠᶜ) > 0, Δψ₁ᶜᶠᶜ / ψ₁ᶜᶠᶜ, 0)
    end

    return Aᶠᶜᶜ, Bᶠᶜᶜ, Aᶜᶠᶜ, Bᶜᶠᶜ
end

@inline function mpdata_pseudo_velocities(i, j, k, grid, U, Aᶠᶜᶜ, Bᶠᶜᶜ, Aᶜᶠᶜ, Bᶜᶠᶜ)

    uᴾ, vᴾ, _ = U

    u_abs = abs(uᵖ[i, j, k])
    v_abs = abs(vᵖ[i, j, k])

    u̅ᶠᶜᶜ = abs(uᵖ[i, j, k]) * Δt / Δxᶠᶜᶜ(i, j, k, grid)
    v̅ᶜᶠᶜ = abs(vᵖ[i, j, k]) * Δt / Δyᶜᶠᶜ(i, j, k, grid)  
    u̅ᶜᶠᶜ = ℑxyᶜᶠᵃ(i, j, k, grid, uᵖ) * Δt / Δxᶜᶠᶜ(i, j, k, grid)
    v̅ᶠᶜᶜ = ℑxyᶠᶜᵃ(i, j, k, grid, vᵖ) * Δt / Δyᶠᶜᶜ(i, j, k, grid)  

    ξ = u_abs * (1 - u̅ᶠᶜᶜ) * Aᶠᶜᶜ - uᵖ[i, j, k] * v̅ᶠᶜᶜ * Bᶠᶜᶜ
    η = v_abs * (1 - v̅ᶜᶠᶜ) * Bᶜᶠᶜ - vᵖ[i, j, k] * u̅ᶜᶠᶜ * Aᶜᶠᶜ

    return ξ, η
end

@kernel function _update_tracer!(c, scheme, pseudo_velocities, grid, divUc, Δt)
    i, j, k = @index(Global, NTuple)

    ∇uc = divUc(i, j, k, grid, scheme, pseudo_velocities, c)
    @inbounds c[i, j, k] -= Δt * ∇uc
end

