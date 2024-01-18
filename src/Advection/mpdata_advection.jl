using Oceananigans.BoundaryConditions
using Oceananigans.Fields: VelocityFields, location
using KernelAbstractions: @kernel, @index
using Oceananigans.Utils

struct MPData{FT, A} <: AbstractUpwindBiasedAdvectionScheme{1, FT} 
    iterations :: Int
    velocities :: A
    f :: FT
end

function MPData(grid; iterations = 1, f = 0.5)
    velocities = VelocityFields(grid)
    return MPData{eltype(grid), typeof(velocities)}(iterations, velocities, f)
end

# Basically just first order upwind (also called the "donor cell" scheme)
@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::MPData, c, args...) = ℑxᶠᵃᵃ(i, j, k, grid, c, args...)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::MPData, c, args...) = ℑyᵃᶠᵃ(i, j, k, grid, c, args...)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::MPData, c, args...) = ℑzᵃᵃᶠ(i, j, k, grid, c, args...)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::MPData, u, args...) = ℑxᶜᵃᵃ(i, j, k, grid, u, args...)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::MPData, v, args...) = ℑyᵃᶜᵃ(i, j, k, grid, v, args...)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::MPData, w, args...) = ℑzᵃᵃᶜ(i, j, k, grid, w, args...)

@inline inner_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::MPData, ψ, idx, loc, args...) = @inbounds ψ[i-1, j, k]
@inline inner_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::MPData, ψ, idx, loc, args...) = @inbounds ψ[i, j-1, k]
@inline inner_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::MPData, ψ, idx, loc, args...) = @inbounds ψ[i, j, k-1]

@inline inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::MPData, ψ, idx, loc, args...) = @inbounds ψ[i, j, k]
@inline inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::MPData, ψ, idx, loc, args...) = @inbounds ψ[i, j, k]
@inline inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::MPData, ψ, idx, loc, args...) = @inbounds ψ[i, j, k]

@inline inner_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::MPData, ψ::Function, idx, loc, args...) = @inbounds ψ(i-1, j, k, grid, args...)
@inline inner_left_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::MPData, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j-1, k, grid, args...)
@inline inner_left_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::MPData, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k-1, grid, args...)

@inline inner_right_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::MPData, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)
@inline inner_right_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::MPData, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)
@inline inner_right_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::MPData, ψ::Function, idx, loc, args...) = @inbounds ψ(i, j, k, grid, args...)

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

    divUc = # "Extractor function
          loc === (Center, Center, Center) ? div_Uc :
          loc === (Face, Center, Center)   ? div_𝐯u :
          loc === (Center, Face, Center)   ? div_𝐯v :
          loc === (Center, Center, Face)   ? div_𝐯w :
          error("Cannot MPData-correct for a field at $location")

    for iter in 1:scheme.iterations
        fill_halo_regions!(field)
        launch!(architecture(grid), grid, :xyz, _calculate_mpdata_velocities!, pseudo_velocities, grid, field, velocities, scheme.f)
        fill_halo_regions!(pseudo_velocities)
        launch!(architecture(grid), grid, :xyz, _update_tracer!, field, scheme, pseudo_velocities, grid, divUc, Δt) 
    end
end

""" 
Pseudo-velocities are calculated as:

uᵖ = abs(u)(1 - abs(u)) A - 2f u v B
vᵖ = abs(v)(1 - abs(v)) A - 2(1 - f) u v B

where A = Δx / 2ψ ∂x(ψ)
and   B = Δy / 2ψ ∂y(ψ)
"""
@kernel function _calculate_mpdata_velocities!(pseudo_velocities, grid, ψ, velocities, f)
    i, j, k = @index(Global, NTuple)

    u,  v,  w  = velocities.u, velocities.v, velocities.w
    uᵖ, vᵖ, wᵖ = pseudo_velocities

    ψᶠᶜᶜ = 2 * ℑxᶠᵃᵃ(i, j, k, grid, ψ)
    ψᶜᶠᶜ = 2 * ℑyᵃᶠᵃ(i, j, k, grid, ψ)

    @inbounds ψ₂ᶠᶜᶜ = (ψ[i, j, k] + ψ[i, j, k] + ψ[i, j, k] + ψ[i, j, k]) / 2
    @inbounds ψ₂ᶜᶠᶜ = (ψ[i, j, k] + ψ[i, j, k] + ψ[i, j, k] + ψ[i, j, k]) / 2
    

    Aᶠᶜᶜ = ifelse(abs(ψᶠᶜᶜ) > 0, Δx_qᶠᶜᶜ(i, j, k, grid, ∂xᶠᶜᶜ, ψ) / ψᶠᶜᶜ, 0)
    Aᶜᶠᶜ = ifelse(abs(ψᶜᶠᶜ) > 0, Δx_qᶜᶠᶜ(i, j, k, grid, ∂xᶜᶠᶜ, ℑxyᶠᶜᵃ, ψ) / ψᶜᶠᶜ, 0)

    Bᶠᶜᶜ = ifelse(abs(ψᶠᶜᶜ) > 0, Δy_qᶠᶜᶜ(i, j, k, grid, ∂yᶠᶜᶜ, ℑxyᶜᶠᵃ, ψ) / ψᶠᶜᶜ, 0)
    Bᶜᶠᶜ = ifelse(abs(ψᶜᶠᶜ) > 0, Δy_qᶜᶠᶜ(i, j, k, grid, ∂yᶜᶠᶜ, ψ) / ψᶜᶠᶜ, 0)

    @inbounds begin
        u_abs = abs(u[i, j, k])
        v_abs = abs(v[i, j, k])
        
        uᵖ[i, j, k] = u_abs * (1 - u_abs) * Aᶠᶜᶜ - 2 * f       * u[i, j, k] * v[i, j, k] * Bᶠᶜᶜ
        vᵖ[i, j, k] = v_abs * (1 - v_abs) * Bᶜᶠᶜ - 2 * (1 - f) * u[i, j, k] * v[i, j, k] * Aᶜᶠᶜ
    end 
end

@kernel function _update_tracer!(c, scheme, pseudo_velocities, grid, divUc, Δt)
    i, j, k = @index(Global, NTuple)

    ∇uc = divUc(i, j, k, grid, scheme, pseudo_velocities, c)
    @inbounds c[i, j, k] -= Δt * ∇uc
end

