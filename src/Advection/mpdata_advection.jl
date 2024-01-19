using Oceananigans.BoundaryConditions
using Oceananigans.Fields
using Oceananigans.Fields: VelocityFields, location
using KernelAbstractions: @kernel, @index
using Oceananigans.Utils
using Adapt 

struct MPData{FT, I, A, V} <: AbstractUpwindBiasedAdvectionScheme{1, FT} 
           velocities :: A # MPData antidiffusive pseudo-velocities 
  previous_velocities :: A # Non corrected velocities
   vertical_advection :: V # if not a nothing, the advection scheme used in the vertical direction
           iterations :: I # number of mpdata passes, if nothing an "optimal" two-pass scheme is used
    
    MPData{FT}(v::A, pv::A, va::V, i::I) where {FT, A, V, I} = new{FT, I, A, V}(v, pv, va, i)
end

function MPData(grid; iterations = nothing,
                      vertical_advection = nothing)
    velocities = VelocityFields(grid)
    previous_velocities = VelocityFields(grid)
    return MPData{eltype(grid)}(velocities, previous_velocities, vertical_advection, iterations)
end

Adapt.adapt_structure(to, scheme::MPData{FT}) where FT = 
    MPData{FT}(Adapt.adapt(to, scheme.velocities),
               Adapt.adapt(to, scheme.previous_velocities),
               Adapt.adapt(to, scheme.vertical_advection),
               Adapt.adapt(to, scheme.iterations))

# Optimal MPData scheme from "Antidiffusive Velocities for Multipass Donor Cell Advection"
# only two passes with a pseudovelocity calculated from compounding infinite passes
const OptimalMPData = MPData{<:Any, <:Nothing}

# Different scheme in the vertical direction
const PartialMPData = MPData{<:Any, <:Any, <:AbstractAdvectionScheme}

# an mpdata pass is equivalent to first order upwind (called the "donor cell" scheme in mpdata literature)
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

# second to Nth correction pass, applied after the tracer/momentum update
# This should probably go in the Models module since it is different for 
# hydrostatic and nonhydrostatic and should be part of `update_state!`
function correct_advection!(model, Δt)
    grid = model.grid
    velocities = model.velocities
    
    for tracer_name in propertynames(model.tracers)
        @inbounds tracer = model.tracers[tracer_name]
        @inbounds scheme = model.advection[tracer_name]
        correct_mpdata_tracer!(tracer, grid, Δt, velocities, scheme)
    end

    correct_mpdata_momentum!(model, Δt)

    return nothing
end

correct_mpdata_momentum!(velocities, grid, Δt, scheme, dims) = nothing

# we need to save the previous velocities to ensure a correct mpdata pass
function correct_mpdata_momentum!(velocities, grid, Δt, scheme::MPData, dims)
    pseudo_velocities = scheme.velocities
    previous_velocities = scheme.previous_velocities

    set!(pseudo_velocities.u, velocities.u)
    set!(pseudo_velocities.v, velocities.v)
    set!(pseudo_velocities.w, velocities.w)

    set!(previous_velocities.u, velocities.u)
    set!(previous_velocities.v, velocities.v)
    set!(previous_velocities.w, velocities.w)

    mpdata_iterate!(velocities.u, grid, scheme, pseudo_velocities, Δt, div_𝐯u)

    set!(pseudo_velocities.u, previous_velocities.u)
    set!(pseudo_velocities.v, previous_velocities.v)
    set!(pseudo_velocities.w, previous_velocities.w)

    mpdata_iterate!(velocities.v, grid, scheme, pseudo_velocities, Δt, div_𝐯u)

    if dims == 3 # 3D evolution including vertical velocity
        set!(pseudo_velocities.u, previous_velocities.u)
        set!(pseudo_velocities.v, previous_velocities.v)
        set!(pseudo_velocities.w, previous_velocities.w)
    
        mpdata_iterate!(velocities.w, grid, scheme, pseudo_velocities, Δt, div_𝐯w)
    end

    return nothing
end

correct_mpdata_tracer!(field, grid, Δt, velocities, scheme) = nothing 

function correct_mpdata_tracer!(field, grid, Δt, velocities, scheme::MPData) 
    pseudo_velocities = scheme.velocities

    set!(pseudo_velocities.u, velocities.u)
    set!(pseudo_velocities.v, velocities.v)
    set!(pseudo_velocities.w, velocities.w)

    mpdata_iterate!(field, grid, scheme, pseudo_velocities, Δt, div_Uc)

    return nothing
end

# The optimal MPData scheme uses only one additional pass
# (with a more complicated antidiffusive pseudo-velocity
# equivalent to 3/4 passes in the standard MPData)
function mpdata_iterate!(field, grid, scheme::OptimalMPData, pseudo_velocities, Δt, divUc)

    fill_halo_regions!(field)
    launch!(architecture(grid), grid, :xyz, _calculate_optimal_mpdata_velocities!, 
            pseudo_velocities, grid, field, Δt)

    fill_halo_regions!(pseudo_velocities)
    launch!(architecture(grid), grid, :xyz, _mpdata_update_field!, field, 
            scheme, pseudo_velocities, grid, divUc, Δt) 

    return nothing
end

# Standard MPData iterates to cancel subsequent error order. As a rule of thumb,
# the scheme converges at the third/fourth pass
function mpdata_iterate!(field, grid, scheme, pseudo_velocities, Δt, divUc)

    for iter in 1:scheme.iterations
        fill_halo_regions!(field)
        launch!(architecture(grid), grid, :xyz, _calculate_mpdata_velocities!, 
                pseudo_velocities, grid, field, Δt)

        fill_halo_regions!(pseudo_velocities)
        launch!(architecture(grid), grid, :xyz, _mpdata_update_field!, field, scheme, 
                pseudo_velocities, grid, divUc, Δt) 
    end

    return nothing
end

""" 
Pseudo-velocities are calculated as:

uᵖ = abs(u)(1 - abs(u)) A - u v B - u w C
vᵖ = abs(v)(1 - abs(v)) B - u v A - v w C
wᵖ = abs(w)(1 - abs(w)) C - u w A - v w B

where A = Δx / 2ψ ∂x(ψ) is updated between iterations
and   B = Δy / 2ψ ∂y(ψ) is updated between iterations
and   C = Δz / 2ψ ∂z(ψ) is updated between iterations
"""
@kernel function _calculate_mpdata_velocities!(velocities, grid, ψ, Δt)
    i, j, k = @index(Global, NTuple)

    Aᶠᶜᶜ, Bᶠᶜᶜ, Cᶠᶜᶜ, Aᶜᶠᶜ, Bᶜᶠᶜ, Cᶜᶠᶜ, Aᶜᶜᶠ, Bᶜᶜᶠ, Cᶜᶜᶠ = mpdata_auxiliaries(i, j, k, grid, ψ)
    uᵖ, vᵖ, wᵖ = velocities

    ξ, η, ζ = mpdata_pseudo_velocities(i, j, k, grid, Δt, velocities, Aᶠᶜᶜ, Bᶠᶜᶜ, Cᶠᶜᶜ, Aᶜᶠᶜ, Bᶜᶠᶜ, Cᶜᶠᶜ, Aᶜᶜᶠ, Bᶜᶜᶠ, Cᶜᶜᶠ)

    @inbounds begin
        uᵖ[i, j, k] = min(abs(uᵖ[i, j, k]), abs(ξ)) * sign(ξ)
        vᵖ[i, j, k] = min(abs(vᵖ[i, j, k]), abs(η)) * sign(η)
        wᵖ[i, j, k] = min(abs(wᵖ[i, j, k]), abs(ζ)) * sign(ζ)
    end 
end

""" 
Pseudo-velocities are calculated as:

uᵖ = ∑₁∞ abs(uᴾ)(1 - abs(uᴾ)) A - uᴾ vᴾ B - uᵖ wᵖ C
vᵖ = ∑₁∞ abs(vᴾ)(1 - abs(vᴾ)) B - uᴾ vᴾ A - vᵖ wᵖ C
wᵖ = ∑₁∞ abs(wᴾ)(1 - abs(wᴾ)) C - uᴾ wᴾ A - vᵖ wᵖ B

where A = Δx / 2ψ ∂x(ψ) remaines fixed
and   B = Δy / 2ψ ∂y(ψ) remaines fixed
and   C = Δz / 2ψ ∂z(ψ) remaines fixed
"""
@kernel function _calculate_optimal_mpdata_velocities!(velocities, grid, ψ, Δt)
    i, j, k = @index(Global, NTuple)

    uᵖ, vᵖ, wᵖ = velocities
    Aᶠᶜᶜ, Bᶠᶜᶜ, Cᶠᶜᶜ, Aᶜᶠᶜ, Bᶜᶠᶜ, Cᶜᶠᶜ, Aᶜᶜᶠ, Bᶜᶜᶠ, Cᶜᶜᶠ = mpdata_auxiliaries(i, j, k, grid, ψ)

    @inbounds begin
        u_abs = abs(uᵖ[i, j, k])
        v_abs = abs(vᵖ[i, j, k])

        ξ, η, ζ = mpdata_pseudo_velocities(i, j, k, grid, Δt, velocities, Aᶠᶜᶜ, Bᶠᶜᶜ, Cᶠᶜᶜ, Aᶜᶠᶜ, Bᶜᶠᶜ, Cᶜᶠᶜ, Aᶜᶜᶠ, Bᶜᶜᶠ, Cᶜᶜᶠ)

        ξ *= Δt / Δxᶠᶜᶜ(i, j, k, grid)
        η *= Δt / Δyᶜᶠᶜ(i, j, k, grid)  
        ζ *= Δt / Δzᶜᶜᶠ(i, j, k, grid)

        Σˣᵅ, Σˣᵝ, Σˣᵞ = Σᵅᵝᵞ(Aᶠᶜᶜ)
        Σʸᵅ, Σʸᵝ, Σʸᵞ = Σᵅᵝᵞ(Bᶜᶠᶜ)
        Σᶻᵅ, Σᶻᵝ, Σᶻᵞ = Σᵅᵝᵞ(Cᶜᶜᶠ)

        Σˣʸᵃ, Σˣʸᵇ, Σˣʸᶜ = Σᵃᵇᶜ(Aᶠᶜᶜ, Bᶠᶜᶜ)
        Σʸˣᵃ, Σʸˣᵇ, Σʸˣᶜ = Σᵃᵇᶜ(Bᶜᶠᶜ, Aᶜᶠᶜ)
        Σˣᶻᵃ, Σˣᶻᵇ, Σˣᶻᶜ = Σᵃᵇᶜ(Aᶠᶜᶜ, Cᶠᶜᶜ)
        Σᶻˣᵃ, Σᶻˣᵇ, Σᶻˣᶜ = Σᵃᵇᶜ(Cᶜᶜᶠ, Aᶜᶜᶠ)
        Σʸᶻᵃ, Σʸᶻᵇ, Σʸᶻᶜ = Σᵃᵇᶜ(Bᶜᶠᶜ, Cᶜᶠᶜ)
        Σᶻʸᵃ, Σᶻʸᵇ, Σᶻʸᶜ = Σᵃᵇᶜ(Cᶜᶜᶠ, Bᶜᶜᶠ)

        uᵖ[i, j, k] = (Σˣᵅ  * ξ     + Σˣᵝ  * ξ^2     + Σˣᵞ  * ξ^3 + 
                       Σˣʸᵃ * ξ * η + Σˣʸᵇ * ξ^2 * η + Σˣʸᶜ * ξ * η^2 +
                       Σˣᶻᵃ * ξ * ζ + Σˣᶻᵇ * ξ^2 * ζ + Σˣᶻᶜ * ξ * ζ^2) * Δxᶠᶜᶜ(i, j, k, grid) / Δt
        vᵖ[i, j, k] = (Σʸᵅ  * η     + Σʸᵝ  * η^2     + Σʸᵞ  * η^3 + 
                       Σʸˣᵃ * η * ξ + Σʸˣᵇ * η^2 * ξ + Σʸˣᶜ * η * ξ^2 +
                       Σʸᶻᵃ * η * ζ + Σʸᶻᵇ * η^2 * ζ + Σʸᶻᶜ * η * ζ^2) * Δyᶜᶠᶜ(i, j, k, grid) / Δt
        wᵖ[i, j, k] = (Σᶻᵅ  * η     + Σᶻᵝ  * η^2     + Σᶻᵞ  * η^3 + 
                       Σᶻˣᵃ * ζ * ξ + Σᶻˣᵇ * ζ^2 * ξ + Σᶻˣᶜ * ζ * ξ^2 +
                       Σᶻʸᵃ * ζ * η + Σᶻʸᵇ * ζ^2 * η + Σᶻʸᶜ * ζ * η^2) * Δzᶜᶜᶠ(i, j, k, grid) / Δt
        
        uᵖ[i, j, k] = min(u_abs, abs(uᵖ[i, j, k])) * sign(uᵖ[i, j, k])
        vᵖ[i, j, k] = min(v_abs, abs(vᵖ[i, j, k])) * sign(vᵖ[i, j, k])
        wᵖ[i, j, k] = min(w_abs, abs(wᵖ[i, j, k])) * sign(wᵖ[i, j, k])
    end 
end

@inline function Σᵅᵝᵞ(C)

    Cₐ = abs(C)

    d₁ = (1 - Cₐ)
    d₂ = (1 - Cₐ^2)
    d₃ = (1 - Cₐ^3)

    cΣᵅ = abs(d₁) > 0
    cΣᵝ = cΣᵅ & (abs(d₂) > 0)
    cΣᵞ = cΣᵝ & (abs(d₃) > 0)
    Σᵅ = ifelse(cΣᵅ, 1 / d₁,                    0)
    Σᵝ = ifelse(cΣᵝ, - C / (d₁ * d₂),           0)
    Σᵞ = ifelse(cΣᵞ, 2 * Cₐ^3 / (d₁ * d₂ * d₃), 0)

    return Σᵅ, Σᵝ, Σᵞ
end

@inline function Σᵃᵇᶜ(C₁, C₂)

    Cₐ₁ = abs(C₁)
    Cₐ₂ = abs(C₂)

    d₁ = (1 - Cₐ₁)
    d₂ = (1 - Cₐ₁^2)

    d²₁ = (1 - abs(C₁   * C₂))
    d²₂ = (1 - abs(C₁^2 * C₂))
    d²₃ = (1 - abs(C₁   * C₂^2))

    cΣᵃ  = (abs(d₁) > 0) & (abs(d²₁) > 1)
    cΣᵇ  = cΣᵃ & (abs(d²₂) > 0) & (abs(d₂) > 0)
    cΣᶜ  = cΣᵃ & (abs(d²₃) > 0) 
    Σᵃ = ifelse(cΣᵃ, - C₂ / (d₁ * d²₁),                             0)
    Σᵇ = ifelse(cΣᵇ, C₂ * C₁ / (d₁ * d²₂) * (Cₐ₂ / d²₁ + 2C₁ / d₂), 0)
    Σᶜ = ifelse(cΣᶜ, Cₐ₁ * C₂^2 / (d₁ * d²₃ * d²₁),                 0)

    return Σᵃ, Σᵇ, Σᶜ
end

@inline abs_ψ(i, j, k, grid, ψ) = abs(ψ[i, j, k])

@inline function mpdata_auxiliaries(i, j, k, grid, ψ)

    ψ₁ᶠᶜᶜ = 2 * ℑxᶠᵃᵃ(i, j, k, grid, abs_ψ, ψ)
    ψ₁ᶜᶠᶜ = 2 * ℑyᵃᶠᵃ(i, j, k, grid, abs_ψ, ψ)
    ψ₁ᶜᶜᶠ = 2 * ℑzᵃᵃᶠ(i, j, k, grid, abs_ψ, ψ)
    Δψ₁ᶠᶜᶜ = δxᶠᵃᵃ(i, j, k, grid, abs_ψ, ψ)
    Δψ₁ᶜᶠᶜ = δyᵃᶠᵃ(i, j, k, grid, abs_ψ, ψ)
    Δψ₁ᶜᶜᶠ = δzᵃᵃᶠ(i, j, k, grid, abs_ψ, ψ)

    # Calculating A and B
    @inbounds begin
        ψ₂ᶠᶜᶜ = (abs(ψ[i, j+1, k]) + abs(ψ[i-1, j+1, k]) + abs(ψ[i, j-1, k]) + abs(ψ[i-1, j-1, k]))
        ψ₂ᶜᶠᶜ = (abs(ψ[i+1, j, k]) + abs(ψ[i+1, j-1, k]) + abs(ψ[i-1, j, k]) + abs(ψ[i-1, j-1, k]))
        ψ₂ᶜᶜᶠ = (abs(ψ[i, j, k+1]) + abs(ψ[i-1, j, k+1]) + abs(ψ[i, j, k-1]) + abs(ψ[i-1, j, k-1]))

        Δψ₂ᶠᶜᶜ = (abs(ψ[i, j+1, k]) + abs(ψ[i-1, j+1, k]) - abs(ψ[i, j-1, k]) - abs(ψ[i-1, j-1, k]))
        Δψ₂ᶜᶠᶜ = (abs(ψ[i+1, j, k]) + abs(ψ[i+1, j-1, k]) - abs(ψ[i-1, j, k]) - abs(ψ[i-1, j-1, k]))
        Δψ₂ᶜᶜᶠ = (abs(ψ[i, j, k+1]) + abs(ψ[i-1, j, k+1]) - abs(ψ[i, j, k-1]) - abs(ψ[i-1, j, k-1]))

        ψ₃ᶠᶜᶜ = (abs(ψ[i+1, j, k]) + abs(ψ[i+1, j, k-1]) + abs(ψ[i-1, j, k]) + abs(ψ[i-1, j, k-1]))
        ψ₃ᶜᶠᶜ = (abs(ψ[i, j+1, k]) + abs(ψ[i, j+1, k-1]) + abs(ψ[i, j-1, k]) + abs(ψ[i, j-1, k-1]))
        ψ₃ᶜᶜᶠ = (abs(ψ[i, j, k+1]) + abs(ψ[i, j-1, k+1]) + abs(ψ[i, j, k-1]) + abs(ψ[i, j-1, k-1]))

        Δψ₃ᶠᶜᶜ = (abs(ψ[i+1, j, k]) + abs(ψ[i+1, j, k-1]) - abs(ψ[i-1, j, k]) - abs(ψ[i-1, j, k-1]))
        Δψ₃ᶜᶠᶜ = (abs(ψ[i, j+1, k]) + abs(ψ[i, j+1, k-1]) - abs(ψ[i, j-1, k]) - abs(ψ[i, j-1, k-1]))
        Δψ₃ᶜᶜᶠ = (abs(ψ[i, j, k+1]) + abs(ψ[i, j-1, k+1]) - abs(ψ[i, j, k-1]) - abs(ψ[i, j-1, k-1]))

        Aᶠᶜᶜ = ifelse(abs(ψ₁ᶠᶜᶜ) > 0, Δψ₁ᶠᶜᶜ / ψ₁ᶠᶜᶜ, 0)
        Bᶠᶜᶜ = ifelse(abs(ψ₂ᶠᶜᶜ) > 0, Δψ₂ᶠᶜᶜ / ψ₂ᶠᶜᶜ, 0)
        Cᶠᶜᶜ = ifelse(abs(ψ₃ᶠᶜᶜ) > 0, Δψ₃ᶠᶜᶜ / ψ₃ᶠᶜᶜ, 0)

        Aᶜᶠᶜ = ifelse(abs(ψ₂ᶜᶠᶜ) > 0, Δψ₂ᶜᶠᶜ / ψ₂ᶜᶠᶜ, 0)
        Bᶜᶠᶜ = ifelse(abs(ψ₁ᶜᶠᶜ) > 0, Δψ₁ᶜᶠᶜ / ψ₁ᶜᶠᶜ, 0)
        Cᶜᶠᶜ = ifelse(abs(ψ₃ᶜᶠᶜ) > 0, Δψ₃ᶜᶠᶜ / ψ₃ᶜᶠᶜ, 0)

        Aᶜᶜᶠ = ifelse(abs(ψ₂ᶜᶜᶠ) > 0, Δψ₂ᶜᶜᶠ / ψ₂ᶜᶜᶠ, 0)
        Bᶜᶜᶠ = ifelse(abs(ψ₃ᶜᶜᶠ) > 0, Δψ₃ᶜᶜᶠ / ψ₃ᶜᶜᶠ, 0)
        Cᶜᶜᶠ = ifelse(abs(ψ₁ᶜᶜᶠ) > 0, Δψ₁ᶜᶜᶠ / ψ₁ᶜᶜᶠ, 0)        
    end

    return Aᶠᶜᶜ, Bᶠᶜᶜ, Cᶠᶜᶜ, Aᶜᶠᶜ, Bᶜᶠᶜ, Cᶜᶠᶜ, Aᶜᶜᶠ, Bᶜᶜᶠ, Cᶜᶜᶠ
end

@inline function mpdata_pseudo_velocities(i, j, k, grid, Δt, U, Aᶠᶜᶜ, Bᶠᶜᶜ, Cᶠᶜᶜ, Aᶜᶠᶜ, Bᶜᶠᶜ, Cᶜᶠᶜ, Aᶜᶜᶠ, Bᶜᶜᶠ, Cᶜᶜᶠ)

    uᵖ, vᵖ, wᵖ = U

    u_abs = abs(uᵖ[i, j, k])
    v_abs = abs(vᵖ[i, j, k])
    w_abs = abs(wᵖ[i, j, k])

    u̅ᶠᶜᶜ = abs(uᵖ[i, j, k]) * Δt / Δxᶠᶜᶜ(i, j, k, grid)
    v̅ᶜᶠᶜ = abs(vᵖ[i, j, k]) * Δt / Δyᶜᶠᶜ(i, j, k, grid)  
    w̅ᶜᶜᶠ = abs(wᵖ[i, j, k]) * Δt / Δzᶜᶜᶠ(i, j, k, grid) 

    u̅ᶜᶠᶜ = ℑxyᶜᶠᵃ(i, j, k, grid, uᵖ) * Δt / Δxᶜᶠᶜ(i, j, k, grid)
    u̅ᶜᶜᶠ = ℑxzᶜᵃᶠ(i, j, k, grid, uᵖ) * Δt / Δxᶜᶜᶠ(i, j, k, grid)
    v̅ᶠᶜᶜ = ℑxyᶠᶜᵃ(i, j, k, grid, vᵖ) * Δt / Δyᶠᶜᶜ(i, j, k, grid) 
    v̅ᶜᶜᶠ = ℑyzᶜᵃᶠ(i, j, k, grid, vᵖ) * Δt / Δyᶜᶜᶠ(i, j, k, grid)  
    w̅ᶠᶜᶜ = ℑxzᶠᵃᶜ(i, j, k, grid, wᵖ) * Δt / Δzᶠᶜᶜ(i, j, k, grid)  
    w̅ᶜᶠᶜ = ℑyzᵃᶠᶜ(i, j, k, grid, wᵖ) * Δt / Δzᶜᶠᶜ(i, j, k, grid)  

    ξ = u_abs * (1 - u̅ᶠᶜᶜ) * Aᶠᶜᶜ - uᵖ[i, j, k] * v̅ᶠᶜᶜ * Bᶠᶜᶜ - uᵖ[i, j, k] * w̅ᶠᶜᶜ * Cᶠᶜᶜ
    η = v_abs * (1 - v̅ᶜᶠᶜ) * Bᶜᶠᶜ - vᵖ[i, j, k] * u̅ᶜᶠᶜ * Aᶜᶠᶜ - vᵖ[i, j, k] * w̅ᶜᶠᶜ * Cᶜᶠᶜ
    ζ = w_abs * (1 - w̅ᶜᶜᶠ) * Cᶜᶜᶠ - wᵖ[i, j, k] * u̅ᶜᶜᶠ * Aᶜᶜᶠ - wᵖ[i, j, k] * v̅ᶜᶜᶠ * Bᶜᶜᶠ

    return ξ, η, ζ
end

# The actual MPData correction
@kernel function _mpdata_update_field!(c, scheme, pseudo_velocities, grid, divUc, Δt)
    i, j, k = @index(Global, NTuple)

    ∇uc = divUc(i, j, k, grid, scheme, pseudo_velocities, c)
    @inbounds c[i, j, k] -= Δt * ∇uc
end

# Different vertical advection for `PartialMPData`
@inline function div_𝐯u(i, j, k, grid, advection::PartialMPData, U, u)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uu, advection, U[1], u) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vu, advection, U[2], u) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wu, advection.vertical_advection, U[3], u))
end

@inline function div_𝐯v(i, j, k, grid, advection::PartialMPData, U, v)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uv, advection, U[1], v) +
                                    δyᵃᶠᵃ(i, j, k, grid, _advective_momentum_flux_Vv, advection, U[2], v) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_momentum_flux_Wv, advection.vertical_advection, U[3], v))
end

@inline function div_𝐯w(i, j, k, grid, advection::PartialMPData, U, w)
    return 1/Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_momentum_flux_Uw, advection, U[1], w) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_momentum_flux_Vw, advection, U[2], w) +
                                    δzᵃᵃᶠ(i, j, k, grid, _advective_momentum_flux_Ww, advection.vertical_advection, U[3], w))
end

@inline function div_Uc(i, j, k, grid, advection::PartialMPData, U, c)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, advection, U.u, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, advection, U.v, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, advection.vertical_advection, U.w, c))
end