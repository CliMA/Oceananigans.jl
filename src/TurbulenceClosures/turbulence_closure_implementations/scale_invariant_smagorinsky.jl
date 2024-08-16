#####
##### In this version of the Smagorinsky closure, the coefficient is dynamically calculated but it's assumed to be invariant
##### with scale. Hence the name Scale-Invariant Smagorinsky. This a type of "dynamic Smagorinsky" closures.
#####
using Oceananigans.Operators: volume
using Statistics: mean!

abstract type AbstractAveragingProcedure end
struct DirectionalAveraging{D} <: AbstractAveragingProcedure
    dims :: D
end

Base.summary(averaging::DirectionalAveraging) = string("DirectionalAveraging over directions $(averaging.dims)")
Base.show(io::IO, averaging::DirectionalAveraging) = print(io, summary(averaging))


struct ScaleInvariantSmagorinsky{TD, FT, P} <: AbstractScalarDiffusivity{TD, ThreeDimensionalFormulation, 2}
    averaging :: AbstractAveragingProcedure
    Pr :: P
    update_frequency :: Integer

    function ScaleInvariantSmagorinsky{TD, FT}(averaging, Pr; update_frequency = 5) where {TD, FT}
        Pr = convert_diffusivity(FT, Pr; discrete_form=false)
        P = typeof(Pr)
        return new{TD, FT, P}(averaging, Pr, update_frequency)
    end
end

@inline viscosity(::ScaleInvariantSmagorinsky, K) = K.νₑ
@inline diffusivity(closure::ScaleInvariantSmagorinsky, K, ::Val{id}) where id = K.νₑ / closure.Pr[id]

"""
    ScaleInvariantSmagorinsky([time_discretization::TD = ExplicitTimeDiscretization(), FT=Float64;] averaging=1.0, Pr=1.0)
"""
function ScaleInvariantSmagorinsky(time_discretization::TD = ExplicitTimeDiscretization(), FT=Float64; averaging=DirectionalAveraging(Colon()), Pr=1.0) where TD
    averaging = (averaging isa AbstractAveragingProcedure) ? averaging : DirectionalAveraging(averaging)
    return ScaleInvariantSmagorinsky{TD, FT}(averaging, Pr)
end


ScaleInvariantSmagorinsky(FT::DataType; kwargs...) = ScaleInvariantSmagorinsky(ExplicitTimeDiscretization(), FT; kwargs...)

function with_tracers(tracers, closure::ScaleInvariantSmagorinsky{TD, FT}) where {TD, FT}
    Pr = tracer_diffusivities(tracers, closure.Pr)
    return ScaleInvariantSmagorinsky{TD, FT}(closure.averaging, Pr)
end

function LᵢⱼMᵢⱼ_ccc(i, j, k, grid, u, v, w)
    Sᶜᶜᶜ = √(ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w))
    S̄ᶜᶜᶜ = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w))
    return (      L₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w) * M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Sᶜᶜᶜ, S̄ᶜᶜᶜ)
            +     L₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w) * M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Sᶜᶜᶜ, S̄ᶜᶜᶜ)
            +     L₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w) * M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Sᶜᶜᶜ, S̄ᶜᶜᶜ)
            + 2 * L₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w) * M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1) 
            + 2 * L₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w) * M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1) 
            + 2 * L₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w) * M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1) )
end

function MᵢⱼMᵢⱼ_ccc(i, j, k, grid, u, v, w)
    Sᶜᶜᶜ = √(ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w))
    S̄ᶜᶜᶜ = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w))
    return (      M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Sᶜᶜᶜ, S̄ᶜᶜᶜ)^2
            +     M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Sᶜᶜᶜ, S̄ᶜᶜᶜ)^2
            +     M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Sᶜᶜᶜ, S̄ᶜᶜᶜ)^2
            + 2 * M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2
            + 2 * M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2
            + 2 * M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2)
end


@kernel function _compute_scale_invariant_smagorinsky_viscosity!(νₑ, LM_avg, MM_avg, grid, closure, buoyancy, velocities, tracers)
    i, j, k = @index(Global, NTuple)

    # Strain tensor dot product
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, velocities.u, velocities.v, velocities.w)

    cₛ² = @inbounds max(LM_avg[i, j, k] / MM_avg[i, j, k], 0)
    @inbounds νₑ[i, j, k] = cₛ² * (Δᶠ(i, j, k, grid))^2 * sqrt(2Σ²)
end

function compute_diffusivities!(diffusivity_fields, closure::ScaleInvariantSmagorinsky, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    buoyancy = model.buoyancy
    velocities = model.velocities
    tracers = model.tracers

    LM_op = Average(KernelFunctionOperation{Center, Center, Center}(LᵢⱼMᵢⱼ_ccc, grid, model.velocities...))
    MM_op = Average(KernelFunctionOperation{Center, Center, Center}(MᵢⱼMᵢⱼ_ccc, grid, model.velocities...))

    if model.clock.iteration % closure.update_frequency == 0
        mean!(diffusivity_fields.LM_avg, LM_op.operand)
        mean!(diffusivity_fields.MM_avg, MM_op.operand)

        launch!(arch, grid, parameters, _compute_scale_invariant_smagorinsky_viscosity!,
                diffusivity_fields.νₑ, diffusivity_fields.LM_avg, diffusivity_fields.MM_avg, grid, closure, buoyancy, velocities, tracers)
    end

    return nothing
end

@inline κᶠᶜᶜ(i, j, k, grid, closure::ScaleInvariantSmagorinsky, K, ::Val{id}, args...) where id = ℑxᶠᵃᵃ(i, j, k, grid, K.νₑ) / closure.Pr[id]
@inline κᶜᶠᶜ(i, j, k, grid, closure::ScaleInvariantSmagorinsky, K, ::Val{id}, args...) where id = ℑyᵃᶠᵃ(i, j, k, grid, K.νₑ) / closure.Pr[id]
@inline κᶜᶜᶠ(i, j, k, grid, closure::ScaleInvariantSmagorinsky, K, ::Val{id}, args...) where id = ℑzᵃᵃᶠ(i, j, k, grid, K.νₑ) / closure.Pr[id]


#####
##### Filters
#####

# TODO: Generalize filter to stretched directions
AG = AbstractGrid
@inline ℱx²ᵟ(i, j, k, grid::AG{FT}, ϕ) where FT = @inbounds FT(0.5) * ϕ[i, j, k] + FT(0.25) * (ϕ[i-1, j, k] + ϕ[i+1, j,  k])
@inline ℱy²ᵟ(i, j, k, grid::AG{FT}, ϕ) where FT = @inbounds FT(0.5) * ϕ[i, j, k] + FT(0.25) * (ϕ[i, j-1, k] + ϕ[i,  j+1, k])
@inline ℱz²ᵟ(i, j, k, grid::AG{FT}, ϕ) where FT = @inbounds FT(0.5) * ϕ[i, j, k] + FT(0.25) * (ϕ[i, j, k-1] + ϕ[i,  j, k+1])

@inline ℱx²ᵟ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * f(i, j, k, grid, args...) + FT(0.25) * (f(i-1, j, k, grid, args...) + f(i+1, j, k, grid, args...))
@inline ℱy²ᵟ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * f(i, j, k, grid, args...) + FT(0.25) * (f(i, j-1, k, grid, args...) + f(i, j+1, k, grid, args...))
@inline ℱz²ᵟ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * f(i, j, k, grid, args...) + FT(0.25) * (f(i, j, k-1, grid, args...) + f(i, j, k+1, grid, args...))

@inline ℱxy²ᵟ(i, j, k, grid, f, args...)  = ℱy²ᵟ(i, j, k, grid, ℱx²ᵟ, f, args...)
@inline ℱyz²ᵟ(i, j, k, grid, f, args...)  = ℱz²ᵟ(i, j, k, grid, ℱy²ᵟ, f, args...)
@inline ℱxz²ᵟ(i, j, k, grid, f, args...)  = ℱz²ᵟ(i, j, k, grid, ℱz²ᵟ, f, args...)
@inline ℱ²ᵟ(i, j, k, grid, f, args...) = ℱz²ᵟ(i, j, k, grid, ℱxy²ᵟ, f, args...)


#####
##### Velocity gradients
#####

# Diagonal
@inline ∂x_ū(i, j, k, grid, u) = ∂xᶜᶜᶜ(i, j, k, grid, ℱ²ᵟ, u)
@inline ∂y_v̄(i, j, k, grid, v) = ∂yᶜᶜᶜ(i, j, k, grid, ℱ²ᵟ, v)
@inline ∂z_w̄(i, j, k, grid, w) = ∂zᶜᶜᶜ(i, j, k, grid, ℱ²ᵟ, w)

# Off-diagonal
@inline ∂x_v̄(i, j, k, grid, v) = ∂xᶠᶠᶜ(i, j, k, grid, ℱ²ᵟ, v)
@inline ∂x_w̄(i, j, k, grid, w) = ∂xᶠᶜᶜ(i, j, k, grid, ℱ²ᵟ, w)

@inline ∂y_ū(i, j, k, grid, u) = ∂yᶠᶠᶜ(i, j, k, grid, ℱ²ᵟ, u)
@inline ∂y_w̄(i, j, k, grid, w) = ∂yᶜᶠᶜ(i, j, k, grid, ℱ²ᵟ, w)

@inline ∂z_ū(i, j, k, grid, u) = ∂zᶠᶜᶠ(i, j, k, grid, ℱ²ᵟ, u)
@inline ∂z_v̄(i, j, k, grid, v) = ∂zᶜᶠᶠ(i, j, k, grid, ℱ²ᵟ, v)

#####
##### Strain components
#####

# ccc strain components
@inline Σ̄₁₁(i, j, k, grid, u) = ∂x_ū(i, j, k, grid, u)
@inline Σ̄₂₂(i, j, k, grid, v) = ∂y_v̄(i, j, k, grid, v)
@inline Σ̄₃₃(i, j, k, grid, w) = ∂z_w̄(i, j, k, grid, w)

@inline tr_Σ̄(i, j, k, grid, u, v, w) = Σ̄₁₁(i, j, k, grid, u) + Σ̄₂₂(i, j, k, grid, v) + Σ̄₃₃(i, j, k, grid, w)
@inline tr_Σ̄²(ijk...) = Σ̄₁₁(ijk...)^2 + Σ̄₂₂(ijk...)^2 + Σ̄₃₃(ijk...)^2

# ffc
@inline Σ̄₁₂(i, j, k, grid::AG{FT}, u, v) where FT = FT(0.5) * (∂y_ū(i, j, k, grid, u) + ∂x_v̄(i, j, k, grid, v))
@inline Σ̄₁₂²(i, j, k, grid, u, v) = Σ̄₁₂(i, j, k, grid, u, v)^2


# fcf
@inline Σ̄₁₃(i, j, k, grid::AG{FT}, u, w) where FT = FT(0.5) * (∂z_ū(i, j, k, grid, u) + ∂x_w̄(i, j, k, grid, w))
@inline Σ̄₁₃²(i, j, k, grid, u, w) = Σ̄₁₃(i, j, k, grid, u, w)^2

# cff
@inline Σ̄₂₃(i, j, k, grid::AG{FT}, v, w) where FT = FT(0.5) * (∂z_v̄(i, j, k, grid, v) + ∂y_w̄(i, j, k, grid, w))
@inline Σ̄₂₃²(i, j, k, grid, v, w) = Σ̄₂₃(i, j, k, grid, v, w)^2



@inline Σ̄₁₁(i, j, k, grid, u, v, w) = Σ̄₁₁(i, j, k, grid, u)
@inline Σ̄₂₂(i, j, k, grid, u, v, w) = Σ̄₂₂(i, j, k, grid, v)
@inline Σ̄₃₃(i, j, k, grid, u, v, w) = Σ̄₃₃(i, j, k, grid, w)

@inline Σ̄₁₂(i, j, k, grid, u, v, w) = Σ̄₁₂(i, j, k, grid, u, v)
@inline Σ̄₁₃(i, j, k, grid, u, v, w) = Σ̄₁₃(i, j, k, grid, u, w)
@inline Σ̄₂₃(i, j, k, grid, u, v, w) = Σ̄₂₃(i, j, k, grid, v, w)


@inline Σ̄₁₂²(i, j, k, grid, u, v, w) = Σ̄₁₂²(i, j, k, grid, u, v)
@inline Σ̄₁₃²(i, j, k, grid, u, v, w) = Σ̄₁₃²(i, j, k, grid, u, w)
@inline Σ̄₂₃²(i, j, k, grid, u, v, w) = Σ̄₂₃²(i, j, k, grid, v, w)




#####
##### Double dot product of strain on cell edges
#####

"Return the double dot product of strain at `ccc` on a 2δ test grid."
@inline function Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
    return (tr_Σ̄²(i, j, k, grid, u, v, w)
            + 2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ̄₁₂², u, v, w)
            + 2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ̄₁₃², u, v, w)
            + 2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ̄₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `ffc`."
@inline function Σ̄ᵢⱼΣ̄ᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w)
    return (
                  ℑxyᶠᶠᵃ(i, j, k, grid, tr_Σ̄², u, v, w)
            + 2 *   Σ̄₁₂²(i, j, k, grid, u, v, w)
            + 2 * ℑyzᵃᶠᶜ(i, j, k, grid, Σ̄₁₃², u, v, w)
            + 2 * ℑxzᶠᵃᶜ(i, j, k, grid, Σ̄₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `fcf`."
@inline function Σ̄ᵢⱼΣ̄ᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w)
    return (
                  ℑxzᶠᵃᶠ(i, j, k, grid, tr_Σ̄², u, v, w)
            + 2 * ℑyzᵃᶜᶠ(i, j, k, grid, Σ̄₁₂², u, v, w)
            + 2 *   Σ̄₁₃²(i, j, k, grid, u, v, w)
            + 2 * ℑxyᶠᶜᵃ(i, j, k, grid, Σ̄₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `cff`."
@inline function Σ̄ᵢⱼΣ̄ᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w)
    return (
                  ℑyzᵃᶠᶠ(i, j, k, grid, tr_Σ̄², u, v, w)
            + 2 * ℑxzᶜᵃᶠ(i, j, k, grid, Σ̄₁₂², u, v, w)
            + 2 * ℑxyᶜᶠᵃ(i, j, k, grid, Σ̄₁₃², u, v, w)
            + 2 *   Σ̄₂₃²(i, j, k, grid, u, v, w)
            )
end

# Here the notation ⟨A⟩ is equivalent to Ā: a filter of size 2Δᶠ, where Δᶠ is the grid scale.

@inline SS₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, Sᶜᶜᶜ) = Sᶜᶜᶜ * Σ₁₁(i, j, k, grid, u, v, w) # ccc
@inline SS₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, Sᶜᶜᶜ) = Sᶜᶜᶜ * Σ₂₂(i, j, k, grid, u, v, w) # ccc
@inline SS₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, Sᶜᶜᶜ) = Sᶜᶜᶜ * Σ₃₃(i, j, k, grid, u, v, w) # ccc

@inline SS₁₂ᶠᶠᶜ(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w)) * Σ₁₂(i, j, k, grid, u, v, w) # ffc
@inline SS₁₃ᶠᶜᶠ(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w)) * Σ₁₃(i, j, k, grid, u, v, w) # fcf
@inline SS₂₃ᶜᶠᶠ(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w)) * Σ₂₃(i, j, k, grid, u, v, w) # cff

@inline var"⟨|S|S₁₁⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Sᶜᶜᶜ) = ℱ²ᵟ(i, j, k, grid, SS₁₁ᶜᶜᶜ, u, v, w, Sᶜᶜᶜ)
@inline var"⟨|S|S₂₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Sᶜᶜᶜ) = ℱ²ᵟ(i, j, k, grid, SS₂₂ᶜᶜᶜ, u, v, w, Sᶜᶜᶜ)
@inline var"⟨|S|S₃₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Sᶜᶜᶜ) = ℱ²ᵟ(i, j, k, grid, SS₃₃ᶜᶜᶜ, u, v, w, Sᶜᶜᶜ)

@inline var"⟨|S|S₁₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℑxyᶜᶜᵃ(i, j, k, grid, ℱ²ᵟ, SS₁₂ᶠᶠᶜ, u, v, w)
@inline var"⟨|S|S₁₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℑxzᶜᵃᶜ(i, j, k, grid, ℱ²ᵟ, SS₁₃ᶠᶜᶠ, u, v, w)
@inline var"⟨|S|S₂₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℑyzᵃᶜᶜ(i, j, k, grid, ℱ²ᵟ, SS₂₃ᶜᶠᶠ, u, v, w)

@inline S̄S̄₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, S̄ᶜᶜᶜ) = S̄ᶜᶜᶜ * Σ̄₁₁(i, j, k, grid, u, v, w) # ccc
@inline S̄S̄₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, S̄ᶜᶜᶜ) = S̄ᶜᶜᶜ * Σ̄₂₂(i, j, k, grid, u, v, w) # ccc
@inline S̄S̄₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, S̄ᶜᶜᶜ) = S̄ᶜᶜᶜ * Σ̄₃₃(i, j, k, grid, u, v, w) # ccc

@inline S̄S̄₁₂ᶠᶠᶜ(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w)) * Σ̄₁₂(i, j, k, grid, u, v, w) # ffc
@inline S̄S̄₁₃ᶠᶜᶠ(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w)) * Σ̄₁₃(i, j, k, grid, u, v, w) # fcf
@inline S̄S̄₂₃ᶜᶠᶠ(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w)) * Σ̄₂₃(i, j, k, grid, u, v, w) # cff

@inline S̄S̄₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxyᶜᶜᵃ(i, j, k, grid, S̄S̄₁₂ᶠᶠᶜ, u, v, w)
@inline S̄S̄₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxzᶜᵃᶜ(i, j, k, grid, S̄S̄₁₃ᶠᶜᶠ, u, v, w)
@inline S̄S̄₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyzᵃᶜᶜ(i, j, k, grid, S̄S̄₂₃ᶜᶠᶠ, u, v, w)


@inline Δᶠ(i, j, k, grid) = ∛volume(i, j, k, grid, Center(), Center(), Center())
@inline M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β, Sᶜᶜᶜ, S̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨|S|S₁₁⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Sᶜᶜᶜ) - α^2*β * S̄S̄₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, S̄ᶜᶜᶜ))
@inline M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β, Sᶜᶜᶜ, S̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨|S|S₂₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Sᶜᶜᶜ) - α^2*β * S̄S̄₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, S̄ᶜᶜᶜ))
@inline M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β, Sᶜᶜᶜ, S̄ᶜᶜᶜ) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨|S|S₃₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w, Sᶜᶜᶜ) - α^2*β * S̄S̄₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, S̄ᶜᶜᶜ))

@inline M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨|S|S₁₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * S̄S̄₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w))
@inline M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨|S|S₁₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * S̄S̄₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w))
@inline M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨|S|S₂₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * S̄S̄₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w))



@inline ϕψ(i, j, k, grid, ϕ, ψ) = @inbounds ϕ[i, j, k] * ψ[i, j, k]
@inline u₁u₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ϕψ, u, u)
@inline u₂u₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, ϕψ, v, v)
@inline u₃u₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑzᵃᵃᶜ(i, j, k, grid, ϕψ, w, w)

@inline ϕ̄ψ̄(i, j, k, grid, ϕ, ψ) = ℱ²ᵟ(i, j, k, grid, ϕ) * ℱ²ᵟ(i, j, k, grid, ψ)
@inline ū₁ū₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ϕ̄ψ̄, u, u)
@inline ū₂ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ϕ̄ψ̄, v, v)
@inline ū₃ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ϕ̄ψ̄, w, w)

@inline u₁u₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, u) * ℑyᵃᶜᵃ(i, j, k, grid, v)
@inline u₁u₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, u) * ℑzᵃᵃᶜ(i, j, k, grid, w)
@inline u₂u₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, v) * ℑzᵃᵃᶜ(i, j, k, grid, w)

@inline ū₁ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ℱ²ᵟ, u) * ℑyᵃᶜᵃ(i, j, k, grid, ℱ²ᵟ, v)
@inline ū₁ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑxᶜᵃᵃ(i, j, k, grid, ℱ²ᵟ, u) * ℑzᵃᵃᶜ(i, j, k, grid, ℱ²ᵟ, w)
@inline ū₂ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℑyᵃᶜᵃ(i, j, k, grid, ℱ²ᵟ, v) * ℑzᵃᵃᶜ(i, j, k, grid, ℱ²ᵟ, w)

@inline L₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₁u₁ᶜᶜᶜ, u, v, w) - ū₁ū₁ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₂u₂ᶜᶜᶜ, u, v, w) - ū₂ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₃u₃ᶜᶜᶜ, u, v, w) - ū₃ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)

@inline L₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₁u₂ᶜᶜᶜ, u, v, w) - ū₁ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₁u₃ᶜᶜᶜ, u, v, w) - ū₁ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)
@inline L₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₂u₃ᶜᶜᶜ, u, v, w) - ū₂ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)


Base.summary(closure::ScaleInvariantSmagorinsky) = string("ScaleInvariantSmagorinsky: averaging=$(closure.averaging), Pr=$(closure.Pr), update_frequency=$(closure.update_frequency)")
Base.show(io::IO, closure::ScaleInvariantSmagorinsky) = print(io, summary(closure))

#####
##### For closures that only require an eddy viscosity νₑ field.
#####

directionally_averaged_field(grid, ::Val{DirectionalAveraging(1)})         = Field{Nothing, Center,  Center }(grid)
directionally_averaged_field(grid, ::Val{DirectionalAveraging(2)})         = Field{Center,  Nothing, Center }(grid)
directionally_averaged_field(grid, ::Val{DirectionalAveraging(3)})         = Field{Center,  Center,  Nothing}(grid)
directionally_averaged_field(grid, ::Val{DirectionalAveraging((1, 2))})    = Field{Nothing, Nothing, Center }(grid)
directionally_averaged_field(grid, ::Val{DirectionalAveraging((1, 3))})    = Field{Nothing, Center,  Nothing}(grid)
directionally_averaged_field(grid, ::Val{DirectionalAveraging((2, 3))})    = Field{Center,  Nothing, Nothing}(grid)
directionally_averaged_field(grid, ::Val{DirectionalAveraging((1, 2, 3))}) = Field{Nothing, Nothing, Nothing}(grid)
directionally_averaged_field(grid, ::Val{DirectionalAveraging(Colon())})   = Field{Nothing, Nothing, Nothing}(grid)
directionally_averaged_field(grid, ::Any)                                  = Field{Center,  Center,  Center}(grid)


function DiffusivityFields(grid, tracer_names, bcs, closure::ScaleInvariantSmagorinsky)

    default_eddy_viscosity_bcs = (; νₑ = FieldBoundaryConditions(grid, (Center, Center, Center)))
    bcs = merge(default_eddy_viscosity_bcs, bcs)
    νₑ = CenterField(grid, boundary_conditions=bcs.νₑ)

    LM_avg = directionally_averaged_field(grid, Val(closure.averaging))
    MM_avg = directionally_averaged_field(grid, Val(closure.averaging))

    return (; νₑ, LM_avg, MM_avg)
end
