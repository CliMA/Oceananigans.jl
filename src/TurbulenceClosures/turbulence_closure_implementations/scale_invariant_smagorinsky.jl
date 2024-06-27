#####
##### The turbulence closure proposed by Smagorinsky and Lilly.
##### We also call this 'Constant Smagorinsky'.
#####
using Oceananigans.Operators: volume

abstract type AveragingProcedure end
abstract type DirectionalAveraging <: AveragingProcedure end
struct       XDirectionalAveraging <: DirectionalAveraging end
struct      XYDirectionalAveraging <: DirectionalAveraging end
struct      XZDirectionalAveraging <: DirectionalAveraging end
struct      YZDirectionalAveraging <: DirectionalAveraging end
struct     XYZDirectionalAveraging <: DirectionalAveraging end

struct ScaleInvariantSmagorinsky{TD, FT, P} <: AbstractScalarDiffusivity{TD, ThreeDimensionalFormulation, 2}
    C #:: FT
    averaging :: FT
    Pr :: P

    function ScaleInvariantSmagorinsky{TD, FT}(C, averaging, Pr) where {TD, FT}
        Pr = convert_diffusivity(FT, Pr; discrete_form=false)
        P = typeof(Pr)
        return new{TD, FT, P}(C, averaging, Pr)
    end
end

@inline viscosity(::ScaleInvariantSmagorinsky, K) = K.νₑ
@inline diffusivity(closure::ScaleInvariantSmagorinsky, K, ::Val{id}) where id = K.νₑ / closure.Pr[id]

"""
    ScaleInvariantSmagorinsky([time_discretization::TD = ExplicitTimeDiscretization(), FT=Float64;] C=0.16, averaging=1.0, Pr=1.0)
"""
ScaleInvariantSmagorinsky(time_discretization::TD = ExplicitTimeDiscretization(), FT=Float64; C=0.16, averaging=1.0, Pr=1.0) where TD =
        ScaleInvariantSmagorinsky{TD, FT}(C, averaging, Pr)

ScaleInvariantSmagorinsky(FT::DataType; kwargs...) = ScaleInvariantSmagorinsky(ExplicitTimeDiscretization(), FT; kwargs...)

function with_tracers(tracers, closure::ScaleInvariantSmagorinsky{TD, FT}) where {TD, FT}
    Pr = tracer_diffusivities(tracers, closure.Pr)
    return ScaleInvariantSmagorinsky{TD, FT}(closure.C, closure.averaging, Pr)
end

function LᵢⱼMᵢⱼ_ccc(i, j, k, grid, u, v, w)
    return (      L₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w) * M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)
            +     L₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w) * M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)
            +     L₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w) * M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)
            + 2 * L₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w) * M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1) 
            + 2 * L₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w) * M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1) 
            + 2 * L₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w) * M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1) )
end

function MᵢⱼMᵢⱼ_ccc(i, j, k, grid, u, v, w)
    return (      M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2
            +     M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2
            +     M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2
            + 2 * M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2
            + 2 * M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2
            + 2 * M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2)
end


@kernel function compute_LL_MM!(LM, MM, grid, closure, velocities)
    i, j, k = @index(Global, NTuple)
    @inbounds LM[i, j, k] = LᵢⱼMᵢⱼ_ccc(i, j, k, grid, velocities...)
    @inbounds MM[i, j, k] = MᵢⱼMᵢⱼ_ccc(i, j, k, grid, velocities...)
end

@kernel function _compute_smagorinsky_viscosity!(νₑ, grid, closure, buoyancy, velocities, tracers)
    i, j, k = @index(Global, NTuple)

    # Strain tensor dot product
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, velocities.u, velocities.v, velocities.w)

    C = 0.16

    @inbounds νₑ[i, j, k] = (C * Δᶠ(i, j, k, grid))^2 * sqrt(2Σ²)
end

function compute_diffusivities!(diffusivity_fields, closure::ScaleInvariantSmagorinsky, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    buoyancy = model.buoyancy
    velocities = model.velocities
    tracers = model.tracers

    launch!(arch, grid, parameters, compute_LL_MM!,
            diffusivity_fields.LM, diffusivity_fields.MM, grid, closure, velocities)

    launch!(arch, grid, parameters, _compute_smagorinsky_viscosity!,
            diffusivity_fields.νₑ, grid, closure, buoyancy, velocities, tracers)

    return nothing
end

@inline κᶠᶜᶜ(i, j, k, grid, closure::ScaleInvariantSmagorinsky, K, ::Val{id}, args...) where id = ℑxᶠᵃᵃ(i, j, k, grid, K.νₑ) / closure.Pr[id]
@inline κᶜᶠᶜ(i, j, k, grid, closure::ScaleInvariantSmagorinsky, K, ::Val{id}, args...) where id = ℑyᵃᶠᵃ(i, j, k, grid, K.νₑ) / closure.Pr[id]
@inline κᶜᶜᶠ(i, j, k, grid, closure::ScaleInvariantSmagorinsky, K, ::Val{id}, args...) where id = ℑzᵃᵃᶠ(i, j, k, grid, K.νₑ) / closure.Pr[id]


#####
##### Filters
#####
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
@inline tr_Σ²(ijk...) = Σ₁₁(ijk...)^2 +  Σ₂₂(ijk...)^2 +  Σ₃₃(ijk...)^2

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



@inline var"|S|S₁₁ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)) * Σ₁₁(i, j, k, grid, u, v, w) # ccc
@inline var"|S|S₂₂ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)) * Σ₂₂(i, j, k, grid, u, v, w) # ccc
@inline var"|S|S₃₃ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)) * Σ₃₃(i, j, k, grid, u, v, w) # ccc

@inline var"|S|S₁₂ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w)) * Σ₁₂(i, j, k, grid, u, v, w) # ffc
@inline var"|S|S₁₃ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w)) * Σ₁₃(i, j, k, grid, u, v, w) # fcf
@inline var"|S|S₂₃ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w)) * Σ₂₃(i, j, k, grid, u, v, w) # cff

@inline var"⟨|S|S₁₁⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, var"|S|S₁₁ᶜᶜᶜ", u, v, w)
@inline var"⟨|S|S₂₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, var"|S|S₂₂ᶜᶜᶜ", u, v, w)
@inline var"⟨|S|S₃₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, var"|S|S₃₃ᶜᶜᶜ", u, v, w)

@inline var"⟨|S|S₁₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℑxyᶜᶜᵃ(i, j, k, grid, ℱ²ᵟ, var"|S|S₁₂ᶜᶜᶜ", u, v, w)
@inline var"⟨|S|S₁₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℑxzᶜᵃᶜ(i, j, k, grid, ℱ²ᵟ, var"|S|S₁₃ᶜᶜᶜ", u, v, w)
@inline var"⟨|S|S₂₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℑyzᵃᶜᶜ(i, j, k, grid, ℱ²ᵟ, var"|S|S₂₃ᶜᶜᶜ", u, v, w)

@inline var"|S̄|S̄₁₁ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)) * Σ̄₁₁(i, j, k, grid, u, v, w) # ccc
@inline var"|S̄|S̄₂₂ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)) * Σ̄₂₂(i, j, k, grid, u, v, w) # ccc
@inline var"|S̄|S̄₃₃ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)) * Σ̄₃₃(i, j, k, grid, u, v, w) # ccc

@inline var"|S̄|S̄₁₂ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w)) * Σ̄₁₂(i, j, k, grid, u, v, w) # ffc
@inline var"|S̄|S̄₁₃ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w)) * Σ̄₁₃(i, j, k, grid, u, v, w) # fcf
@inline var"|S̄|S̄₂₃ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w)) * Σ̄₂₃(i, j, k, grid, u, v, w) # cff


@inline Δᶠ(i, j, k, grid) = ∛volume(i, j, k, grid, Center(), Center(), Center())
@inline M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨|S|S₁₁⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * var"|S̄|S̄₁₁ᶜᶜᶜ"(i, j, k, grid, u, v, w))
@inline M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨|S|S₂₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * var"|S̄|S̄₂₂ᶜᶜᶜ"(i, j, k, grid, u, v, w))
@inline M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨|S|S₃₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * var"|S̄|S̄₃₃ᶜᶜᶜ"(i, j, k, grid, u, v, w))

@inline M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨|S|S₁₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * var"|S̄|S̄₁₂ᶜᶜᶜ"(i, j, k, grid, u, v, w))
@inline M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨|S|S₁₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * var"|S̄|S̄₁₃ᶜᶜᶜ"(i, j, k, grid, u, v, w))
@inline M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δᶠ(i, j, k, grid)^2 * (var"⟨|S|S₂₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * var"|S̄|S̄₂₃ᶜᶜᶜ"(i, j, k, grid, u, v, w))



@inline ϕψ(i, j, k, grid, ϕ, ψ) = ϕ[i, j, k] * ψ[i, j, k]
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



Base.summary(closure::ScaleInvariantSmagorinsky) = string("ScaleInvariantSmagorinsky: C=$(closure.C), averaging=$(closure.averaging), Pr=$(closure.Pr)")
Base.show(io::IO, closure::ScaleInvariantSmagorinsky) = print(io, summary(closure))

#####
##### For closures that only require an eddy viscosity νₑ field.
#####

function DiffusivityFields(grid, tracer_names, bcs, closure::ScaleInvariantSmagorinsky)

    default_eddy_viscosity_bcs = (; νₑ = FieldBoundaryConditions(grid, (Center, Center, Center)))
    bcs = merge(default_eddy_viscosity_bcs, bcs)
    νₑ = CenterField(grid, boundary_conditions=bcs.νₑ)
    LM = CenterField(grid)
    MM = CenterField(grid)

    return (; νₑ, LM, MM)
end
