#####
##### The turbulence closure proposed by Smagorinsky and Lilly.
##### We also call this 'Constant Smagorinsky'.
#####

struct LagrangianDynamicSmagorinsky{TD, FT, P} <: AbstractScalarDiffusivity{TD, ThreeDimensionalFormulation, 5}
    C :: FT
   Cb :: FT
   Pr :: P

   function LagrangianDynamicSmagorinsky{TD, FT}(C, Cb, Pr) where {TD, FT}
       Pr = convert_diffusivity(FT, Pr; discrete_form=false)
       P = typeof(Pr)
       return new{TD, FT, P}(C, Cb, Pr)
   end
end

const LDS{TD, FT, P} = LagrangianDynamicSmagorinsky{TD, FT, P} where {TD, FT, P}

@inline viscosity(::LDS, K) = K.νₑ
@inline diffusivity(closure::LDS, K, ::Val{id}) where id = K.νₑ / closure.Pr[id]

LagrangianDynamicSmagorinsky(time_discretization::TD = ExplicitTimeDiscretization(), FT=Float64; C=0.16, Cb=1.0, Pr=1.0) where TD =
        SmagorinskyLilly{TD, FT}(C, Cb, Pr)

LagrangianDynamicSmagorinsky(FT::DataType; kwargs...) = LagrangianDynamicSmagorinsky(ExplicitTimeDiscretization(), FT; kwargs...)

function with_tracers(tracers, closure::LDS{TD, FT}) where {TD, FT}
    Pr = tracer_diffusivities(tracers, closure.Pr)
    return SmagorinskyLilly{TD, FT}(closure.C, closure.Cb, Pr)
end

@inline filter_width(i, j, k, grid) = cbrt(Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid) * Δzᶜᶜᶜ(i, j, k, grid))

@kernel function _compute_lagrangian_smagorinsky_viscosity!(νₑ, grid, closure, buoyancy, velocities, tracers)
    i, j, k = @index(Global, NTuple)

    # Strain tensor dot product
    Σ² = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, velocities.u, velocities.v, velocities.w)

    # Stability function
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    ς  = stability(N², Σ², closure.Cb) # Use unity Prandtl number.

    # Filter width
    Δᶠ = filter_width(i, j, k, grid)

    𝒥ᴹᴹ = @inbounds tracers.𝒥ᴹᴹ[i, j, k]
    𝒥ᴺᴺ = @inbounds tracers.𝒥ᴺᴺ[i, j, k]
    𝒥ᴸᴹ = @inbounds tracers.𝒥ᴸᴹ[i, j, k]
    𝒥ᴿᴺ = @inbounds tracers.𝒥ᴿᴺ[i, j, k]

    C² = 𝒥ᴸᴹ / 𝒥ᴹᴹ / max(𝒥ᴹᴹ * 𝒥ᴿᴺ / 𝒥ᴺᴺ / 𝒥ᴸᴹ, 0.125)

    @inbounds νₑ[i, j, k] = ς * C² * Δᶠ^2 * sqrt(2Σ²)
end

function compute_diffusivities!(diffusivity_fields, closure::LagrangianDynamicSmagorinsky, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    buoyancy = model.buoyancy
    velocities = model.velocities
    tracers = model.tracers

    launch!(arch, grid, parameters, _compute_lagrangian_smagorinsky_viscosity!,
            diffusivity_fields.νₑ, grid, closure, buoyancy, velocities, tracers)

    return nothing
end

@inline κᶠᶜᶜ(i, j, k, grid, closure::LDS, K, ::Val{id}, args...) where id = ℑxᶠᵃᵃ(i, j, k, grid, K.νₑ) / closure.Pr[id]
@inline κᶜᶠᶜ(i, j, k, grid, closure::LDS, K, ::Val{id}, args...) where id = ℑyᵃᶠᵃ(i, j, k, grid, K.νₑ) / closure.Pr[id]
@inline κᶜᶜᶠ(i, j, k, grid, closure::LDS, K, ::Val{id}, args...) where id = ℑzᵃᵃᶠ(i, j, k, grid, K.νₑ) / closure.Pr[id]

@inline uᵢuⱼ(i, j, k, grid, uᵢ, uⱼ) = @inbounds uᵢ[i, j, k] * uⱼ[i, j, k]
@inline uᵢuⱼ(i, j, k, grid, 𝒰ᵢ::Function, 𝒰ⱼ::Function, uᵢ, uⱼ) = @inbounds 𝒰ᵢ(i, j, k, grid, uᵢ) * 𝒰ⱼ(i, j, k, grid, uⱼ)

@inline ϕ⁰⁵(i, j, k, grid, ϕ::Function, args...) = sqrt(ϕ(i, j, k, grid, args...))

@inline SₘSᵢⱼ(i, j, k, grid, Σ₁::Function, u::AbstractArray,                   Σ₂::Function, args...) = Σ₁(i, j, k, grid, u)    * Σ₂(i, j, k, grid, args...)
@inline SₘSᵢⱼ(i, j, k, grid, Σ₁::Function, u::AbstractArray, v::AbstractArray, Σ₂::Function, args...) = Σ₁(i, j, k, grid, u, v) * Σ₂(i, j, k, grid, args...)

@inline ℑΣ₁₂(i, j, k, grid, u, v) = ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂, u, v)
@inline ℑΣ₁₃(i, j, k, grid, u, w) = ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃, u, w)
@inline ℑΣ₂₃(i, j, k, grid, v, w) = ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃, v, w)

@inline function 𝒥ᴸᴹ_forcing_function(i, j, k, grid, clock, fields)
    𝒥ᴹᴹ = @inbounds fields.𝒥ᴹᴹ[i, j, k]
    𝒥ᴸᴹ = @inbounds fields.𝒥ᴸᴹ[i, j, k]
   
    u = fields.u
    v = fields.v
    w = fields.w

    # Averaging over a 27-point stencil
    # Remember! This is not a box-filter, more
    # of a gaussian filter
    # TODO: explore different filters
    u̅  = ℑxᶜᵃᵃ(i, j, k, grid, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, u)
    v̅  = ℑyᵃᶜᵃ(i, j, k, grid, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶠ, v)
    w̅  = ℑzᵃᵃᶜ(i, j, k, grid, ℑxyzᶜᶜᶠ, ℑxyzᶠᶠᶜ, w)

    u̅u̅ = ℑxᶜᵃᵃ(i, j, k, grid, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, uᵢuⱼ, u, u)
    v̅v̅ = ℑyᵃᶜᵃ(i, j, k, grid, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶠ, uᵢuⱼ, v, v)
    w̅w̅ = ℑzᵃᵃᶜ(i, j, k, grid, ℑxyzᶜᶜᶠ, ℑxyzᶠᶠᶜ, uᵢuⱼ, w, w)

    u̅v̅ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, uᵢuⱼ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, u, v)
    u̅w̅ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, uᵢuⱼ, ℑxᶜᵃᵃ, ℑzᵃᵃᶜ, u, w)
    v̅w̅ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, uᵢuⱼ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ, v, w)

    S̅₁₁ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, Σ₁₁, u) # Directly at centers
    S̅₂₂ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, Σ₂₂, v) # Directly at centers
    S̅₃₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, Σ₃₃, w) # Directly at centers
    
    S̅₁₂ = ℑxyᶜᶜᵃ(i, j, k, grid, ℑxyzᶠᶠᶜ, ℑxyzᶜᶜᶠ, Σ₁₂, u, v) # originally at ffc
    S̅₁₃ = ℑxzᶜᵃᶜ(i, j, k, grid, ℑxyzᶠᶜᶠ, ℑxyzᶜᶠᶜ, Σ₁₃, u, w) # originally at fcf
    S̅₂₃ = ℑyzᵃᶜᶜ(i, j, k, grid, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶜ, Σ₁₂, u, v) # originally at cff

    S̅ₘ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)

    S̅S̅₁₁ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, Σ₁₁, u, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₂₂ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, Σ₂₂, v, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₃₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, Σ₃₃, w, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    
    S̅S̅₁₂ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, ℑΣ₁₂, u, v, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₁₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, ℑΣ₁₃, u, w, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₂₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, ℑΣ₂₃, v, w, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)

    Δᶠ  = filter_width(i, j, k, grid)
    TΔ  = 3/2 * Δᶠ * (𝒥ᴹᴹ * 𝒥ᴸᴹ)^(-1 / 8)
    
    L₁₁ = u̅u̅ - u̅ * u̅
    L₂₂ = v̅v̅ - v̅ * v̅
    L₃₃ = w̅w̅ - w̅ * w̅    
    L₁₂ = u̅v̅ - u̅ * v̅
    L₁₃ = u̅w̅ - u̅ * w̅
    L₂₃ = v̅w̅ - v̅ * w̅

    # Here we assume that α (ratio between scales) is 2 and
    # β (ratio between the Smagorinsky coefficient at different scales)
    # is one because the model is assumed to be scale - invariant
    α = 2
    β = 1

    M₁₁ = 2Δᶠ * (S̅S̅₁₁ - α^2 * β * S̅ₘ * S̅₁₁)
    M₂₂ = 2Δᶠ * (S̅S̅₂₂ - α^2 * β * S̅ₘ * S̅₂₂)
    M₃₃ = 2Δᶠ * (S̅S̅₃₃ - α^2 * β * S̅ₘ * S̅₃₃)
    M₁₂ = 2Δᶠ * (S̅S̅₁₂ - α^2 * β * S̅ₘ * S̅₁₂)
    M₁₃ = 2Δᶠ * (S̅S̅₁₃ - α^2 * β * S̅ₘ * S̅₁₃)
    M₂₃ = 2Δᶠ * (S̅S̅₂₃ - α^2 * β * S̅ₘ * S̅₂₃)

    LᵢⱼMᵢⱼ = L₁₁ * M₁₁ + L₂₂ * M₂₂ + L₃₃ * M₃₃ + 2 * L₁₂ * M₁₂ + 2 * L₁₃ * M₁₃ + 2 * L₂₃ * M₂₃

    return 1 / TΔ * (LᵢⱼMᵢⱼ - 𝒥ᴸᴹ)
end

@inline function 𝒥ᴹᴹ_forcing_function(i, j, k, grid, clock, fields)
    𝒥ᴹᴹ = @inbounds fields.𝒥ᴹᴹ[i, j, k]
    𝒥ᴸᴹ = @inbounds fields.𝒥ᴸᴹ[i, j, k]
   
    u = fields.u
    v = fields.v
    w = fields.w

    S̅₁₁ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, Σ₁₁, u) # Directly at centers
    S̅₂₂ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, Σ₂₂, v) # Directly at centers
    S̅₃₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, Σ₃₃, w) # Directly at centers
    
    S̅₁₂ = ℑxyᶜᶜᵃ(i, j, k, grid, ℑxyzᶠᶠᶜ, ℑxyzᶜᶜᶠ, Σ₁₂, u, v) # originally at ffc
    S̅₁₃ = ℑxzᶜᵃᶜ(i, j, k, grid, ℑxyzᶠᶜᶠ, ℑxyzᶜᶠᶜ, Σ₁₃, u, w) # originally at fcf
    S̅₂₃ = ℑyzᵃᶜᶜ(i, j, k, grid, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶜ, Σ₁₂, u, v) # originally at cff

    S̅ₘ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)

    S̅S̅₁₁ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, Σ₁₁, u, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₂₂ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, Σ₂₂, v, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₃₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, Σ₃₃, w, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    
    S̅S̅₁₂ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, ℑΣ₁₂, u, v, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₁₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, ℑΣ₁₃, u, w, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₂₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, ℑΣ₂₃, v, w, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)

    Δᶠ  = filter_width(i, j, k, grid)
    TΔ  = 3/2 * Δᶠ * (𝒥ᴹᴹ * 𝒥ᴸᴹ)^(-1 / 8)

    # Here we assume that α (ratio between scales) is 2 and
    # β (ratio between the Smagorinsky coefficient at different scales)
    # is one because the model is assumed to be scale - invariant
    α = 2
    β = 1

    M₁₁ = 2Δᶠ * (S̅S̅₁₁ - α^2 * β * S̅ₘ * S̅₁₁)
    M₂₂ = 2Δᶠ * (S̅S̅₂₂ - α^2 * β * S̅ₘ * S̅₂₂)
    M₃₃ = 2Δᶠ * (S̅S̅₃₃ - α^2 * β * S̅ₘ * S̅₃₃)
    M₁₂ = 2Δᶠ * (S̅S̅₁₂ - α^2 * β * S̅ₘ * S̅₁₂)
    M₁₃ = 2Δᶠ * (S̅S̅₁₃ - α^2 * β * S̅ₘ * S̅₁₃)
    M₂₃ = 2Δᶠ * (S̅S̅₂₃ - α^2 * β * S̅ₘ * S̅₂₃)

    MᵢⱼMᵢⱼ = M₁₁^2 + M₂₂^2 + M₃₃^2 + 2 * M₁₂^2 + 2 * M₁₃^2 + 2 * M₂₃^2

    return 1 / TΔ * (MᵢⱼMᵢⱼ - 𝒥ᴹᴹ)
end

@inline function 𝒥ᴿᴺ_forcing_function(i, j, k, grid, clock, fields)
    𝒥ᴺᴺ = @inbounds fields.𝒥ᴺᴺ[i, j, k]
    𝒥ᴿᴺ = @inbounds fields.𝒥ᴿᴺ[i, j, k]
   
    u = fields.u
    v = fields.v
    w = fields.w

    # Averaging over a 27-point stencil
    # Remember! This is not a box-filter, more
    # of a gaussian filter
    # TODO: explore different filters
    u̅  = ℑxᶜᵃᵃ(i, j, k, grid, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, u)
    v̅  = ℑyᵃᶜᵃ(i, j, k, grid, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶠ, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶠ, v)
    w̅  = ℑzᵃᵃᶜ(i, j, k, grid, ℑxyzᶜᶜᶠ, ℑxyzᶠᶠᶜ, ℑxyzᶜᶜᶠ, ℑxyzᶠᶠᶜ, w)

    u̅u̅ = ℑxᶜᵃᵃ(i, j, k, grid, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, uᵢuⱼ, u, u)
    v̅v̅ = ℑyᵃᶜᵃ(i, j, k, grid, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶠ, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶠ, uᵢuⱼ, v, v)
    w̅w̅ = ℑzᵃᵃᶜ(i, j, k, grid, ℑxyzᶜᶜᶠ, ℑxyzᶠᶠᶜ, ℑxyzᶜᶜᶠ, ℑxyzᶠᶠᶜ, uᵢuⱼ, w, w)

    u̅v̅ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, uᵢuⱼ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, u, v)
    u̅w̅ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, uᵢuⱼ, ℑxᶜᵃᵃ, ℑzᵃᵃᶜ, u, w)
    v̅w̅ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, uᵢuⱼ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ, v, w)

    S̅₁₁ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, Σ₁₁, u) # Directly at centers
    S̅₂₂ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, Σ₂₂, v) # Directly at centers
    S̅₃₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, Σ₃₃, w) # Directly at centers
    
    S̅₁₂ = ℑxyᶜᶜᵃ(i, j, k, grid, ℑxyzᶠᶠᶜ, ℑxyzᶜᶜᶠ, ℑxyzᶠᶠᶜ, ℑxyzᶜᶜᶠ, Σ₁₂, u, v) # originally at ffc
    S̅₁₃ = ℑxzᶜᵃᶜ(i, j, k, grid, ℑxyzᶠᶜᶠ, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶠ, ℑxyzᶜᶠᶜ, Σ₁₃, u, w) # originally at fcf
    S̅₂₃ = ℑyzᵃᶜᶜ(i, j, k, grid, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶜ, Σ₁₂, u, v) # originally at cff

    S̅ₘ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)

    S̅S̅₁₁ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, Σ₁₁, u, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₂₂ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, Σ₂₂, v, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₃₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, Σ₃₃, w, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    
    S̅S̅₁₂ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, ℑΣ₁₂, u, v, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₁₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, ℑΣ₁₃, u, w, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₂₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, ℑΣ₂₃, v, w, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)

    Δᶠ  = filter_width(i, j, k, grid)
    TΔ  = 3/2 * Δᶠ * (𝒥ᴺᴺ * 𝒥ᴿᴺ)^(-1 / 8)
    
    L₁₁ = u̅u̅ - u̅ * u̅
    L₂₂ = v̅v̅ - v̅ * v̅
    L₃₃ = w̅w̅ - w̅ * w̅    
    L₁₂ = u̅v̅ - u̅ * v̅
    L₁₃ = u̅w̅ - u̅ * w̅
    L₂₃ = v̅w̅ - v̅ * w̅

    # Here we assume that α (ratio between scales) is 4 and
    # β (ratio between the Smagorinsky coefficient at different scales)
    # is one because the model is assumed to be scale - invariant
    α = 4
    β = 1

    M₁₁ = 2Δᶠ * (S̅S̅₁₁ - α^2 * β^2 * S̅ₘ * S̅₁₁)
    M₂₂ = 2Δᶠ * (S̅S̅₂₂ - α^2 * β^2 * S̅ₘ * S̅₂₂)
    M₃₃ = 2Δᶠ * (S̅S̅₃₃ - α^2 * β^2 * S̅ₘ * S̅₃₃)
    M₁₂ = 2Δᶠ * (S̅S̅₁₂ - α^2 * β^2 * S̅ₘ * S̅₁₂)
    M₁₃ = 2Δᶠ * (S̅S̅₁₃ - α^2 * β^2 * S̅ₘ * S̅₁₃)
    M₂₃ = 2Δᶠ * (S̅S̅₂₃ - α^2 * β^2 * S̅ₘ * S̅₂₃)

    LᵢⱼMᵢⱼ = L₁₁ * M₁₁ + L₂₂ * M₂₂ + L₃₃ * M₃₃ + 2 * L₁₂ * M₁₂ + 2 * L₁₃ * M₁₃ + 2 * L₂₃ * M₂₃

    return 1 / TΔ * (LᵢⱼMᵢⱼ - 𝒥ᴿᴺ)
end

@inline function 𝒥ᴺᴺ_forcing_function(i, j, k, grid, clock, fields)
    𝒥ᴺᴺ = @inbounds fields.𝒥ᴺᴺ[i, j, k]
    𝒥ᴿᴺ = @inbounds fields.𝒥ᴿᴺ[i, j, k]
   
    u = fields.u
    v = fields.v
    w = fields.w

    S̅₁₁ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, Σ₁₁, u) # Directly at centers
    S̅₂₂ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, Σ₂₂, v) # Directly at centers
    S̅₃₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, Σ₃₃, w) # Directly at centers
    
    S̅₁₂ = ℑxyᶜᶜᵃ(i, j, k, grid, ℑxyzᶠᶠᶜ, ℑxyzᶜᶜᶠ, ℑxyzᶠᶠᶜ, ℑxyzᶜᶜᶠ, Σ₁₂, u, v) # originally at ffc
    S̅₁₃ = ℑxzᶜᵃᶜ(i, j, k, grid, ℑxyzᶠᶜᶠ, ℑxyzᶜᶠᶜ, ℑxyzᶠᶜᶠ, ℑxyzᶜᶠᶜ, Σ₁₃, u, w) # originally at fcf
    S̅₂₃ = ℑyzᵃᶜᶜ(i, j, k, grid, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶜ, ℑxyzᶜᶠᶠ, ℑxyzᶠᶜᶜ, Σ₁₂, u, v) # originally at cff

    S̅ₘ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)

    S̅S̅₁₁ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, Σ₁₁, u, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₂₂ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, Σ₂₂, v, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₃₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, Σ₃₃, w, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    
    S̅S̅₁₂ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, ℑΣ₁₂, u, v, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₁₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, ℑΣ₁₃, u, w, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)
    S̅S̅₂₃ = ℑxyzᶜᶜᶜ(i, j, k, grid, ℑxyzᶠᶠᶠ, ℑxyzᶜᶜᶜ, ℑxyzᶠᶠᶠ, SₘSᵢⱼ, ℑΣ₂₃, v, w, ϕ⁰⁵, ΣᵢⱼΣᵢⱼᶜᶜᶜ, u, v, w)

    Δᶠ  = filter_width(i, j, k, grid)
    TΔ  = 3/2 * Δᶠ * (𝒥ᴺᴺ * 𝒥ᴿᴺ)^(-1 / 8)
    
    # Here we assume that α (ratio between scales) is 4 and
    # β (ratio between the Smagorinsky coefficient at different scales)
    # is one because the model is assumed to be scale - invariant
    α = 4
    β = 1

    M₁₁ = 2Δᶠ * (S̅S̅₁₁ - α^2 * β^2 * S̅ₘ * S̅₁₁)
    M₂₂ = 2Δᶠ * (S̅S̅₂₂ - α^2 * β^2 * S̅ₘ * S̅₂₂)
    M₃₃ = 2Δᶠ * (S̅S̅₃₃ - α^2 * β^2 * S̅ₘ * S̅₃₃)
    M₁₂ = 2Δᶠ * (S̅S̅₁₂ - α^2 * β^2 * S̅ₘ * S̅₁₂)
    M₁₃ = 2Δᶠ * (S̅S̅₁₃ - α^2 * β^2 * S̅ₘ * S̅₁₃)
    M₂₃ = 2Δᶠ * (S̅S̅₂₃ - α^2 * β^2 * S̅ₘ * S̅₂₃)
    
    MᵢⱼMᵢⱼ = M₁₁^2 + M₂₂^2 + M₃₃^2 + 2 * M₁₂^2 + 2 * M₁₃^2 + 2 * M₂₃^2

    return 1 / TΔ * (MᵢⱼMᵢⱼ - 𝒥ᴺᴺ)
end