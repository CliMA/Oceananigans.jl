"""
    AnisotropicDiffusivity{NX, NY, NZ, KX, KY, KZ}

Parameters for anisotropic diffusivity models.
"""
struct AnisotropicDiffusivity{NX, NY, NZ, KX, KY, KZ} <: AbstractTensorDiffusivity
    νx :: NX
    νy :: NY
    νz :: NZ
    κx :: KX
    κy :: KY
    κz :: KZ
end

"""
    AnisotropicDiffusivity(; νx=ν₀, νy=ν₀, νz=ν₀, κx=κ₀, κy=κ₀, κz=κ₀,
                             νh=nothing, κh=nothing)

Returns parameters for a closure with a diagonal diffusivity tensor with heterogeneous
'anisotropic' components labeled by `x`, `y`, `z`.
Each component may be a number or function.
The tracer diffusivities `κx`, `κy`, and `κz` may be `NamedTuple`s with fields corresponding
to each tracer, or a single number or function to be a applied to all tracers.

If `νh` or `κh` are provided, then `νx = νy = νh`, and `κx = κy = κh`, respectively.

By default, a viscosity of `ν₀ = 1.05×10⁻⁶` m² s⁻¹ is used for all viscosity components
and a diffusivity of `κ₀ = 1.46×10⁻⁷` m² s⁻¹ is used for all diffusivity components for every tracer.
These values are the approximate viscosity and thermal diffusivity for seawater at 20°C
and 35 psu, according to Sharqawy et al., "Thermophysical properties of seawater: A review
of existing correlations and data" (2010).
"""
function AnisotropicDiffusivity(FT=Float64; νx=ν₀, νy=ν₀, νz=ν₀, κx=κ₀, κy=κ₀, κz=κ₀, νh=nothing, κh=nothing)
    if !isnothing(νh)
        νx = νh
        νy = νh
    end

    if !isnothing(κh)
        κx = κh
        κy = κh
    end

    if all(isa.((νx, νy, νz, κx, κy, κz), Number))
        κx = convert_diffusivity(FT, κx)
        κy = convert_diffusivity(FT, κy)
        κz = convert_diffusivity(FT, κz)
        return AnisotropicDiffusivity(FT(νx), FT(νy), FT(νz), κx, κy, κz)
    else
        return AnisotropicDiffusivity(νx, νy, νz, κx, κy, κz)
    end
end

function with_tracers(tracers, closure::AnisotropicDiffusivity)
    κx = tracer_diffusivities(tracers, closure.κx)
    κy = tracer_diffusivities(tracers, closure.κy)
    κz = tracer_diffusivities(tracers, closure.κz)
    return AnisotropicDiffusivity(closure.νx, closure.νy, closure.νz, κx, κy, κz)
end

calculate_diffusivities!(K, arch, grid, closure::AnisotropicDiffusivity, args...) = nothing

@inline ∂ⱼ_2ν_Σ₁ⱼ(i, j, k, grid, clock, closure::AnisotropicDiffusivity, U, args...) =
    ∂ⱼνᵢⱼ∂ᵢu(i, j, k, grid, clock, closure.νx, closure.νy, closure.νz, U.u)

@inline ∂ⱼ_2ν_Σ₂ⱼ(i, j, k, grid, clock, closure::AnisotropicDiffusivity, U, args...) =
    ∂ⱼνᵢⱼ∂ᵢv(i, j, k, grid, clock, closure.νx, closure.νy, closure.νz, U.v)

@inline ∂ⱼ_2ν_Σ₃ⱼ(i, j, k, grid, clock, closure::AnisotropicDiffusivity, U, args...) =
    ∂ⱼνᵢⱼ∂ᵢw(i, j, k, grid, clock, closure.νx, closure.νy, closure.νz, U.w)

@inline function ∇_κ_∇c(i, j, k, grid, clock, closure::AnisotropicDiffusivity,
                        c, ::Val{tracer_index}, args...) where tracer_index

    @inbounds κx = closure.κx[tracer_index]
    @inbounds κy = closure.κy[tracer_index]
    @inbounds κz = closure.κz[tracer_index]

    return ∂ⱼκᵢⱼ∂ᵢc(i, j, k, grid, clock, κx, κy, κz, c)
end

Base.show(io::IO, closure::AnisotropicDiffusivity) =
    print(io, "AnisotropicDiffusivity: " *
              "(νx=$(closure.νx), νy=$(closure.νy), νz=$(closure.νz)), " *
              "(κx=$(closure.κx), κy=$(closure.κy), κz=$(closure.κz))")
