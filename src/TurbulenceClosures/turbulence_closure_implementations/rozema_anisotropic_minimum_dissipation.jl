struct RozemaAnisotropicMinimumDissipation{FT, K} <: AbstractAnisotropicMinimumDissipation{FT}
     C :: FT
    Cb :: FT
     ν :: FT
     κ :: K

    function RozemaAnisotropicMinimumDissipation{FT}(C, Cb, ν, κ) where FT
        return new{FT, typeof(κ)}(C, Cb, ν, convert_diffusivity(FT, κ))
    end
end

"""
    RozemaAnisotropicMinimumDissipation(FT=Float64; C=0.33, ν=1.05e-6, κ=1.46e-7)

Returns a `RozemaAnisotropicMinimumDissipation` closure object of type `FT` with

    * `C` : Poincaré constant
    * `ν` : 'molecular' background viscosity
    * `κ` : 'molecular' background diffusivity for each tracer

See Rozema et al., " (2015)
"""
RozemaAnisotropicMinimumDissipation(FT=Float64; C=0.33, Cb=0.0, ν=ν₀, κ=κ₀) =
    RozemaAnisotropicMinimumDissipation{FT}(C, Cb, ν, κ)

function with_tracers(tracers, closure::RozemaAnisotropicMinimumDissipation{FT}) where FT
    κ = tracer_diffusivities(tracers, closure.κ)
    return RozemaAnisotropicMinimumDissipation{FT}(closure.C, closure.Cb, closure.ν, κ)
end

# Bindings
const RAMD = RozemaAnisotropicMinimumDissipation

"Return the filter width for Anisotropic Minimum Dissipation on a Regular Cartesian grid."
@inline Δx(i, j, k, grid::RegularCartesianGrid, ::RAMD) = grid.Δx
@inline Δy(i, j, k, grid::RegularCartesianGrid, ::RAMD) = grid.Δy
@inline Δz(i, j, k, grid::RegularCartesianGrid, ::RAMD) = grid.Δz

# We only have regular grids for now. When we have non-regular grids this will need to be changed.
const Δxᶜᶜᶜ = Δx
const Δyᶜᶜᶜ = Δy
const Δzᶜᶜᶜ = Δz
const Δxᶜᶜᶠ = Δx
const Δyᶜᶜᶠ = Δy
const Δzᶜᶜᶠ = Δz

function TurbulentDiffusivities(arch::AbstractArchitecture, grid::AbstractGrid, tracers, ::RAMD)
    νₑ = CellField(arch, grid)
    κₑ = TracerFields(arch, grid, tracers)
    return (νₑ=νₑ, κₑ=κₑ)
end

@inline function νᶜᶜᶜ(i, j, k, grid::AbstractGrid{FT}, closure::RAMD, buoyancy, U, C) where FT
    q = tr_∇uᶜᶜᶜ(i, j, k, grid, U.u, U.v, U.w)

    if q == 0
        νˢᵍˢ = zero(FT)
    else
        r = Δ²ₐ_uᵢₐ_uⱼₐ_Σᵢⱼᶜᶜᶜ(i, j, k, grid, closure, U.u, U.v, U.w)
        ζ = Δ²ᵢ_wᵢ_bᵢᶜᶜᶜ(i, j, k, grid, closure, buoyancy, U.w, C)
        νˢᵍˢ = -closure.C * (r - closure.Cb * ζ) / q
    end

    return max(zero(FT), νˢᵍˢ) + closure.ν
end

@inline function κᶜᶜᶜ(i, j, k, grid::AbstractGrid{FT}, closure::RAMD, c, ::Val{tracer_index},
                       U) where {FT, tracer_index}

    @inbounds κ = closure.κ[tracer_index]

    σ = θᵢ²ᶜᶜᶜ(i, j, k, grid, c)

    if σ == 0
        κˢᵍˢ = zero(FT)
    else
        ϑ =  Δ²ⱼ_uᵢⱼ_cⱼ_cᵢᶜᶜᶜ(i, j, k, grid, closure, U.u, U.v, U.w, c)
        κˢᵍˢ = - closure.C * ϑ / σ
    end

    return max(zero(FT), κˢᵍˢ) + κ
end

#####
##### The *** 30 terms *** of AMD
#####

@inline function Δ²ₐ_uᵢₐ_uⱼₐ_Σᵢⱼᶜᶜᶜ(i, j, k, grid, closure, u, v, w)
    Δx = Δxᶜᶜᶜ(i, j, k, grid, closure)
    Δy = Δyᶜᶜᶜ(i, j, k, grid, closure)
    Δz = Δzᶜᶜᶜ(i, j, k, grid, closure)

    ijk = (i, j, k, grid)
    uvw = (u, v, w)
    ijkuvw = (i, j, k, grid, u, v, w)

    Δx²_uᵢ₁_uⱼ₁_Σ₁ⱼ = Δx^2 * (
         Σ₁₁(ijkuvw...) * ∂x_u(ijk..., u)^2
      +  Σ₂₂(ijkuvw...) * ℑxyᶜᶜᵃ(ijk..., ∂x_v², uvw...)
      +  Σ₃₃(ijkuvw...) * ℑxzᶜᵃᶜ(ijk..., ∂x_w², uvw...)

      +  2 * ∂x_u(ijkuvw...) * ℑxyᶜᶜᵃ(ijk..., ∂x_v_Σ₁₂, uvw...)
      +  2 * ∂x_u(ijkuvw...) * ℑxzᶜᵃᶜ(ijk..., ∂x_w_Σ₁₃, uvw...)
      +  2 * ℑxyᶜᶜᵃ(ijk..., ∂x_v, uvw...) * ℑxzᶜᵃᶜ(ijk..., ∂x_w, uvw...) * ℑyzᵃᶜᶜ(ijk..., Σ₂₃, uvw...)
    )

    Δy²_uᵢ₂_uⱼ₂_Σ₂ⱼ = Δy^2 * (
      + Σ₁₁(ijkuvw...) * ℑxyᶜᶜᵃ(ijk..., ∂y_u², uvw...)
      + Σ₂₂(ijkuvw...) * ∂y_v(ijk..., v)^2
      + Σ₃₃(ijkuvw...) * ℑyzᵃᶜᶜ(ijk..., ∂y_w², uvw...)

      +  2 * ∂y_v(ijkuvw...) * ℑxyᶜᶜᵃ(ijk..., ∂y_u_Σ₁₂, uvw...)
      +  2 * ℑxyᶜᶜᵃ(ijk..., ∂y_u, uvw...) * ℑyzᵃᶜᶜ(ijk..., ∂y_w, uvw...) * ℑxzᶜᵃᶜ(ijk..., Σ₁₃, uvw...)
      +  2 * ∂y_v(ijkuvw...) * ℑyzᵃᶜᶜ(ijk..., ∂y_w_Σ₂₃, uvw...)
    )

    Δz²_uᵢ₃_uⱼ₃_Σ₃ⱼ = Δz^2 * (
      + Σ₁₁(ijkuvw...) * ℑxzᶜᵃᶜ(ijk..., ∂z_u², uvw...)
      + Σ₂₂(ijkuvw...) * ℑyzᵃᶜᶜ(ijk..., ∂z_v², uvw...)
      + Σ₃₃(ijkuvw...) * ∂z_w(ijk..., w)^2

      +  2 * ℑxzᶜᵃᶜ(ijk..., ∂z_u, uvw...) * ℑyzᵃᶜᶜ(ijk..., ∂z_v, uvw...) * ℑxyᶜᶜᵃ(ijk..., Σ₁₂, uvw...)
      +  2 * ∂z_w(ijkuvw...) * ℑxzᶜᵃᶜ(ijk..., ∂z_u_Σ₁₃, uvw...)
      +  2 * ∂z_w(ijkuvw...) * ℑyzᵃᶜᶜ(ijk..., ∂z_v_Σ₂₃, uvw...)
    )

    return Δx²_uᵢ₁_uⱼ₁_Σ₁ⱼ + Δy²_uᵢ₂_uⱼ₂_Σ₂ⱼ + Δz²_uᵢ₃_uⱼ₃_Σ₃ⱼ
end

#####
##### trace(∇u) = uᵢⱼ uᵢⱼ
#####

@inline function tr_∇uᶜᶜᶜ(i, j, k, grid, uvw...)
    ijk = (i, j, k, grid)

    return (
        # ccc
        ∂x_u²(ijk..., uvw...)
      + ∂y_v²(ijk..., uvw...)
      + ∂z_w²(ijk..., uvw...)

        # ffc
      + ℑxyᶜᶜᵃ(ijk..., ∂x_v², uvw...)
      + ℑxyᶜᶜᵃ(ijk..., ∂y_u², uvw...)

        # fcf
      + ℑxzᶜᵃᶜ(ijk..., ∂x_w², uvw...)
      + ℑxzᶜᵃᶜ(ijk..., ∂z_u², uvw...)

        # cff
      + ℑyzᵃᶜᶜ(ijk..., ∂y_w², uvw...)
      + ℑyzᵃᶜᶜ(ijk..., ∂z_v², uvw...)
    )
end

@inline function Δ²ᵢ_wᵢ_bᵢᶜᶜᶜ(i, j, k, grid, closure, buoyancy, w, C)
    ijk = (i, j, k, grid)

    Δx = Δxᶜᶜᶜ(ijk..., closure)
    Δy = Δyᶜᶜᶜ(ijk..., closure)
    Δz = Δzᶜᶜᶜ(ijk..., closure)

    Δx²_wx_bx = Δx^2 * (ℑxzᶜᵃᶜ(ijk..., ∂xᶠᵃᵃ, w)
                        * ℑxᶜᵃᵃ(ijk..., ∂xᶠᵃᵃ, buoyancy_perturbation, buoyancy, C))

    Δy²_wy_by = Δy^2 * (ℑyzᵃᶜᶜ(ijk..., ∂yᵃᶠᵃ, w)
                        * ℑyᵃᶜᵃ(ijk..., ∂yᵃᶠᵃ, buoyancy_perturbation, buoyancy, C))

    Δz²_wz_bz = Δz^2 * (∂zᵃᵃᶜ(ijk..., w)
                        * ℑzᵃᵃᶜ(ijk..., ∂zᵃᵃᶠ, buoyancy_perturbation, buoyancy, C))

    return Δx²_wx_bx + Δy²_wy_by + Δz²_wz_bz
end

@inline function Δ²ⱼ_uᵢⱼ_cⱼ_cᵢᶜᶜᶜ(i, j, k, grid, closure, u, v, w, c)
    ijk = (i, j, k, grid)

    Δx = Δxᶜᶜᶜ(ijk..., closure)
    Δy = Δyᶜᶜᶜ(ijk..., closure)
    Δz = Δzᶜᶜᶜ(ijk..., closure)

    Δx²_cx_ux = Δx^2 * (
                 ∂xᶜᵃᵃ(ijk..., u) * ℑxᶜᵃᵃ(ijk..., ∂x_c², c)
        + ℑxyᶜᶜᵃ(ijk..., ∂x_v, v) * ℑxᶜᵃᵃ(ijk..., ∂xᶠᵃᵃ, c) * ℑyᵃᶜᵃ(ijk..., ∂yᵃᶠᵃ, c)
        + ℑxzᶜᵃᶜ(ijk..., ∂x_w, w) * ℑxᶜᵃᵃ(ijk..., ∂xᶠᵃᵃ, c) * ℑzᵃᵃᶜ(ijk..., ∂zᵃᵃᶠ, c)
    )

    Δy²_cy_uy = Δy^2 * (
          ℑxyᶜᶜᵃ(ijk..., ∂y_u, u) * ℑyᵃᶜᵃ(ijk..., ∂yᵃᶠᵃ, c) * ℑxᶜᵃᵃ(ijk..., ∂xᶠᵃᵃ, c)
        +        ∂yᵃᶜᵃ(ijk..., v) * ℑyᵃᶜᵃ(ijk..., ∂y_c², c)
        + ℑxzᶜᵃᶜ(ijk..., ∂y_w, w) * ℑyᵃᶜᵃ(ijk..., ∂yᵃᶠᵃ, c) * ℑzᵃᵃᶜ(ijk..., ∂zᵃᵃᶠ, c)
    )

    Δz²_cz_uz = Δz^2 * (
          ℑxzᶜᵃᶜ(ijk..., ∂z_u, u) * ℑzᵃᵃᶜ(ijk..., ∂zᵃᵃᶠ, c) * ℑxᶜᵃᵃ(ijk..., ∂xᶠᵃᵃ, c)
        + ℑyzᵃᶜᶜ(ijk..., ∂z_v, v) * ℑzᵃᵃᶜ(ijk..., ∂zᵃᵃᶠ, c) * ℑyᵃᶜᵃ(ijk..., ∂yᵃᶠᵃ, c)
        +        ∂zᵃᵃᶜ(ijk..., w) * ℑzᵃᵃᶜ(ijk..., ∂z_c², c)
    )

    return Δx²_cx_ux + Δy²_cy_uy + Δz²_cz_uz
end

@inline θᵢ²ᶜᶜᶜ(i, j, k, grid, c) = (
      ℑxᶜᵃᵃ(i, j, k, grid, ∂x_c², c)
    + ℑyᵃᶜᵃ(i, j, k, grid, ∂y_c², c)
    + ℑzᵃᵃᶜ(i, j, k, grid, ∂z_c², c)
)
