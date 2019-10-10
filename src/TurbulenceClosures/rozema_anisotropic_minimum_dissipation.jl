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
const Δx_ccc = Δx
const Δy_ccc = Δy
const Δz_ccc = Δz
const Δx_ccf = Δx
const Δy_ccf = Δy
const Δz_ccf = Δz

function TurbulentDiffusivities(arch::AbstractArchitecture, grid::AbstractGrid, tracers, ::RAMD)
    νₑ = CellField(arch, grid)
    κₑ = TracerFields(arch, grid, tracers)
    return (νₑ=νₑ, κₑ=κₑ)
end

@inline function ν_ccc(i, j, k, grid::AbstractGrid{FT}, closure::RAMD, buoyancy, U, C) where FT
    q = tr_∇u_ccc(i, j, k, grid, U.u, U.v, U.w)

    if q == 0
        νˢᵍˢ = zero(FT)
    else
        r = Δ²ₐ_uᵢₐ_uⱼₐ_Σᵢⱼ_ccc(i, j, k, grid, closure, U.u, U.v, U.w)
        ζ = Δ²ᵢ_wᵢ_bᵢ_ccc(i, j, k, grid, closure, buoyancy, U.w, C)
        νˢᵍˢ = -closure.C * (r - closure.Cb * ζ) / q
    end

    return max(zero(FT), νˢᵍˢ) + closure.ν
end

@inline function κ_ccc(i, j, k, grid::AbstractGrid{FT}, closure::RAMD, c, tracer_idx, U) where FT
    @inbounds κ = closure.κ[tracer_idx]

    σ = θᵢ²_ccc(i, j, k, grid, c) 

    if σ == 0
        κˢᵍˢ = zero(FT)
    else
        ϑ =  Δ²ⱼ_uᵢⱼ_cⱼ_cᵢ_ccc(i, j, k, grid, closure, U.u, U.v, U.w, c)
        κˢᵍˢ = - closure.C * ϑ / σ
    end

    return max(zero(FT), κˢᵍˢ) + κ
end

#####
##### The *** 30 terms *** of AMD
#####

@inline function Δ²ₐ_uᵢₐ_uⱼₐ_Σᵢⱼ_ccc(i, j, k, grid, closure, u, v, w)
    Δx = Δx_ccc(i, j, k, grid, closure)
    Δy = Δy_ccc(i, j, k, grid, closure)
    Δz = Δz_ccc(i, j, k, grid, closure)

    ijk = (i, j, k, grid)
    uvw = (u, v, w)
    ijkuvw = (i, j, k, grid, u, v, w)

    Δx²_uᵢ₁_uⱼ₁_Σ₁ⱼ = Δx^2 * (
         Σ₁₁(ijkuvw...) * ∂x_u(ijk..., u)^2
      +  Σ₂₂(ijkuvw...) * ▶xy_cca(ijk..., ∂x_v², uvw...)
      +  Σ₃₃(ijkuvw...) * ▶xz_cac(ijk..., ∂x_w², uvw...)

      +  2 * ∂x_u(ijkuvw...) * ▶xy_cca(ijk..., ∂x_v_Σ₁₂, uvw...)
      +  2 * ∂x_u(ijkuvw...) * ▶xz_cac(ijk..., ∂x_w_Σ₁₃, uvw...)
      +  2 * ▶xy_cca(ijk..., ∂x_v, uvw...) * ▶xz_cac(ijk..., ∂x_w, uvw...) * ▶yz_acc(ijk..., Σ₂₃, uvw...)
    )

    Δy²_uᵢ₂_uⱼ₂_Σ₂ⱼ = Δy^2 * (
      + Σ₁₁(ijkuvw...) * ▶xy_cca(ijk..., ∂y_u², uvw...)
      + Σ₂₂(ijkuvw...) * ∂y_v(ijk..., v)^2
      + Σ₃₃(ijkuvw...) * ▶yz_acc(ijk..., ∂y_w², uvw...)

      +  2 * ∂y_v(ijkuvw...) * ▶xy_cca(ijk..., ∂y_u_Σ₁₂, uvw...)
      +  2 * ▶xy_cca(ijk..., ∂y_u, uvw...) * ▶yz_acc(ijk..., ∂y_w, uvw...) * ▶xz_cac(ijk..., Σ₁₃, uvw...)
      +  2 * ∂y_v(ijkuvw...) * ▶yz_acc(ijk..., ∂y_w_Σ₂₃, uvw...)
    )

    Δz²_uᵢ₃_uⱼ₃_Σ₃ⱼ = Δz^2 * (
      + Σ₁₁(ijkuvw...) * ▶xz_cac(ijk..., ∂z_u², uvw...)
      + Σ₂₂(ijkuvw...) * ▶yz_acc(ijk..., ∂z_v², uvw...)
      + Σ₃₃(ijkuvw...) * ∂z_w(ijk..., w)^2

      +  2 * ▶xz_cac(ijk..., ∂z_u, uvw...) * ▶yz_acc(ijk..., ∂z_v, uvw...) * ▶xy_cca(ijk..., Σ₁₂, uvw...)
      +  2 * ∂z_w(ijkuvw...) * ▶xz_cac(ijk..., ∂z_u_Σ₁₃, uvw...)
      +  2 * ∂z_w(ijkuvw...) * ▶yz_acc(ijk..., ∂z_v_Σ₂₃, uvw...)
    )

    return Δx²_uᵢ₁_uⱼ₁_Σ₁ⱼ + Δy²_uᵢ₂_uⱼ₂_Σ₂ⱼ + Δz²_uᵢ₃_uⱼ₃_Σ₃ⱼ
end

#####
##### trace(∇u) = uᵢⱼ uᵢⱼ
#####

@inline function tr_∇u_ccc(i, j, k, grid, uvw...)
    ijk = (i, j, k, grid)

    return (
        # ccc
        ∂x_u²(ijk..., uvw...)
      + ∂y_v²(ijk..., uvw...)
      + ∂z_w²(ijk..., uvw...)

        # ffc
      + ▶xy_cca(ijk..., ∂x_v², uvw...)
      + ▶xy_cca(ijk..., ∂y_u², uvw...)

        # fcf
      + ▶xz_cac(ijk..., ∂x_w², uvw...)
      + ▶xz_cac(ijk..., ∂z_u², uvw...)

        # cff
      + ▶yz_acc(ijk..., ∂y_w², uvw...)
      + ▶yz_acc(ijk..., ∂z_v², uvw...)
    )
end

@inline function Δ²ᵢ_wᵢ_bᵢ_ccc(i, j, k, grid, closure, buoyancy, w, C)
    ijk = (i, j, k, grid)

    Δx = Δx_ccc(ijk..., closure)
    Δy = Δy_ccc(ijk..., closure)
    Δz = Δz_ccc(ijk..., closure)

    Δx²_wx_bx = Δx^2 * (▶xz_cac(ijk..., ∂x_faa, w)
                          * ▶x_caa(ijk..., ∂x_faa, buoyancy_perturbation, buoyancy, C))

    Δy²_wy_by = Δy^2 * (▶yz_acc(ijk..., ∂y_afa, w)
                          * ▶y_aca(ijk..., ∂y_afa, buoyancy_perturbation, buoyancy, C))

    Δz²_wz_bz = Δz^2 * (∂z_aac(ijk..., w)
                          * ▶z_aac(ijk..., ∂z_aaf, buoyancy_perturbation, buoyancy, C))

    return Δx²_wx_bx + Δy²_wy_by + Δz²_wz_bz
end

@inline function Δ²ⱼ_uᵢⱼ_cⱼ_cᵢ_ccc(i, j, k, grid, closure, u, v, w, c)
    ijk = (i, j, k, grid)

    Δx = Δx_ccc(ijk..., closure)
    Δy = Δy_ccc(ijk..., closure)
    Δz = Δz_ccc(ijk..., closure)

    Δx²_cx_ux = Δx^2 * (
                 ∂x_caa(ijk..., u) * ▶x_caa(ijk..., ∂x_c², c)
        + ▶xy_cca(ijk..., ∂x_v, v) * ▶x_caa(ijk..., ∂x_faa, c) * ▶y_aca(ijk..., ∂y_afa, c)
        + ▶xz_cac(ijk..., ∂x_w, w) * ▶x_caa(ijk..., ∂x_faa, c) * ▶z_aac(ijk..., ∂z_aaf, c)
    )

    Δy²_cy_uy = Δy^2 * (
          ▶xy_cca(ijk..., ∂y_u, u) * ▶y_aca(ijk..., ∂y_afa, c) * ▶x_caa(ijk..., ∂x_faa, c)
        +        ∂y_aca(ijk..., v) * ▶y_aca(ijk..., ∂y_c², c)
        + ▶xz_cac(ijk..., ∂y_w, w) * ▶y_aca(ijk..., ∂y_afa, c) * ▶z_aac(ijk..., ∂z_aaf, c)
    )

    Δz²_cz_uz = Δz^2 * (
          ▶xz_cac(ijk..., ∂z_u, u) * ▶z_aac(ijk..., ∂z_aaf, c) * ▶x_caa(ijk..., ∂x_faa, c)
        + ▶yz_acc(ijk..., ∂z_v, v) * ▶z_aac(ijk..., ∂z_aaf, c) * ▶y_aca(ijk..., ∂y_afa, c)
        +        ∂z_aac(ijk..., w) * ▶z_aac(ijk..., ∂z_c², c)
    )

    return Δx²_cx_ux + Δy²_cy_uy + Δz²_cz_uz
end

@inline θᵢ²_ccc(i, j, k, grid, c) = (
      ▶x_caa(i, j, k, grid, ∂x_c², c)
    + ▶y_aca(i, j, k, grid, ∂y_c², c)
    + ▶z_aac(i, j, k, grid, ∂z_c², c)
)
