struct VerstappenAnisotropicMinimumDissipation{T} <: AbstractAnisotropicMinimumDissipation{T}
     C :: T
    Cb :: T
     ν :: T
     κ :: T
end

"""
    VerstappenAnisotropicMinimumDissipation(T=Float64; C=1/12, ν=1.05e-6, κ=1.46e-7)

Returns a `VerstappenAnisotropicMinimumDissipation` closure object of type `T` with

    * `C`  : Poincaré constant
    * `Cb` : Buoyancy modification constant
    * `ν`  : 'molecular' background viscosity for momentum
    * `κ`  : 'molecular' background diffusivity for tracers

Based on the version of the anisotropic minimum dissipation closure proposed by:

 - Verstappen, R., "How much eddy dissipation is needed to counterbalance the
    "nonlinear production of small, unresolved scales in a large-eddy simulation
    of turbulence?", 2018

and described by

 - Vreugdenhil C., and Taylor J., "Large-eddy simulations of stratified plane Couette
 flow using the anisotropic minimum-dissipation model", (2018).
"""
function VerstappenAnisotropicMinimumDissipation(FT=Float64;
     C = 1/12,
    Cb = 0.0,
     ν = ν₀,
     κ = κ₀,
    )
    return VerstappenAnisotropicMinimumDissipation{FT}(C, Cb, ν, κ)
end

const VAMD = VerstappenAnisotropicMinimumDissipation

function TurbulentDiffusivities(arch::AbstractArchitecture, grid::AbstractGrid, ::VAMD)
     νₑ = CellField(arch, grid)
    κTₑ = CellField(arch, grid)
    κSₑ = CellField(arch, grid)
    return (νₑ=νₑ, κₑ=(T=κTₑ, S=κSₑ))
end

@inline function ν_ccc(i, j, k, grid::AbstractGrid{FT}, closure::VAMD, c, buoyancy, U, Φ) where FT
    ijk = (i, j, k, grid)
    q = norm_tr_∇u_ccc(ijk..., U.u, U.v, U.w)

    if q == 0 # SGS viscosity is zero when strain is 0
        νˢᵍˢ = zero(FT)
    else
        r = norm_uᵢₐ_uⱼₐ_Σᵢⱼ_ccc(ijk..., closure, U.u, U.v, U.w)
        ζ = norm_wᵢ_bᵢ_ccc(ijk..., closure, buoyancy, w, Φ) / Δᶠz_ccc(ijk...)
        δ² = 3 / (1 / Δᶠx_ccc(ijk...)^2 + 1 / Δᶠy_ccc(ijk...)^2 + 1 / Δᶠz_ccc(ijk...)^2)
        νˢᵍˢ = - closure.C * δ² * (r - closure.Cb * ζ) / q
    end

    return max(zero(FT), νˢᵍˢ) + closure.ν
end

@inline function κ_ccc(i, j, k, grid::AbstractGrid{FT}, closure::VAMD, c, buoyancy, U, Φ) where FT
    ijk = (i, j, k, grid)
    σ =  norm_θᵢ²_ccc(i, j, k, grid, c) # Tracer variance

    if σ == 0
        κˢᵍˢ = zero(FT)
    else
        ϑ =  norm_uᵢⱼ_cⱼ_cᵢ_ccc(ijk..., closure, U.u, U.v, U.w, c)
        δ² = 3 / (1 / Δᶠx_ccc(ijk...)^2 + 1 / Δᶠy_ccc(ijk...)^2 + 1 / Δᶠz_ccc(ijk...)^2)
        κˢᵍˢ = - closure.C * δ² * ϑ / σ
    end

    return max(zero(FT), κˢᵍˢ) + closure.κ
end

#####
##### The *** 30 terms *** of AMD
#####

@inline function norm_uᵢₐ_uⱼₐ_Σᵢⱼ_ccc(i, j, k, grid, closure, u, v, w)
    ijk = (i, j, k, grid)
    uvw = (u, v, w)
    ijkuvw = (i, j, k, grid, u, v, w)

    uᵢ₁_uⱼ₁_Σ₁ⱼ = (
         norm_Σ₁₁(ijkuvw...) * norm_∂x_u(ijk..., u)^2
      +  norm_Σ₂₂(ijkuvw...) * ▶xy_cca(ijk..., norm_∂x_v², uvw...)
      +  norm_Σ₃₃(ijkuvw...) * ▶xz_cac(ijk..., norm_∂x_w², uvw...)

      +  2 * norm_∂x_u(ijkuvw...) * ▶xy_cca(ijk..., norm_∂x_v_Σ₁₂, uvw...)
      +  2 * norm_∂x_u(ijkuvw...) * ▶xz_cac(ijk..., norm_∂x_w_Σ₁₃, uvw...)
      +  2 * ▶xy_cca(ijk..., norm_∂x_v, uvw...) * ▶xz_cac(ijk..., norm_∂x_w, uvw...)
           * ▶yz_acc(ijk..., norm_Σ₂₃, uvw...)
    )

    uᵢ₂_uⱼ₂_Σ₂ⱼ = (
      + norm_Σ₁₁(ijkuvw...) * ▶xy_cca(ijk..., norm_∂y_u², uvw...)
      + norm_Σ₂₂(ijkuvw...) * norm_∂y_v(ijk..., v)^2
      + norm_Σ₃₃(ijkuvw...) * ▶yz_acc(ijk..., norm_∂y_w², uvw...)

      +  2 * norm_∂y_v(ijkuvw...) * ▶xy_cca(ijk..., norm_∂y_u_Σ₁₂, uvw...)
      +  2 * ▶xy_cca(ijk..., norm_∂y_u, uvw...) * ▶yz_acc(ijk..., norm_∂y_w, uvw...)
           * ▶xz_cac(ijk..., norm_Σ₁₃, uvw...)
      +  2 * norm_∂y_v(ijkuvw...) * ▶yz_acc(ijk..., norm_∂y_w_Σ₂₃, uvw...)
    )

    uᵢ₃_uⱼ₃_Σ₃ⱼ = (
      + norm_Σ₁₁(ijkuvw...) * ▶xz_cac(ijk..., norm_∂z_u², uvw...)
      + norm_Σ₂₂(ijkuvw...) * ▶yz_acc(ijk..., norm_∂z_v², uvw...)
      + norm_Σ₃₃(ijkuvw...) * norm_∂z_w(ijk..., w)^2

      +  2 * ▶xz_cac(ijk..., norm_∂z_u, uvw...) * ▶yz_acc(ijk..., norm_∂z_v, uvw...)
           * ▶xy_cca(ijk..., norm_Σ₁₂, uvw...)
      +  2 * norm_∂z_w(ijkuvw...) * ▶xz_cac(ijk..., norm_∂z_u_Σ₁₃, uvw...)
      +  2 * norm_∂z_w(ijkuvw...) * ▶yz_acc(ijk..., norm_∂z_v_Σ₂₃, uvw...)
    )

    return uᵢ₁_uⱼ₁_Σ₁ⱼ + uᵢ₂_uⱼ₂_Σ₂ⱼ + uᵢ₃_uⱼ₃_Σ₃ⱼ
end

@inline function norm_uᵢₐ_uⱼₐ_Σᵢⱼ_ccf(i, j, k, grid, closure, u, v, w)
    ijk = (i, j, k, grid)
    uvw = (u, v, w)
    ijkuvw = (i, j, k, grid, u, v, w)

    uᵢ₁_uⱼ₁_Σ₁ⱼ = (
         ▶z_aaf(ijk..., norm_Σ₁₁, uvw...) *   ▶z_aaf(ijk..., norm_∂x_u², uvw...)
      +  ▶z_aaf(ijk..., norm_Σ₂₂, uvw...) * ▶xyz_ccf(ijk..., norm_∂x_v², uvw...)
      +  ▶z_aaf(ijk..., norm_Σ₃₃, uvw...) *   ▶x_caa(ijk..., norm_∂x_w², uvw...)

      +  2 *   ▶z_aaf(ijk..., norm_∂x_u, uvw...) * ▶xyz_ccf(ijk..., norm_∂x_v_Σ₁₂, uvw...)
      +  2 *   ▶z_aaf(ijk..., norm_∂x_u, uvw...) *   ▶x_caa(ijk..., norm_∂x_w_Σ₁₃, uvw...)
      +  2 * ▶xyz_ccf(ijk..., norm_∂x_v, uvw...) *   ▶x_caa(ijk..., norm_∂x_w, uvw...)
           *   ▶y_aca(ijk..., norm_Σ₂₃, uvw...)
    )

    uᵢ₂_uⱼ₂_Σ₂ⱼ = (
      + ▶z_aaf(ijk..., norm_Σ₁₁, uvw...) * ▶xyz_ccf(ijk..., norm_∂y_u², uvw...)
      + ▶z_aaf(ijk..., norm_Σ₂₂, uvw...) *   ▶z_aaf(ijk..., norm_∂y_v², uvw...)
      + ▶z_aaf(ijk..., norm_Σ₃₃, uvw...) *   ▶y_aca(ijk..., norm_∂y_w², uvw...)

      +  2 *  ▶z_aaf(ijk..., norm_∂y_v, uvw...) * ▶xyz_ccf(ijk..., norm_∂y_u_Σ₁₂, uvw...)
      +  2 * ▶xy_cca(ijk..., norm_∂y_u, uvw...) *   ▶y_aca(ijk..., norm_∂y_w, uvw...)
           *  ▶x_caa(ijk..., norm_Σ₁₃, uvw...)
      +  2 *  ▶z_aaf(ijk..., norm_∂y_v, uvw...) *   ▶y_aca(ijk..., norm_∂y_w_Σ₂₃, uvw...)
    )

    uᵢ₃_uⱼ₃_Σ₃ⱼ = (
      + ▶z_aaf(ijk..., norm_Σ₁₁, uvw...) * ▶x_caa(ijk..., norm_∂z_u², uvw...)
      + ▶z_aaf(ijk..., norm_Σ₂₂, uvw...) * ▶y_aca(ijk..., norm_∂z_v², uvw...)
      + ▶z_aaf(ijk..., norm_Σ₃₃, uvw...) * ▶z_aaf(ijk..., norm_∂z_w², uvw...)

      +  2 *   ▶x_caa(ijk..., norm_∂z_u, uvw...) * ▶y_aca(ijk..., norm_∂z_v, uvw...)
           * ▶xyz_ccf(ijk..., norm_Σ₁₂, uvw...)
      +  2 *   ▶z_aaf(ijk..., norm_∂z_w, uvw...) * ▶x_caa(ijk..., norm_∂z_u_Σ₁₃, uvw...)
      +  2 *   ▶z_aaf(ijk..., norm_∂z_w, uvw...) * ▶y_aca(ijk..., norm_∂z_v_Σ₂₃, uvw...)
    )

    return uᵢ₁_uⱼ₁_Σ₁ⱼ + uᵢ₂_uⱼ₂_Σ₂ⱼ + uᵢ₃_uⱼ₃_Σ₃ⱼ
end


#####
##### trace(∇u) = uᵢⱼ uᵢⱼ
#####

@inline function norm_tr_∇u_ccc(i, j, k, grid, uvw...)
    ijk = (i, j, k, grid)

    return (
        # ccc
        norm_∂x_u²(ijk..., uvw...)
      + norm_∂y_v²(ijk..., uvw...)
      + norm_∂z_w²(ijk..., uvw...)

        # ffc
      + ▶xy_cca(ijk..., norm_∂x_v², uvw...)
      + ▶xy_cca(ijk..., norm_∂y_u², uvw...)

        # fcf
      + ▶xz_cac(ijk..., norm_∂x_w², uvw...)
      + ▶xz_cac(ijk..., norm_∂z_u², uvw...)

        # cff
      + ▶yz_acc(ijk..., norm_∂y_w², uvw...)
      + ▶yz_acc(ijk..., norm_∂z_v², uvw...)
    )
end

@inline function norm_tr_∇u_ccf(i, j, k, grid, uvw...)
    ijk = (i, j, k, grid)

    return (
        # ccc
          ▶z_aaf(ijk..., norm_∂x_u², uvw...)
        + ▶z_aaf(ijk..., norm_∂y_v², uvw...)
        + ▶z_aaf(ijk..., norm_∂z_w², uvw...)

        # ffc
      + ▶xyz_ccf(ijk..., norm_∂x_v², uvw...)
      + ▶xyz_ccf(ijk..., norm_∂y_u², uvw...)

        # fcf
        + ▶x_caa(ijk..., norm_∂x_w², uvw...)
        + ▶x_caa(ijk..., norm_∂z_u², uvw...)

        # cff
        + ▶y_aca(ijk..., norm_∂y_w², uvw...)
        + ▶y_aca(ijk..., norm_∂z_v², uvw...)
    )
end

@inline function norm_wᵢ_bᵢ_ccc(i, j, k, grid, closure, buoyancy, w, C)
    ijk = (i, j, k, grid)

    wx_bx = (▶xz_cac(ijk..., norm_∂x_w, w)
             * Δᶠx_ccc(ijk...) * ▶x_caa(ijk..., ∂x_faa, buoyancy_perturbation, buoyancy, C))

    wy_by = (▶yz_acc(ijk..., norm_∂y_w, w)
             * Δᶠy_ccc(ijk...) * ▶y_aca(ijk..., ∂y_afa, buoyancy_perturbation, buoyancy, C))

    wz_bz = (norm_∂z_w(ijk..., w)
             * Δᶠz_ccc(ijk...) * ▶z_aac(ijk..., ∂z_aaf, buoyancy_perturbation, buoyancy, C))

    return wx_bx + wy_by + wz_bz
end

@inline function norm_uᵢⱼ_cⱼ_cᵢ_ccc(i, j, k, grid, closure, u, v, w, c)
    ijk = (i, j, k, grid)

    cx_ux = (
                   norm_∂x_u(ijk..., u) * ▶x_caa(ijk..., norm_∂x_c², c)
        + ▶xy_cca(ijk..., norm_∂x_v, v) * ▶x_caa(ijk..., norm_∂x_c, c) * ▶y_aca(ijk..., norm_∂y_c, c)
        + ▶xz_cac(ijk..., norm_∂x_w, w) * ▶x_caa(ijk..., norm_∂x_c, c) * ▶z_aac(ijk..., norm_∂z_c, c)
    )

    cy_uy = (
          ▶xy_cca(ijk..., norm_∂y_u, u) * ▶y_aca(ijk..., norm_∂y_c, c) * ▶x_caa(ijk..., norm_∂x_c, c)
        +          norm_∂y_v(ijk..., v) * ▶y_aca(ijk..., norm_∂y_c², c)
        + ▶xz_cac(ijk..., norm_∂y_w, w) * ▶y_aca(ijk..., norm_∂y_c, c) * ▶z_aac(ijk..., norm_∂z_c, c)
    )

    cz_uz = (
          ▶xz_cac(ijk..., norm_∂z_u, u) * ▶z_aac(ijk..., norm_∂z_c, c) * ▶x_caa(ijk..., norm_∂x_c, c)
        + ▶yz_acc(ijk..., norm_∂z_v, v) * ▶z_aac(ijk..., norm_∂z_c, c) * ▶y_aca(ijk..., norm_∂y_c, c)
        +          norm_∂z_w(ijk..., w) * ▶z_aac(ijk..., norm_∂z_c², c)
    )

    return cx_ux + cy_uy + cz_uz
end

@inline norm_θᵢ²_ccc(i, j, k, grid, c) = (
      ▶x_caa(i, j, k, grid, norm_∂x_c², c)
    + ▶y_aca(i, j, k, grid, norm_∂y_c², c)
    + ▶z_aac(i, j, k, grid, norm_∂z_c², c)
)

"""
    ∇_κ_∇T(i, j, k, grid, T, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ c)` for the turbulence
`closure`, where `c` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇T(i, j, k, grid, T, closure::VAMD, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_c, T, diffusivities.κₑ.T, closure)
    + ∂y_aca(i, j, k, grid, κ_∂y_c, T, diffusivities.κₑ.T, closure)
    + ∂z_aac(i, j, k, grid, κ_∂z_c, T, diffusivities.κₑ.T, closure)
)

"""
    ∇_κ_∇S(i, j, k, grid, S, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ S)` for the turbulence
`closure`, where `c` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇S(i, j, k, grid, S, closure::VAMD, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_c, S, diffusivities.κₑ.S, closure)
    + ∂y_aca(i, j, k, grid, κ_∂y_c, S, diffusivities.κₑ.S, closure)
    + ∂z_aac(i, j, k, grid, κ_∂z_c, S, diffusivities.κₑ.S, closure)
)

function calc_diffusivities!(K, grid, closure::VAMD, buoyancy, U, Φ)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds K.νₑ[i, j, k]   = ν_ccc(i, j, k, grid, closure, nothing, buoyancy, U, Φ)
                @inbounds K.κₑ.T[i, j, k] = κ_ccc(i, j, k, grid, closure, Φ.T,     buoyancy, U, Φ)
                @inbounds K.κₑ.S[i, j, k] = κ_ccc(i, j, k, grid, closure, Φ.S,     buoyancy, U, Φ)
            end
        end
    end
    return nothing
end

#####
##### Filter width at various locations
#####

# Recall that filter widths are 2x the grid spacing in VAMD
@inline Δᶠx_ccc(i, j, k, grid::RegularCartesianGrid) = 2 * grid.Δx
@inline Δᶠy_ccc(i, j, k, grid::RegularCartesianGrid) = 2 * grid.Δy
@inline Δᶠz_ccc(i, j, k, grid::RegularCartesianGrid) = 2 * grid.Δz

for loc in (:ccf, :fcc, :cfc, :ffc, :cff, :fcf), ξ in (:x, :y, :z)
    Δ_loc = Symbol(:Δᶠ, ξ, :_, loc)
    Δ_ccc = Symbol(:Δᶠ, ξ, :_ccc)
    @eval begin
        const $Δ_loc = $Δ_ccc
    end
end
