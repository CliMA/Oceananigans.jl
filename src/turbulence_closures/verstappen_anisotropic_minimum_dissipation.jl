struct VerstappenAnisotropicMinimumDissipation{Z, T} <: IsotropicDiffusivity{T}
     C :: T
    Cb :: T
     ν :: T
     κ :: T
end

"""
    VerstappenAnisotropicMinimumDissipation(T=Float64; C=0.33, ν=1e-6, κ=1e-7)

Returns a `AnisotropicMinimumDissipation` closure object of type `T` with

    * `C` : Poincaré constant
    * `ν` : 'molecular' background viscosity
    * `κ` : 'molecular' background diffusivity
"""
function VerstappenAnisotropicMinimumDissipation(FT=Float64;
            C = 1/12,
           Cb = 0.0,
            ν = 1e-6,
            κ = 1e-7,
     wall_adj = false
    )
    return VerstappenAnisotropicMinimumDissipation{wall_adj, FT}(C, Cb, ν, κ)
end

const VAMD = VerstappenAnisotropicMinimumDissipation

function TurbulentDiffusivities(arch::Architecture, grid::Grid, ::VAMD)
     νₑ = CellField(arch, grid)
    κTₑ = CellField(arch, grid)
    κSₑ = CellField(arch, grid)
    return (νₑ=νₑ, κₑ=(T=κTₑ, S=κSₑ))
end

@inline function ν_ccc(i, j, k, grid::Grid{FT}, closure::VAMD, c,
                       eos, grav, u, v, w, T, S) where FT

    ijk = (i, j, k, grid)
    q = norm_tr_∇u_ccc(ijk..., u, v, w)

    if q == 0 # SGS viscosity is zero when strain is 0
        νˢᶠˢ = zero(FT)
    else
        r = norm_uᵢₐ_uⱼₐ_Σᵢⱼ_ccc(ijk..., closure, u, v, w)
        ζ = norm_wᵢ_bᵢ_ccc(ijk..., closure, eos, grav, w, T, S) / Δᶠz_ccc(ijk...)
        δ² = 3 / (1 / Δᶠx_ccc(ijk...)^2 + 1 / Δᶠy_ccc(ijk...)^2 + 1 / Δᶠz_ccc(ijk...)^2)
        νˢᶠˢ = - closure.C * δ² * (r - closure.Cb * ζ) / q
    end

    return max(zero(FT), νˢᶠˢ) + closure.ν
end

@inline function ν_ccf(i, j, k, grid::Grid{FT}, closure::VAMD, c,
                       eos, grav, u, v, w, T, S) where FT

    ijk = (i, j, k, grid)
    q = norm_tr_∇u_ccf(ijk..., u, v, w)

    if q == 0 # SGS viscosity is zero when strain is 0
        νˢᶠˢ = zero(FT)
    else
        r = norm_uᵢₐ_uⱼₐ_Σᵢⱼ_ccf(ijk..., closure, u, v, w)
        ζ = norm_wᵢ_bᵢ_ccf(ijk..., closure, eos, grav, w, T, S) / Δᶠz_ccf(ijk...)
        δ² = 3 / (1 / Δᶠx_ccf(ijk...)^2 + 1 / Δᶠy_ccf(ijk...)^2 + 1 / Δᶠz_ccf(ijk...)^2)
        νˢᶠˢ = - closure.C * δ² * (r - closure.Cb * ζ) / q
    end

    return max(zero(FT), νˢᶠˢ) + closure.ν
end

@inline function κ_ccc(i, j, k, grid::Grid{FT}, closure::VAMD, c,
                       eos, grav, u, v, w, T, S) where FT

    ijk = (i, j, k, grid)
    σ =  norm_θᵢ²_ccc(i, j, k, grid, c) # Tracer variance

    if σ == 0
        κˢᶠˢ = zero(FT)
    else
        ϑ =  norm_uᵢⱼ_cⱼ_cᵢ_ccc(ijk..., closure, u, v, w, c)
        δ² = 3 / (1 / Δᶠx_ccc(ijk...)^2 + 1 / Δᶠy_ccc(ijk...)^2 + 1 / Δᶠz_ccc(ijk...)^2)
        κˢᶠˢ = - closure.C * δ² * ϑ / σ
    end

    return max(zero(FT), κˢᶠˢ) + closure.κ
end

@inline function κ_ccf(i, j, k, grid::Grid{FT}, closure::VAMD, c,
                       eos, grav, u, v, w, T, S) where FT

    ijk = (i, j, k, grid)
    σ = norm_θᵢ²_ccf(ijk..., c) # Tracer variance

    if σ == 0
        κˢᶠˢ = zero(FT)
    else
        ϑ =  norm_uᵢⱼ_cⱼ_cᵢ_ccf(ijk..., closure, u, v, w, c)
        δ² = 3 / (1 / Δᶠx_ccf(ijk...)^2 + 1 / Δᶠy_ccf(ijk...)^2 + 1 / Δᶠz_ccf(ijk...)^2)
        κˢᶠˢ = - closure.C * δ² * ϑ / σ
    end

    return max(zero(FT), κˢᶠˢ) + closure.κ
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

@inline function norm_wᵢ_bᵢ_ccc(i, j, k, grid, closure, eos, grav, w, T, S)
    ijk = (i, j, k, grid)

    wx_bx = (▶xz_cac(ijk..., norm_∂x_w, w)
             * Δᶠx_ccc(ijk...) * ▶x_caa(ijk..., ∂x_faa, buoyancy, eos, grav, T, S))

    wy_by = (▶yz_acc(ijk..., norm_∂y_w, w)
             * Δᶠy_ccc(ijk...) * ▶y_aca(ijk..., ∂y_afa, buoyancy, eos, grav, T, S))

    wz_bz = (norm_∂z_w(ijk..., w)
             * Δᶠz_ccc(ijk...) * ▶z_aac(ijk..., ∂z_aaf, buoyancy, eos, grav, T, S))

    return wx_bx + wy_by + wz_bz
end

@inline function norm_wᵢ_bᵢ_ccf(i, j, k, grid, closure, eos, grav, w, T, S)
    ijk = (i, j, k, grid)

    wx_bx = (▶x_caa(ijk..., norm_∂x_w, w)
              * Δᶠx_ccf(ijk...) * ▶xz_caf(ijk..., ∂x_faa, buoyancy, eos, grav, T, S))

    wy_by = (▶y_aca(ijk..., norm_∂y_w, w)
              * Δᶠy_ccf(ijk...) * ▶yz_acf(ijk..., ∂y_afa, buoyancy, eos, grav, T, S))

    wz_bz = (▶z_aaf(ijk..., norm_∂z_w, w)
              * Δᶠz_ccf(ijk...) * ∂z_aaf(ijk..., buoyancy, eos, grav, T, S))

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

@inline function norm_uᵢⱼ_cⱼ_cᵢ_ccf(i, j, k, grid, closure, u, v, w, c)
    ijk = (i, j, k, grid)

    cx_ux = (
            ▶z_aaf(ijk..., norm_∂x_u, u) * ▶xz_caf(ijk..., norm_∂x_c², c)
        + ▶xyz_ccf(ijk..., norm_∂x_v, v) * ▶xz_caf(ijk..., norm_∂x_c, c) * ▶yz_acf(ijk..., norm_∂y_c, c)
        +   ▶x_caa(ijk..., norm_∂x_w, w) * ▶xz_caf(ijk..., norm_∂x_c, c) * norm_∂z_ccf(ijk..., c)
    )

    cy_uy = (
          ▶xyz_ccf(ijk..., norm_∂y_u, u) * ▶yz_acf(ijk..., norm_∂y_c, c) * ▶xz_caf(ijk..., norm_∂x_c, c)
        +     norm_∂y_v(ijk..., v)       * ▶yz_acf(ijk..., norm_∂y_c², c)
        +   ▶x_caa(ijk..., norm_∂y_w, w) * ▶yz_acf(ijk..., norm_∂y_c, c) * norm_∂z_ccf(ijk..., c)
    )

    cz_uz = (
          ▶x_caa(ijk..., norm_∂z_u, u) * norm_∂z_ccf(ijk..., c) * ▶xz_caf(ijk..., norm_∂x_c, c)
        + ▶y_aca(ijk..., norm_∂z_v, v) * norm_∂z_ccf(ijk..., c) * ▶yz_acf(ijk..., norm_∂y_c, c)
        + ▶z_aaf(ijk..., norm_∂z_w, w) *  norm_∂z_c²(ijk..., c)
    )

    return cx_ux + cy_uy + cz_uz
end

@inline norm_θᵢ²_ccc(i, j, k, grid, c) = (
      ▶x_caa(i, j, k, grid, norm_∂x_c², c)
    + ▶y_aca(i, j, k, grid, norm_∂y_c², c)
    + ▶z_aac(i, j, k, grid, norm_∂z_c², c)
)

@inline norm_θᵢ²_ccf(i, j, k, grid, c) = (
      ▶xz_caf(i, j, k, grid, norm_∂x_c², c)
    + ▶yz_acf(i, j, k, grid, norm_∂y_c², c)
    + norm_∂z_c²(i, j, k, grid, c)
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

# This function assumes rigid top and bottom boundary conditions
function calc_diffusivities!(K, grid, closure::VAMD{true}, eos, grav, U, Φ)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                if k == 1
                    @inbounds K.νₑ[i, j, k]   = ν_ccf(i, j, k+1, grid, closure, nothing, eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                    @inbounds K.κₑ.T[i, j, k] = κ_ccf(i, j, k+1, grid, closure, Φ.T,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                    @inbounds K.κₑ.S[i, j, k] = κ_ccf(i, j, k+1, grid, closure, Φ.S,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                elseif k == grid.Nz
                    @inbounds K.νₑ[i, j, k]   = ν_ccf(i, j, k, grid, closure, nothing, eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                    @inbounds K.κₑ.T[i, j, k] = κ_ccf(i, j, k, grid, closure, Φ.T,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                    @inbounds K.κₑ.S[i, j, k] = κ_ccf(i, j, k, grid, closure, Φ.S,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                else
                    @inbounds K.νₑ[i, j, k]   = ν_ccc(i, j, k, grid, closure, nothing, eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                    @inbounds K.κₑ.T[i, j, k] = κ_ccc(i, j, k, grid, closure, Φ.T,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                    @inbounds K.κₑ.S[i, j, k] = κ_ccc(i, j, k, grid, closure, Φ.S,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                end
            end
        end
    end
    return nothing
end

function calc_diffusivities!(K, grid, closure::VAMD{false}, eos, grav, U, Φ)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds K.νₑ[i, j, k]   = ν_ccc(i, j, k, grid, closure, nothing, eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                @inbounds K.κₑ.T[i, j, k] = κ_ccc(i, j, k, grid, closure, Φ.T,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                @inbounds K.κₑ.S[i, j, k] = κ_ccc(i, j, k, grid, closure, Φ.S,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
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

#####
##### Normalized gradients
#####

# ccc
const norm_∂x_u = ∂x_u
const norm_∂y_v = ∂y_v
const norm_∂z_w = ∂z_w

# ffc
@inline norm_∂x_v(i, j, k, grid, v) = 
    Δᶠx_ffc(i, j, k, grid) / Δᶠy_ffc(i, j, k, grid) * ∂x_faa(i, j, k, grid, v)

@inline norm_∂y_u(i, j, k, grid, u) = 
    Δᶠy_ffc(i, j, k, grid) / Δᶠx_ffc(i, j, k, grid) * ∂y_afa(i, j, k, grid, u)

# fcf
@inline norm_∂x_w(i, j, k, grid, w) = 
    Δᶠx_fcf(i, j, k, grid) / Δᶠz_fcf(i, j, k, grid) * ∂x_faa(i, j, k, grid, w)

@inline norm_∂z_u(i, j, k, grid, u) = 
    Δᶠz_fcf(i, j, k, grid) / Δᶠx_fcf(i, j, k, grid) * ∂z_aaf(i, j, k, grid, u)

# cff
@inline norm_∂y_w(i, j, k, grid, w) = 
    Δᶠy_cff(i, j, k, grid) / Δᶠz_cff(i, j, k, grid) * ∂y_afa(i, j, k, grid, w)

@inline norm_∂z_v(i, j, k, grid, v) = 
    Δᶠz_cff(i, j, k, grid) / Δᶠy_cff(i, j, k, grid) * ∂z_aaf(i, j, k, grid, v)

# tracers
@inline norm_∂x_c(i, j, k, grid, c) = Δᶠx_fcc(i, j, k, grid) * ∂x_faa(i, j, k, grid, c)
@inline norm_∂y_c(i, j, k, grid, c) = Δᶠy_cfc(i, j, k, grid) * ∂y_afa(i, j, k, grid, c)
@inline norm_∂z_c(i, j, k, grid, c) = Δᶠz_ccf(i, j, k, grid) * ∂z_aaf(i, j, k, grid, c)

#####
##### Strain operators
#####

# ccc
@inline norm_Σ₁₁(i, j, k, grid, u) = norm_∂x_u(i, j, k, grid, u)
@inline norm_Σ₂₂(i, j, k, grid, v) = norm_∂y_v(i, j, k, grid, v)
@inline norm_Σ₃₃(i, j, k, grid, w) = norm_∂z_w(i, j, k, grid, w)

@inline norm_tr_Σ(i, j, k, grid, u, v, w) =
    norm_Σ₁₁(i, j, k, grid, u) + norm_Σ₂₂(i, j, k, grid, v) + norm_Σ₃₃(i, j, k, grid, w)

# ffc
@inline norm_Σ₁₂(i, j, k, grid::Grid{T}, u, v) where T =
    T(0.5) * (norm_∂y_u(i, j, k, grid, u) + norm_∂x_v(i, j, k, grid, v))

# fcf
@inline norm_Σ₁₃(i, j, k, grid::Grid{T}, u, w) where T =
    T(0.5) * (norm_∂z_u(i, j, k, grid, u) + norm_∂x_w(i, j, k, grid, w))

# cff
@inline norm_Σ₂₃(i, j, k, grid::Grid{T}, v, w) where T =
    T(0.5) * (norm_∂z_v(i, j, k, grid, v) + norm_∂y_w(i, j, k, grid, w))

@inline norm_Σ₁₂²(i, j, k, grid, u, v) = norm_Σ₁₂(i, j, k, grid, u, v)^2
@inline norm_Σ₁₃²(i, j, k, grid, u, w) = norm_Σ₁₃(i, j, k, grid, u, w)^2
@inline norm_Σ₂₃²(i, j, k, grid, v, w) = norm_Σ₂₃(i, j, k, grid, v, w)^2

# Consistent function signatures for convenience:
@inline norm_∂x_v(i, j, k, grid, u, v, w) = norm_∂x_v(i, j, k, grid, v)
@inline norm_∂x_w(i, j, k, grid, u, v, w) = norm_∂x_w(i, j, k, grid, w)

@inline norm_∂y_u(i, j, k, grid, u, v, w) = norm_∂y_u(i, j, k, grid, u)
@inline norm_∂y_w(i, j, k, grid, u, v, w) = norm_∂y_w(i, j, k, grid, w)

@inline norm_∂z_u(i, j, k, grid, u, v, w) = norm_∂z_u(i, j, k, grid, u)
@inline norm_∂z_v(i, j, k, grid, u, v, w) = norm_∂z_v(i, j, k, grid, v)

@inline norm_Σ₁₁(i, j, k, grid, u, v, w) = norm_Σ₁₁(i, j, k, grid, u)
@inline norm_Σ₂₂(i, j, k, grid, u, v, w) = norm_Σ₂₂(i, j, k, grid, v)
@inline norm_Σ₃₃(i, j, k, grid, u, v, w) = norm_Σ₃₃(i, j, k, grid, w)

@inline norm_Σ₁₂(i, j, k, grid, u, v, w) = norm_Σ₁₂(i, j, k, grid, u, v)
@inline norm_Σ₁₃(i, j, k, grid, u, v, w) = norm_Σ₁₃(i, j, k, grid, u, w)
@inline norm_Σ₂₃(i, j, k, grid, u, v, w) = norm_Σ₂₃(i, j, k, grid, v, w)

# Symmetry relations
const norm_Σ₂₁ = norm_Σ₁₂
const norm_Σ₃₁ = norm_Σ₁₃
const norm_Σ₃₂ = norm_Σ₂₃

# Trace and squared strains
@inline norm_tr_Σ²(ijk...) = norm_Σ₁₁(ijk...)^2 +  norm_Σ₂₂(ijk...)^2 +  norm_Σ₃₃(ijk...)^2

@inline norm_Σ₁₂²(i, j, k, grid, u, v, w) = norm_Σ₁₂²(i, j, k, grid, u, v)
@inline norm_Σ₁₃²(i, j, k, grid, u, v, w) = norm_Σ₁₃²(i, j, k, grid, u, w)
@inline norm_Σ₂₃²(i, j, k, grid, u, v, w) = norm_Σ₂₃²(i, j, k, grid, v, w)

#####
##### Same-location velocity products
#####

# ccc
@inline norm_∂x_u²(ijk...) = norm_∂x_u(ijk...)^2
@inline norm_∂y_v²(ijk...) = norm_∂y_v(ijk...)^2
@inline norm_∂z_w²(ijk...) = norm_∂z_w(ijk...)^2

# ffc
@inline norm_∂x_v²(ijk...) = norm_∂x_v(ijk...)^2
@inline norm_∂y_u²(ijk...) = norm_∂y_u(ijk...)^2

@inline norm_∂x_v_Σ₁₂(ijk...) = norm_∂x_v(ijk...) * norm_Σ₁₂(ijk...)
@inline norm_∂y_u_Σ₁₂(ijk...) = norm_∂y_u(ijk...) * norm_Σ₁₂(ijk...)

# fcf
@inline norm_∂z_u²(ijk...) = norm_∂z_u(ijk...)^2
@inline norm_∂x_w²(ijk...) = norm_∂x_w(ijk...)^2

@inline norm_∂x_w_Σ₁₃(ijk...) = norm_∂x_w(ijk...) * norm_Σ₁₃(ijk...)
@inline norm_∂z_u_Σ₁₃(ijk...) = norm_∂z_u(ijk...) * norm_Σ₁₃(ijk...)

# cff
@inline norm_∂z_v²(ijk...) = norm_∂z_v(ijk...)^2
@inline norm_∂y_w²(ijk...) = norm_∂y_w(ijk...)^2

@inline norm_∂z_v_Σ₂₃(ijk...) = norm_∂z_v(ijk...) * norm_Σ₂₃(ijk...)
@inline norm_∂y_w_Σ₂₃(ijk...) = norm_∂y_w(ijk...) * norm_Σ₂₃(ijk...)

#####
##### Tracer gradients squared
#####

@inline norm_∂x_c²(ijk...) = norm_∂x_c(ijk...)^2
@inline norm_∂y_c²(ijk...) = norm_∂y_c(ijk...)^2
@inline norm_∂z_c²(ijk...) = norm_∂z_c(ijk...)^2
