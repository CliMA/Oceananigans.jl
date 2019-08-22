struct RozemaAnisotropicMinimumDissipation{T} <: IsotropicDiffusivity{T}
     C :: T
    Cb :: T
     ν :: T
     κ :: T
end

"""
    RozemaAnisotropicMinimumDissipation(T=Float64; C=0.33, ν=1e-6, κ=1e-7)

Returns a `RozemaAnisotropicMinimumDissipation` closure object of type `T` with

    * `C` : Poincaré constant
    * `ν` : 'molecular' background viscosity
    * `κ` : 'molecular' background diffusivity
"""
function RozemaAnisotropicMinimumDissipation(FT=Float64;
         C = 0.33,
        Cb = 0.0,
         ν = 1e-6,
         κ = 1e-7
    )
    return RozemaAnisotropicMinimumDissipation{FT}(C, Cb, ν, κ)
end

# Bindings
const RAMD = RozemaAnisotropicMinimumDissipation
const AnisotropicMinimumDissipation = RozemaAnisotropicMinimumDissipation

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

function TurbulentDiffusivities(arch::Architecture, grid::Grid, ::RAMD)
     νₑ = CellField(arch, grid)
    κTₑ = CellField(arch, grid)
    κSₑ = CellField(arch, grid)
    return (νₑ=νₑ, κₑ=(T=κTₑ, S=κSₑ))
end

@inline function ν_ccc(i, j, k, grid::Grid{FT}, closure::RAMD, c,
                       eos, grav, u, v, w, T, S) where FT

    q = tr_∇u_ccc(i, j, k, grid, u, v, w)

    if q == 0
        νˢᶠˢ = zero(FT)
    else
        r = Δ²ₐ_uᵢₐ_uⱼₐ_Σᵢⱼ_ccc(i, j, k, grid, closure, u, v, w)
        ζ = Δ²ᵢ_wᵢ_bᵢ_ccc(i, j, k, grid, closure, eos, grav, w, T, S)
        νˢᶠˢ = -closure.C * (r - closure.Cb * ζ) / q
    end

    return max(zero(FT), νˢᶠˢ) + closure.ν
end

@inline function ν_ccf(i, j, k, grid::Grid{FT}, closure::RAMD, c,
                       eos, grav, u, v, w, T, S) where FT

    q = tr_∇u_ccf(i, j, k, grid, u, v, w)

    if q == 0
        νˢᶠˢ = zero(FT)
    else
        r = Δ²ₐ_uᵢₐ_uⱼₐ_Σᵢⱼ_ccf(i, j, k, grid, closure, u, v, w)
        ζ = Δ²ᵢ_wᵢ_bᵢ_ccf(i, j, k, grid, closure, eos, grav, w, T, S)
        νˢᶠˢ = -closure.C * (r - closure.Cb * ζ) / q
    end

    return max(zero(FT), νˢᶠˢ) + closure.ν
end

@inline function κ_ccc(i, j, k, grid::Grid{FT}, closure::RAMD, c,
                       eos, grav, u, v, w, T, S) where FT
    
    σ = θᵢ²_ccc(i, j, k, grid, c) # Tracer variance

    if σ == 0
        κˢᶠˢ = zero(FT)
    else
        ϑ =  Δ²ⱼ_uᵢⱼ_cⱼ_cᵢ_ccc(i, j, k, grid, closure, c, u, v, w)
        κˢᶠˢ = - closure.C * ϑ / σ
    end

    return max(zero(FT), κˢᶠˢ) + closure.κ
end

@inline function κ_ccf(i, j, k, grid::Grid{FT}, closure::RAMD, c,
                       eos, grav, u, v, w, T, S) where FT
    
    σ = θᵢ²_ccf(i, j, k, grid, c) # Tracer variance

    if σ == 0
        κˢᶠˢ = zero(FT)
    else
        ϑ = Δ²ⱼ_uᵢⱼ_cⱼ_cᵢ_ccf(i, j, k, grid, closure, c, u, v, w)
        κˢᶠˢ = - closure.C * ϑ / σ
    end

    return max(zero(FT), κˢᶠˢ) + closure.κ
end

#####
##### *** 30 terms ***
#####

#####
##### the heinous
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

@inline function Δ²ₐ_uᵢₐ_uⱼₐ_Σᵢⱼ_ccf(i, j, k, grid, closure, u, v, w)
    Δx = Δx_ccf(i, j, k, grid, closure)
    Δy = Δy_ccf(i, j, k, grid, closure)
    Δz = Δz_ccf(i, j, k, grid, closure)

    ijk = (i, j, k, grid)
    uvw = (u, v, w)
    ijkuvw = (i, j, k, grid, u, v, w)

    Δx²_uᵢ₁_uⱼ₁_Σ₁ⱼ = Δx^2 * (
         ▶z_aaf(ijk..., Σ₁₁, uvw...) *   ▶z_aaf(ijk..., ∂x_u², uvw...)
      +  ▶z_aaf(ijk..., Σ₂₂, uvw...) * ▶xyz_ccf(ijk..., ∂x_v², uvw...)
      +  ▶z_aaf(ijk..., Σ₃₃, uvw...) *   ▶x_caa(ijk..., ∂x_w², uvw...)

      +  2 *   ▶z_aaf(ijk..., ∂x_u, uvw...) * ▶xyz_ccf(ijk..., ∂x_v_Σ₁₂, uvw...)
      +  2 *   ▶z_aaf(ijk..., ∂x_u, uvw...) *   ▶x_caa(ijk..., ∂x_w_Σ₁₃, uvw...)
      +  2 * ▶xyz_ccf(ijk..., ∂x_v, uvw...) *   ▶x_caa(ijk..., ∂x_w, uvw...) * ▶y_aca(ijk..., Σ₂₃, uvw...)
    )

    Δy²_uᵢ₂_uⱼ₂_Σ₂ⱼ = Δy^2 * (
      + ▶z_aaf(ijk..., Σ₁₁, uvw...) * ▶xyz_ccf(ijk..., ∂y_u², uvw...)
      + ▶z_aaf(ijk..., Σ₂₂, uvw...) *   ▶z_aaf(ijk..., ∂y_v², uvw...)
      + ▶z_aaf(ijk..., Σ₃₃, uvw...) *   ▶y_aca(ijk..., ∂y_w², uvw...)

      +  2 *  ▶z_aaf(ijk..., ∂y_v, uvw...) * ▶xyz_ccf(ijk..., ∂y_u_Σ₁₂, uvw...)
      +  2 * ▶xy_cca(ijk..., ∂y_u, uvw...) *   ▶y_aca(ijk..., ∂y_w, uvw...) * ▶x_caa(ijk..., Σ₁₃, uvw...)
      +  2 *  ▶z_aaf(ijk..., ∂y_v, uvw...) *   ▶y_aca(ijk..., ∂y_w_Σ₂₃, uvw...)
    )

    Δz²_uᵢ₃_uⱼ₃_Σ₃ⱼ = Δz^2 * (
      + ▶z_aaf(ijk..., Σ₁₁, uvw...) * ▶x_caa(ijk..., ∂z_u², uvw...)
      + ▶z_aaf(ijk..., Σ₂₂, uvw...) * ▶y_aca(ijk..., ∂z_v², uvw...)
      + ▶z_aaf(ijk..., Σ₃₃, uvw...) * ▶z_aaf(ijk..., ∂z_w², uvw...)

      +  2 * ▶x_caa(ijk..., ∂z_u, uvw...) * ▶y_aca(ijk..., ∂z_v, uvw...) * ▶xyz_ccf(ijk..., Σ₁₂, uvw...)
      +  2 * ▶z_aaf(ijk..., ∂z_w, uvw...) * ▶x_caa(ijk..., ∂z_u_Σ₁₃, uvw...)
      +  2 * ▶z_aaf(ijk..., ∂z_w, uvw...) * ▶y_aca(ijk..., ∂z_v_Σ₂₃, uvw...)
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

@inline function tr_∇u_ccf(i, j, k, grid, uvw...)
    ijk = (i, j, k, grid)

    return (
        # ccc
        ▶z_aaf(ijk..., ∂x_u², uvw...)
      + ▶z_aaf(ijk..., ∂y_v², uvw...)
      + ▶z_aaf(ijk..., ∂z_w², uvw...)

        # ffc
      + ▶xyz_ccf(ijk..., ∂x_v², uvw...)
      + ▶xyz_ccf(ijk..., ∂y_u², uvw...)

        # fcf
      + ▶x_caa(ijk..., ∂x_w², uvw...)
      + ▶x_caa(ijk..., ∂z_u², uvw...)

        # cff
      + ▶y_aca(ijk..., ∂y_w², uvw...)
      + ▶y_aca(ijk..., ∂z_v², uvw...)
    )
end

@inline function Δ²ᵢ_wᵢ_bᵢ_ccc(i, j, k, grid, closure, eos, grav, w, T, S)
    ijk = (i, j, k, grid)

    Δx = Δx_ccc(ijk..., closure)
    Δy = Δy_ccc(ijk..., closure)
    Δz = Δz_ccc(ijk..., closure)

    Δx²_wx_bx = Δx^2 * (▶xz_cac(ijk..., ∂x_faa, w)
                          * ▶x_caa(ijk..., ∂x_faa, buoyancy, eos, grav, T, S))

    Δy²_wy_by = Δy^2 * (▶yz_acc(ijk..., ∂y_afa, w)
                          * ▶y_aca(ijk..., ∂y_afa, buoyancy, eos, grav, T, S))

    Δz²_wz_bz = Δz^2 * (∂z_aac(ijk..., w)
                          * ▶z_aac(ijk..., ∂z_aaf, buoyancy, eos, grav, T, S))

    return Δx²_wx_bx + Δy²_wy_by + Δz²_wz_bz
end

@inline function Δ²ᵢ_wᵢ_bᵢ_ccf(i, j, k, grid, closure, eos, grav, w, T, S)
    ijk = (i, j, k, grid)

    Δx = Δx_ccf(ijk..., closure)
    Δy = Δy_ccf(ijk..., closure)
    Δz = Δz_ccf(ijk..., closure)

    Δx²_wx_bx = Δx^2 * (▶x_caa(ijk..., ∂x_faa, w)
                          * ▶xz_caf(ijk..., ∂x_faa, buoyancy, eos, grav, T, S))

    Δy²_wy_by = Δy^2 * (▶y_aca(ijk..., ∂y_afa, w)
                          * ▶yz_acf(ijk..., ∂y_afa, buoyancy, eos, grav, T, S))

    Δz²_wz_bz = Δz^2 * (▶z_aaf(ijk..., ∂z_aac, w)
                          * ∂z_aaf(ijk..., buoyancy, eos, grav, T, S))

    return Δx²_wx_bx + Δy²_wy_by + Δz²_wz_bz
end

@inline function Δ²ⱼ_uᵢⱼ_cⱼ_cᵢ_ccc(i, j, k, grid, closure, c, u, v, w)
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

@inline function Δ²ⱼ_uᵢⱼ_cⱼ_cᵢ_ccf(i, j, k, grid, closure, c, u, v, w)
    ijk = (i, j, k, grid)

    Δx = Δx_ccf(ijk..., closure)
    Δy = Δy_ccf(ijk..., closure)
    Δz = Δz_ccf(ijk..., closure)

    Δx²_cx_ux = Δx^2 * (
            ▶z_aaf(ijk..., ∂x_caa, u) * ▶xz_caf(ijk..., ∂x_c², c)
        + ▶xyz_ccf(ijk..., ∂x_v, v)   * ▶xz_caf(ijk..., ∂x_faa, c) * ▶yz_acf(ijk..., ∂y_afa, c)
        +   ▶x_caa(ijk..., ∂x_w, w)   * ▶xz_caf(ijk..., ∂x_faa, c) *  ∂z_aaf(ijk..., c)
    )

    Δy²_cy_uy = Δy^2 * (
          ▶xyz_ccf(ijk..., ∂y_u, u) * ▶yz_acf(ijk..., ∂y_afa, c) * ▶xz_caf(ijk..., ∂x_faa, c)
        +   ∂y_aca(ijk..., v)       * ▶yz_acf(ijk..., ∂y_c², c)
        +   ▶x_caa(ijk..., ∂y_w, w) * ▶yz_acf(ijk..., ∂y_afa, c) * ∂z_aaf(ijk..., c)
    )

    Δz²_cz_uz = Δz^2 * (
          ▶x_caa(ijk..., ∂z_u, u)   * ∂z_aaf(ijk..., c) * ▶xz_caf(ijk..., ∂x_faa, c)
        + ▶y_aca(ijk..., ∂z_v, v)   * ∂z_aaf(ijk..., c) * ▶yz_acf(ijk..., ∂y_afa, c)
        + ▶z_aaf(ijk..., ∂z_aac, w) *  ∂z_c²(ijk..., c)
    )

    return Δx²_cx_ux + Δy²_cy_uy + Δz²_cz_uz
end

@inline θᵢ²_ccc(i, j, k, grid, c) = (
      ▶x_caa(i, j, k, grid, ∂x_c², c)
    + ▶y_aca(i, j, k, grid, ∂y_c², c)
    + ▶z_aac(i, j, k, grid, ∂z_c², c)
)

@inline θᵢ²_ccf(i, j, k, grid, c) = (
      ▶xz_caf(i, j, k, grid, ∂x_c², c)
    + ▶yz_acf(i, j, k, grid, ∂y_c², c)
    + ∂z_c²(i, j, k, grid, c)
)

"""
    ∇_κ_∇T(i, j, k, grid, T, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ c)` for the turbulence
`closure`, where `c` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇T(i, j, k, grid, T, closure::RAMD, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_c, T, diffusivities.κₑ.T, closure)
    + ∂y_aca(i, j, k, grid, κ_∂y_c, T, diffusivities.κₑ.T, closure)
    + ∂z_aac(i, j, k, grid, κ_∂z_c, T, diffusivities.κₑ.T, closure)
)

"""
    ∇_κ_∇S(i, j, k, grid, S, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ S)` for the turbulence
`closure`, where `c` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇S(i, j, k, grid, S, closure::RAMD, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_c, S, diffusivities.κₑ.S, closure)
    + ∂y_aca(i, j, k, grid, κ_∂y_c, S, diffusivities.κₑ.S, closure)
    + ∂z_aac(i, j, k, grid, κ_∂z_c, S, diffusivities.κₑ.S, closure)
)

# This function assumes rigid top and bottom boundary conditions
function calc_diffusivities!(K, grid, closure::RAMD, eos, grav, U, Φ)
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

#=
An idea:

function calc_z_boundary_diffusivities!(K, grid, closure, eos, grav, U, Φ)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds K.νₑ[i, j, 1]         = ν_ccf(i, j, 2,       grid, closure, nothing, eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
            @inbounds K.κₑ.T[i, j, 1]       = κ_ccf(i, j, 2,       grid, closure, Φ.T,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
            @inbounds K.κₑ.S[i, j, 1]       = κ_ccf(i, j, 2,       grid, closure, Φ.S,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)

            @inbounds K.νₑ[i, j, grid.Nz]   = ν_ccf(i, j, grid.Nz, grid, closure, nothing, eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
            @inbounds K.κₑ.T[i, j, grid.Nz] = κ_ccf(i, j, grid.Nz, grid, closure, Φ.T,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
            @inbounds K.κₑ.S[i, j, grid.Nz] = κ_ccf(i, j, grid.Nz, grid, closure, Φ.S,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
        end
    end
    return nothing
end
=#
