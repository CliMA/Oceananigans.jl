struct AnisotropicMinimumDissipation{T} <: IsotropicDiffusivity{T}
     C :: T
    Cb :: T
     ν :: T
     κ :: T
end

"""
    AnisotropicMinimumDissipation(T=Float64; C=0.33, ν=1e-6, κ=1e-7)

Returns a `AnisotropicMinimumDissipation` closure object of type `T` with

    * `C` : Poincaré constant
    * `ν` : 'molecular' background viscosity
    * `κ` : 'molecular' background diffusivity
"""
function AnisotropicMinimumDissipation(FT=Float64;
         C = 0.33,
        Cb = 1.0,
         ν = 1e-6,
         κ = 1e-7
    )
    return AnisotropicMinimumDissipation{FT}(C, Cb, ν, κ)
end

"Return the filter width for Anisotropic Minimum Dissipation on a Regular Cartesian grid."
@inline Δx(i, j, k, grid::RegularCartesianGrid, ::AnisotropicMinimumDissipation) = grid.Δx
@inline Δy(i, j, k, grid::RegularCartesianGrid, ::AnisotropicMinimumDissipation) = grid.Δy
@inline Δz(i, j, k, grid::RegularCartesianGrid, ::AnisotropicMinimumDissipation) = grid.Δz

# We only have regular grids for now. When we have non-regular grids this will need to be changed.
const Δx_ccc = Δx
const Δy_ccc = Δy
const Δz_ccc = Δz

function TurbulentDiffusivities(arch::Architecture, grid::Grid, ::AnisotropicMinimumDissipation)
     νₑ = CellField(arch, grid)
    κTₑ = CellField(arch, grid)
    κSₑ = CellField(arch, grid)
    return (νₑ=νₑ, κₑ=(T=κTₑ, S=κSₑ))
end

@inline function ν_ccc(i, j, k, grid::Grid{FT}, closure::AnisotropicMinimumDissipation, c,
                       eos, grav, u, v, w, T, S) where FT

    q = tr_∇u_ccc(i, j, k, grid, u, v, w)

    if q == 0
        return closure.ν
    else
        r = Δ²ₐ_uᵢₐ_uⱼₐ_Σᵢⱼ_ccc(i, j, k, grid, closure, u, v, w)
        ζ = Δ²ᵢ_wᵢ_bᵢ_ccc(i, j, k, grid, closure, eos, grav, w, T, S)
        νdagger = -closure.C * (r - closure.Cb * ζ) / q
        return max(zero(FT), νdagger) + closure.ν
    end
end

@inline function κ_ccc(i, j, k, grid::Grid{FT}, closure::AnisotropicMinimumDissipation, c,
                       eos, grav, u, v, w, T, S) where FT

    d =  θᵢ²_ccc(i, j, k, grid, c)

    if d == 0
        return closure.κ
    else
        n =  Δ²ⱼ_uᵢⱼ_cⱼ_cᵢ_ccc(i, j, k, grid, closure, u, v, w, c)
        κdagger = - closure.C * n / d
        return max(zero(FT), κdagger) + closure.κ
    end
end

#
# Same-location products
#

# ccc
@inline ∂x_u²(ijk...) = ∂x_u(ijk...)^2
@inline ∂y_v²(ijk...) = ∂y_v(ijk...)^2
@inline ∂z_w²(ijk...) = ∂z_w(ijk...)^2

# ffc
@inline ∂x_v²(ijk...) = ∂x_v(ijk...)^2
@inline ∂y_u²(ijk...) = ∂y_u(ijk...)^2

@inline ∂x_v_Σ₁₂(ijk...) = ∂x_v(ijk...) * Σ₁₂(ijk...)
@inline ∂y_u_Σ₁₂(ijk...) = ∂y_u(ijk...) * Σ₁₂(ijk...)

# fcf
@inline ∂z_u²(ijk...) = ∂z_u(ijk...)^2
@inline ∂x_w²(ijk...) = ∂x_w(ijk...)^2

@inline ∂x_w_Σ₁₃(ijk...) = ∂x_w(ijk...) * Σ₁₃(ijk...)
@inline ∂z_u_Σ₁₃(ijk...) = ∂z_u(ijk...) * Σ₁₃(ijk...)

# cff
@inline ∂z_v²(ijk...) = ∂z_v(ijk...)^2
@inline ∂y_w²(ijk...) = ∂y_w(ijk...)^2
@inline ∂z_v_Σ₂₃(ijk...) = ∂z_v(ijk...) * Σ₂₃(ijk...)
@inline ∂y_w_Σ₂₃(ijk...) = ∂y_w(ijk...) * Σ₂₃(ijk...)

#
# *** 30 terms ***
#

#
# the heinous
#

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

#
# trace(∇u) = uᵢⱼ uᵢⱼ
#

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

@inline ∂x_c²(ijk...) = ∂x_faa(ijk...)^2
@inline ∂y_c²(ijk...) = ∂y_afa(ijk...)^2
@inline ∂z_c²(ijk...) = ∂z_aaf(ijk...)^2

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


@inline θᵢ²_ccc(i, j, k, grid, c) = (
      ▶x_caa(i, j, k, grid, ∂x_c², c)
    + ▶y_aca(i, j, k, grid, ∂y_c², c)
    + ▶z_aac(i, j, k, grid, ∂z_c², c)
)

"""
    ∇_κ_∇T(i, j, k, grid, T, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ c)` for the turbulence
`closure`, where `c` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇T(i, j, k, grid, T, closure::AnisotropicMinimumDissipation, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_c, T, diffusivities.κₑ.T, closure)
    + ∂y_aca(i, j, k, grid, κ_∂y_c, T, diffusivities.κₑ.T, closure)
    + ∂z_aac(i, j, k, grid, κ_∂z_c, T, diffusivities.κₑ.T, closure)
)

"""
    ∇_κ_∇S(i, j, k, grid, S, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ S)` for the turbulence
`closure`, where `c` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇S(i, j, k, grid, S, closure::AnisotropicMinimumDissipation, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_c, S, diffusivities.κₑ.S, closure)
    + ∂y_aca(i, j, k, grid, κ_∂y_c, S, diffusivities.κₑ.S, closure)
    + ∂z_aac(i, j, k, grid, κ_∂z_c, S, diffusivities.κₑ.S, closure)
)

function calc_diffusivities!(K, grid, closure::AnisotropicMinimumDissipation, eos, grav, U, Φ)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds K.νₑ[i, j, k]   = ν_ccc(i, j, k, grid, closure, nothing, eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                @inbounds K.κₑ.T[i, j, k] = κ_ccc(i, j, k, grid, closure, Φ.T,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
                @inbounds K.κₑ.S[i, j, k] = κ_ccc(i, j, k, grid, closure, Φ.S,     eos, grav, U.u, U.v, U.w, Φ.T, Φ.S)
            end
        end
    end
end
