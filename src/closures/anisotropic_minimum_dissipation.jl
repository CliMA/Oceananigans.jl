struct AnisotropicMinimumDissipation{T} <: IsotropicDiffusivity{T}
    C :: T
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
        ν = 1e-6,
        κ = 1e-7
    )
    return AnisotropicMinimumDissipation{FT}(C, ν, κ)
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
     νₑ = zeros(arch, grid)
    κTₑ = zeros(arch, grid)
    return (νₑ=νₑ, κₑ=(T=κTₑ,))
end

@inline function ν_ccc(i, j, k, grid::Grid{FT}, closure::AnisotropicMinimumDissipation, ϕ,
                       eos, grav, u, v, w, T, S) where FT

    r = Δ²ₐ_uᵢₐ_uⱼₐ_Σᵢⱼ_ccc(i, j, k, grid, closure, u, v, w)
    ζ = Δ²ᵢ_wᵢ_bᵢ_ccc(i, j, k, grid, closure, eos, grav, w, T, S)
    q = tr_∇u_ccc(i, j, k, grid, u, v, w)

    νdagger = -closure.C * (r - ζ) / q
    #νdagger = -closure.C * r / q #(r - ζ) / q

    return max(zero(FT), νdagger) + closure.ν
end

@inline function κ_ccc(i, j, k, grid::Grid{FT}, closure::AnisotropicMinimumDissipation,
               ϕ, eos, grav, u, v, w, T, S) where FT

    n =  Δ²ⱼ_uᵢⱼ_ϕⱼ_ϕᵢ_ccc(i, j, k, grid, closure, u, v, w, ϕ)
    d =  θᵢ²_ccc(i, j, k, grid, ϕ)

    κdagger = - closure.C * n / d

    return max(zero(FT), κdagger) + closure.κ
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

@inline ∂x_ϕ²(ijk...) = ∂x_faa(ijk...)^2
@inline ∂y_ϕ²(ijk...) = ∂y_afa(ijk...)^2
@inline ∂z_ϕ²(ijk...) = ∂z_aaf(ijk...)^2

@inline function Δ²ⱼ_uᵢⱼ_ϕⱼ_ϕᵢ_ccc(i, j, k, grid, closure, ϕ, u, v, w)
    ijk = (i, j, k, grid)

    Δx = Δx_ccc(ijk..., closure)
    Δy = Δy_ccc(ijk..., closure)
    Δz = Δz_ccc(ijk..., closure)

    Δx²_ϕx_ux = Δx^2 * (
                 ∂x_caa(ijk..., u) * ▶x_caa(ijk..., ∂x_ϕ², ϕ)
        + ▶xy_cca(ijk..., ∂x_v, v) * ▶x_caa(ijk..., ∂x_faa, ϕ) * ▶y_aca(ijk..., ∂y_afa, ϕ)
        + ▶xz_cac(ijk..., ∂x_w, w) * ▶x_caa(ijk..., ∂x_faa, ϕ) * ▶z_aac(ijk..., ∂z_aaf, ϕ)
    )

    Δy²_ϕy_uy = Δy^2 * (
          ▶xy_cca(ijk..., ∂y_u, u) * ▶y_aca(ijk..., ∂y_afa, ϕ) * ▶x_caa(ijk..., ∂x_faa, ϕ)
        +        ∂y_aca(ijk..., v) * ▶y_aca(ijk..., ∂y_ϕ², ϕ)
        + ▶xz_cac(ijk..., ∂y_w, w) * ▶y_aca(ijk..., ∂y_afa, ϕ) * ▶z_aac(ijk..., ∂z_aaf, ϕ)
    )

    Δz²_ϕz_uz = Δz^2 * (
          ▶xz_cac(ijk..., ∂z_u, u) * ▶z_aac(ijk..., ∂z_aaf, ϕ) * ▶x_caa(ijk..., ∂x_faa, ϕ)
        + ▶yz_acc(ijk..., ∂z_v, v) * ▶z_aac(ijk..., ∂z_aaf, ϕ) * ▶y_aca(ijk..., ∂y_afa, ϕ)
        +        ∂z_aac(ijk..., w) * ▶z_aac(ijk..., ∂z_ϕ², ϕ)
    )

    return Δx²_ϕx_ux + Δy²_ϕy_uy + Δz²_ϕz_uz
end


@inline θᵢ²_ccc(i, j, k, grid, ϕ) = (
      ▶x_caa(i, j, k, grid, ∂x_ϕ², ϕ)
    + ▶y_aca(i, j, k, grid, ∂y_ϕ², ϕ)
    + ▶z_aac(i, j, k, grid, ∂z_ϕ², ϕ)
)

"""
    ∇_κ_∇T(i, j, k, grid, T, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ ϕ)` for the turbulence
`closure`, where `ϕ` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇T(i, j, k, grid, T, closure::AnisotropicMinimumDissipation, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_ϕ, T, diffusivities.κₑ.T, closure)
    + ∂y_aca(i, j, k, grid, κ_∂y_ϕ, T, diffusivities.κₑ.T, closure)
    + ∂z_aac(i, j, k, grid, κ_∂z_ϕ, T, diffusivities.κₑ.T, closure)
)

"""
    ∇_κ_∇S(i, j, k, grid, S, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ S)` for the turbulence
`closure`, where `ϕ` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇S(i, j, k, grid, S, closure::AnisotropicMinimumDissipation, diffusivities) = (
      ∂x_caa(i, j, k, grid, κ_∂x_ϕ, S, diffusivities.κₑ.S, closure)
    + ∂y_aca(i, j, k, grid, κ_∂y_ϕ, S, diffusivities.κₑ.S, closure)
    + ∂z_aac(i, j, k, grid, κ_∂z_ϕ, S, diffusivities.κₑ.S, closure)
)

function calc_diffusivities!(diffusivities, grid, closure::AnisotropicMinimumDissipation,
                                  eos, grav, u, v, w, T, S)

    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds diffusivities.νₑ[i, j, k]  = ν_ccc(i, j, k, grid, closure, nothing,
                                                                eos, grav, u, v, w, T, S)

                @inbounds diffusivities.κₑ.T[i, j, k] = κ_ccc(i, j, k, grid, closure, T,
                                                                eos, grav, u, v, w, T, S)

            end
        end
    end

    @synchronize
end
