using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans.Fields: ZeroField

struct TracerAdvection{N, FT, A, B, C} <: AbstractAdvectionScheme{N, FT}
    x :: A
    y :: B
    z :: C

    TracerAdvection{N, FT}(x::A, y::B, z::C) where {N, FT, A, B, C} = new{N, FT, A, B, C}(x, y, z)
end

"""
    function TracerAdvection(x, y, z)

Builds a `TracerAdvection` type with reconstructions schemes `x`, `y`, and `z` to be applied in
the x, y, and z direction, respectively.
"""
function TracerAdvection(x_advection, y_advection, z_advection)
    Hx = required_halo_size(x_advection)
    Hy = required_halo_size(y_advection)
    Hz = required_halo_size(z_advection)

    FT = eltype(x_advection)
    H = max(Hx, Hy, Hz)

    return TracerAdvection{H, FT}(x_advection, y_advection, z_advection)
end

@inline _advective_tracer_flux_x(args...) = advective_tracer_flux_x(args...)
@inline _advective_tracer_flux_y(args...) = advective_tracer_flux_y(args...)
@inline _advective_tracer_flux_z(args...) = advective_tracer_flux_z(args...)

@inline div_Uc(i, j, k, grid, advection, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)

@inline div_Uc(i, j, k, grid, ::Nothing, U, c) = zero(grid)
@inline div_Uc(i, j, k, grid, ::Nothing, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)

#####
##### Tracer advection operator
#####


"""
    div_uc(i, j, k, grid, advection, U, c)

Calculate the divergence of the flux of a tracer quantity ``c`` being advected by
a velocity field, ``𝛁⋅(𝐯 c)``,

```
1/V * [δxᶜᵃᵃ(Ax * u * ℑxᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * v * ℑyᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * w * ℑzᵃᵃᶠ(c))]
```
which ends up at the location `ccc`.
"""
@inline function div_Uc(i, j, k, grid, advection, U, c)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, advection, U.u, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, advection, U.v, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, advection, U.w, c))
end

@inline function div_Uc(i, j, k, grid, advection::TracerAdvection, U, c)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, advection.x, U.u, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, advection.y, U.v, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, advection.z, U.w, c))
end
