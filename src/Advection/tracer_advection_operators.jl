using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans.Fields: ZeroField

@inline _advective_tracer_flux_x(args...) = advective_tracer_flux_x(args...)
@inline _advective_tracer_flux_y(args...) = advective_tracer_flux_y(args...)
@inline _advective_tracer_flux_z(args...) = advective_tracer_flux_z(args...)

@inline div_Uc(i, j, k, grid, advection, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)

@inline div_Uc(i, j, k, grid, ::Nothing, U, c) = zero(grid)
@inline div_Uc(i, j, k, grid, ::Nothing, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)

@inline div_Uc_x(i, j, k, grid, advection, ::ZeroU, c) = zero(grid)
@inline div_Uc_x(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)

@inline div_Uc_x(i, j, k, grid, ::Nothing, U, c) = zero(grid)
@inline div_Uc_x(i, j, k, grid, ::Nothing, ::ZeroU, c) = zero(grid)
@inline div_Uc_x(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)

@inline div_Uc_y(i, j, k, grid, advection, ::ZeroU, c) = zero(grid)
@inline div_Uc_y(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)

@inline div_Uc_y(i, j, k, grid, ::Nothing, U, c) = zero(grid)
@inline div_Uc_y(i, j, k, grid, ::Nothing, ::ZeroU, c) = zero(grid)
@inline div_Uc_y(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)

@inline div_Uc_z(i, j, k, grid, advection, ::ZeroU, c) = zero(grid)
@inline div_Uc_z(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)

@inline div_Uc_z(i, j, k, grid, ::Nothing, U, c) = zero(grid)
@inline div_Uc_z(i, j, k, grid, ::Nothing, ::ZeroU, c) = zero(grid)
@inline div_Uc_z(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)

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

@inline div_Uc_x(i, j, k, grid, advection, U, c) = 
    1/Vᶜᶜᶜ(i, j, k, grid) * δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, advection, U.u, c)

@inline div_Uc_y(i, j, k, grid, advection, U, c) = 
    1/Vᶜᶜᶜ(i, j, k, grid) * δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, advection, U.v, c) 

@inline div_Uc_z(i, j, k, grid, advection, U, c) = 
    1/Vᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, advection, U.w, c)
