function div_hUu(i, j, k, grid, advection, solution)

    return 1 / Vᵃᵃᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, momentum_flux_huu, advection, solution) +
                                      δyᵃᶜᵃ(i, j, k, grid, momentum_flux_huv, advection, solution))
end

function div_hUv(i, j, k, grid, advection, solution)
    return 1 / Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, momentum_flux_hvu, advection, solution) +
                                      δyᵃᶠᵃ(i, j, k, grid, momentum_flux_hvv, advection, solution))
end

@inline momentum_flux_huu(i, j, k, grid, advection, solution) =
    @inbounds momentum_flux_uu(i, j, k, grid, advection, solution.uh, solution.uh) / solution.h[i, j, k]

@inline momentum_flux_huv(i, j, k, grid, advection, solution) =
    @inbounds momentum_flux_uv(i, j, k, grid, advection, solution.uh, solution.vh) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

@inline momentum_flux_hvu(i, j, k, grid, advection, solution) =
    @inbounds momentum_flux_vu(i, j, k, grid, advection, solution.uh, solution.vh) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

@inline momentum_flux_hvv(i, j, k, grid, advection, solution) =
    @inbounds momentum_flux_vv(i, j, k, grid, advection, solution.vh, solution.vh) / solution.h[i, j, k]


@inline ∂x₄ᶠᵃᵃ(i, j, k, grid, f::F, args...) where F<:Function = ( -f(i+1, j, k, grid, args...) + 15*f(i, j, k, grid, args...)
                                                                   - 15*f(i-1, j, k, grid, args...) + f(i-2, j, k, grid, args...)) / ( 12*Δx(i, j, k, grid) )

@inline ∂y₄ᵃᶠᵃ(i, j, k, grid, f::F, args...) where F<:Function = (- f(i, j+1, k, grid, args...) + 15*f(i, j, k, grid, args...)
                                                                  - 15*f(i, j-1, k, grid, args...) + f(i, j-1, k, grid, args...)) / ( 12*Δy(i, j, k, grid) )


@inline ∂x₄ᶜᵃᵃ(i, j, k, grid, f::F, args...) where F<:Function = (- f(i+2, j, k, grid, args...) + 15*f(i+1, j, k, grid, args...)
                                                                  - 15*f(i, j, k, grid, args...) + f(i-1, j, k, grid, args...)) / ( 12*Δx(i, j, k, grid) )

@inline ∂y₄ᵃᶜᵃ(i, j, k, grid, f::F, args...) where F<:Function = (- f(i, j+2, k, grid, args...) + 15*f(i, j+1, k, grid, args...)
                                                                  - 15*f(i, j, k, grid, args...) + f(i, j-1, k, grid, args...)) / ( 12*Δy(i, j, k, grid) )
