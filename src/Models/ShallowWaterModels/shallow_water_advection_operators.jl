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


function div_UV(i, j, k, grid, advection, solution)
    
    #u = solution.uh[i,j,k]/ℑxᶠᵃᵃ(i, j, k, grid, solution.h)
    #v = solution.vh/ℑyᵃᶠᵃ(i, j, k, grid, solution.h)

    u = 1
    v = 0

    #print(i, " ", j, " ", k, " u ", solution.uh[i,j,k], "\n")

    return 1/Vᵃᵃᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, advective_tracer_flux_x, advection, u, solution.h)  +
                                    δyᵃᶜᵃ(i, j, k, grid, advective_tracer_flux_y, advection, v, solution.h) )
end
