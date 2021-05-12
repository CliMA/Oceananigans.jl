#####                                                            
##### Viscous flux divergences
#####

@inline function ∂ⱼ_τ₁ⱼ(i, j, k, grid, ib::AbstractImmersedBoundary, clock, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1/Vᶠᶜᶜ(i, j, k, grid) * (δxᶠᵃᵃ(i, j, k, grid, Ax_cᶜᶜᶜ, viscous_flux_ux, ib, disc, clock, closure, args...) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_ζᶠᶠᶜ, viscous_flux_uy, ib, disc, clock, closure, args...) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_ηᶠᶜᵃ, viscous_flux_uz, ib, disc, clock, closure, args...))
end

@inline function ∂ⱼ_τ₂ⱼ(i, j, k, grid, ib::AbstractImmersedBoundary, clock, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1/Vᶜᶠᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_ζᶠᶠᶜ, viscous_flux_vx, ib, disc, clock, closure, args...) +
                                    δyᵃᶠᵃ(i, j, k, grid, Ay_cᶜᶜᶜ, viscous_flux_vy, ib, disc, clock, closure, args...) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_ξᶜᶠᵃ, viscous_flux_vz, ib, disc, clock, closure, args...))
end

@inline function ∂ⱼ_τ₃ⱼ(i, j, k, grid, ib::AbstractImmersedBoundary, clock, closure::AbstractTurbulenceClosure, args...)
    disc = time_discretization(closure)
    return 1/Vᶜᶜᶠ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_ηᶠᶜᶠ, viscous_flux_wx, ib, disc, clock, closure, args...) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_ξᶜᶠᶠ, viscous_flux_wy, ib, disc, clock, closure, args...) +
                                    δzᵃᵃᶠ(i, j, k, grid, Az_cᶜᶜᵃ, viscous_flux_wz, ib, disc, clock, closure, args...))
end

#####
##### Diffusive flux divergence
#####

@inline function ∇_dot_qᶜ(i, j, k, grid, ib::AbstractImmersedBoundary, clock, closure::AbstractTurbulenceClosure, c, ::Val{tracer_index}, args...) where tracer_index
    disc = time_discretization(closure)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_uᶠᶜᶜ, diffusive_flux_x, ib, disc, clock, closure, c, Val(tracer_index), args...) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_vᶜᶠᶜ, diffusive_flux_y, ib, disc, clock, closure, c, Val(tracer_index), args...) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_wᶜᶜᵃ, diffusive_flux_z, ib, disc, clock, closure, c, Val(tracer_index), args...))
end

