const ArrayOrField = Union{AbstractArray, AbstractField}

#####
##### Diffusivities
#####

κᶠᶜᶜ(i, j, k, grid, clock, κ::Number) = κ
κᶜᶠᶜ(i, j, k, grid, clock, κ::Number) = κ
κᶜᶜᶠ(i, j, k, grid, clock, κ::Number) = κ

κᶠᶜᶜ(i, j, k, grid, clock, κ::Function) = κ(xnode(Face,   i, grid), ynode(Center, j, grid), znode(Center, k, grid), clock.time)
κᶜᶠᶜ(i, j, k, grid, clock, κ::Function) = κ(xnode(Center, i, grid), ynode(Face,   j, grid), znode(Center, k, grid), clock.time)
κᶜᶜᶠ(i, j, k, grid, clock, κ::Function) = κ(xnode(Center, i, grid), ynode(Center, j, grid), znode(Face,   k, grid), clock.time)

# Assumes that `κ` is located at cell centers
κᶠᶜᶜ(i, j, k, grid, clock, κ::ArrayOrField) = ℑxᶠᵃᵃ(i, j, k, grid, κ)
κᶜᶠᶜ(i, j, k, grid, clock, κ::ArrayOrField) = ℑyᵃᶠᵃ(i, j, k, grid, κ)
κᶜᶜᶠ(i, j, k, grid, clock, κ::ArrayOrField) = ℑzᵃᵃᶠ(i, j, k, grid, κ)

#####
##### Diffusive fluxes
#####

@inline diffusive_flux_x(i, j, k, grid, clock, κ, c) = κᶠᶜᶜ(i, j, k, grid, clock, κ) * ∂xᶠᶜᵃ(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, clock, κ, c) = κᶜᶠᶜ(i, j, k, grid, clock, κ) * ∂yᶜᶠᵃ(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, clock, κ, c) = κᶜᶜᶠ(i, j, k, grid, clock, κ) * ∂zᵃᵃᶠ(i, j, k, grid, c)

#####
##### Diffusive flux divergence
#####

"""
    ∇_κ_∇c(i, j, k, grid, clock, closure::AbstractTurbulenceClosure, c, ::Val{tracer_index}, args...)

Calculates diffusion for a tracer c via

    1/V * [δxᶜᵃᵃ(Ax * diffusive_flux_x) + δyᵃᶜᵃ(Ay * diffusive_flux_y) + δzᵃᵃᶜ(Az * diffusive_flux_z)]

which will end up at the location `ccc`.
"""
@inline function ∇_κ_∇c(i, j, k, grid, clock, closure::AbstractTurbulenceClosure, c, ::Val{tracer_index}, args...) where tracer_index
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_uᶠᶜᶜ, diffusive_flux_x, clock, closure, c, Val(tracer_index), args...) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_vᶜᶠᶜ, diffusive_flux_y, clock, closure, c, Val(tracer_index), args...) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_wᶜᶜᵃ, diffusive_flux_z, clock, closure, c, Val(tracer_index), args...))
end
