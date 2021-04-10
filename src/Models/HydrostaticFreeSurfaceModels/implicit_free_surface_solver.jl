using Oceananigans.Solvers
using Oceananigans.Operators

"""
    implicit_free_surface_linear_operation!(result, x, arch, grid, bcs; args...)

Returns `L(ηⁿ)`, where `ηⁿ` is the free surface displacement at time step `n`
and `L` is the linear operator that arises
in an implicit time step for the free surface displacement `η`.

(See the docs section on implicit time stepping.)
"""
function implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᵃ, ∫ᶻ_Ayᶜᶠᵃ, g, Δt)
    grid = L_ηⁿ⁺¹.grid
    arch = architecture(L_ηⁿ⁺¹)

    event = launch!(arch, grid, :xy, compute_implicit_η_left_hand_side!,
                    L_ηⁿ⁺¹, grid,  ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᵃ, ∫ᶻ_Ayᶜᶠᵃ, g, Δt,
                    dependencies=Event(device(arch)))

    wait(device(arch), event)

    fill_halo_regions!(L_ηⁿ⁺¹, arch)

    return nothing
end

function ImplicitFreeSurfaceSolver(η;
                                   maximum_iterations = η.grid.Nx * η.grid.Ny,
                                   tolerance = 1e-13,
                                   precondition = nothing)

    return PreconditionedConjugateGradientSolver(implicit_free_surface_linear_operation!,
                                                 template_field = η,
                                                 maximum_iterations = maximum_iterations,
                                                 tolerance = tolerance,
                                                 preconditioner = preconditioner)
end

# Kernels that act on vertically integrated / surface quantities
@inline ∫ᶻ_Ax_∂x_ηᶠᶜᵃ(i, j, k, grid, ∫ᶻ_Axᶠᶜᵃ, η) = @inbounds ∫ᶻ_Axᶠᶜᵃ[i, j, k] * ∂xᶠᶜᵃ(i, j, k, grid, η)
@inline ∫ᶻ_Ay_∂y_ηᶜᶠᵃ(i, j, k, grid, ∫ᶻ_Ayᶜᶠᵃ, η) = @inbounds ∫ᶻ_Ayᶜᶠᵃ[i, j, k] * ∂yᶜᶠᵃ(i, j, k, grid, η)

"""
Compute the horizontal divergence of vertically-uniform quantity using
vertically-integrated face areas `∫ᶻ_Axᶠᶜᵃ` and `∫ᶻ_Ayᶜᶠᵃ`.
"""
@inline ∇h²ᶜᶜᵃ(i, j, k, grid, ∫ᶻ_Axᶠᶜᵃ, ∫ᶻ_Ayᶜᶠᵃ, η::ReducedField{X, Y, Nothing}) where {X, Y} =
    1 / Azᶜᶜᵃ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, ∫ᶻ_Ax_∂x_ηᶠᶜᵃ, ∫ᶻ_Axᶠᶜᵃ, η) +
                                δyᵃᶜᵃ(i, j, k, grid, ∫ᶻ_Ay_∂y_ηᶜᶠᵃ, ∫ᶻ_Ayᶜᶠᵃ, η))

"""
Returns the left hand side of the "implicit η" equation

```
( ∇ʰ⋅H∇ʰ - 1/gΔt² ) ηⁿ⁺¹ = 1 / (gΔt) ∇ʰH U̅ˢᵗᵃʳ - 1 / (gΔt²) ηⁿ
------------------------
        ≡ L_ηⁿ⁺¹
```

which is written in a discrete finite volume form in which the equation
is arranged to ensure a symmtric form by multiplying by horizontal areas Az:

```
δⁱÂʷ∂ˣηⁿ⁺¹ + δʲÂˢ∂ʸηⁿ⁺¹ - 1/gΔt² Az ηⁿ⁺¹ = 1 / (gΔt) (δⁱÂʷu̅ˢᵗᵃʳ + δʲÂˢv̅ˢᵗᵃʳ) - 1 / gΔt² Az ηⁿ
```

where  ̂ indicates a vertical integral, and                   
       ̅ indicates a vertical average                         
"""
@kernel function _implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, grid, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᵃ, ∫ᶻ_Ayᶜᶠᵃ, g, Δt)
    i, j = @index(Global, NTuple)

    @inbounds L_ηⁿ⁺¹[i, j, 1] = (  Azᶜᶜᵃ(i, j, 1, grid) * ∇h²ᶜᶜᵃ(i, j, 1, grid, ∫ᶻ_Axᶠᶜᵃ, ∫ᶻ_Ayᶜᶠᵃ, ηⁿ⁺¹)
                                 - Azᶜᶜᵃ(i, j, 1, grid) * ηⁿ⁺¹[i, j, 1] / (g * Δt^2))
end
