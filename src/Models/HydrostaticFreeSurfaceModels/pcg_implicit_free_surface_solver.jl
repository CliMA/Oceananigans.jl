using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.Architectures
using Oceananigans.Grids: with_halo
using Oceananigans.Fields: Field, ZReducedField

import Oceananigans.Solvers: solve!, precondition!
import Oceananigans.Architectures: architecture

"""
    struct PCGImplicitFreeSurfaceSolver{V, S, R}

The preconditioned conjugate gradient iterative implicit free-surface solver.

$(TYPEDFIELDS)
"""
struct PCGImplicitFreeSurfaceSolver{V, S, R}
    "The vertically-integrated lateral areas"
    vertically_integrated_lateral_areas :: V
    "The preconditioned conjugate gradient solver"
    preconditioned_conjugate_gradient_solver :: S
    "The right hand side of the free surface evolution equation"
    right_hand_side :: R
end

architecture(solver::PCGImplicitFreeSurfaceSolver) =
    architecture(solver.preconditioned_conjugate_gradient_solver)

"""
    PCGImplicitFreeSurfaceSolver(grid, settings)

Return a solver based on a preconditioned conjugate gradient method for the elliptic equation
    
```math
[∇ ⋅ H ∇ - Az / (g Δt²)] ηⁿ⁺¹ = (∇ʰ ⋅ Q★ - Az ηⁿ / Δt) / (g Δt)
```

representing an implicit time discretization of the linear free surface evolution equation
for a fluid with variable depth `H`, horizontal areas `Az`, barotropic volume flux `Q★`, time
step `Δt`, gravitational acceleration `g`, and free surface at time-step `n` `ηⁿ`.
"""
function PCGImplicitFreeSurfaceSolver(grid::AbstractGrid, settings, gravitational_acceleration=nothing)
    # Initialize vertically integrated lateral face areas
    ∫ᶻ_Axᶠᶜᶜ = Field{Face, Center, Nothing}(with_halo((3, 3, 1), grid))
    ∫ᶻ_Ayᶜᶠᶜ = Field{Center, Face, Nothing}(with_halo((3, 3, 1), grid))

    vertically_integrated_lateral_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    compute_vertically_integrated_lateral_areas!(vertically_integrated_lateral_areas)

    # Set maximum iterations to Nx * Ny if not set
    settings = Dict{Symbol, Any}(settings)
    maximum_iterations = get(settings, :maximum_iterations, grid.Nx * grid.Ny)
    settings[:maximum_iterations] = maximum_iterations

    # Set preconditioner to default preconditioner if not specified
    #preconditioner = get(settings, :preconditioner, DiagonallyDominantPreconditioner())
    preconditioner = get(settings, :preconditioner, nothing)
    settings[:preconditioner] = preconditioner

    # TODO: reuse solver.storage for rhs when preconditioner isa FFTImplicitFreeSurfaceSolver
    right_hand_side = Field{Center, Center, Nothing}(grid)

    solver = PreconditionedConjugateGradientSolver(implicit_free_surface_linear_operation!;
                                                   template_field = right_hand_side,
                                                   settings...)

    return PCGImplicitFreeSurfaceSolver(vertically_integrated_lateral_areas, solver, right_hand_side)
end

build_implicit_step_solver(::Val{:PreconditionedConjugateGradient}, grid, settings, gravitational_acceleration) =
    PCGImplicitFreeSurfaceSolver(grid, settings, gravitational_acceleration)

#####
##### Solve...
#####

function solve!(η, implicit_free_surface_solver::PCGImplicitFreeSurfaceSolver, rhs, g, Δt)
    #=
    # Somehow take an explicit step first to produce an improved initial guess
    # for η for the iterative solver.
    event = explicit_ab2_step_free_surface!(free_surface, model, Δt, χ)
    wait(device(model.architecture), event)
    =#

    ∫ᶻA = implicit_free_surface_solver.vertically_integrated_lateral_areas
    solver = implicit_free_surface_solver.preconditioned_conjugate_gradient_solver

    # solve!(x, solver, b, args...) solves A*x = b for x.
    solve!(η, solver, rhs, ∫ᶻA.xᶠᶜᶜ, ∫ᶻA.yᶜᶠᶜ, g, Δt)

    return η
end

function compute_implicit_free_surface_right_hand_side!(rhs,
                                                        implicit_solver::PCGImplicitFreeSurfaceSolver,
                                                        g, Δt, ∫ᶻQ, η)

    solver = implicit_solver.preconditioned_conjugate_gradient_solver
    arch = architecture(solver)
    grid = solver.grid

    event = launch!(arch, grid, :xy,
                    implicit_free_surface_right_hand_side!,
                    rhs, grid, g, Δt, ∫ᶻQ, η,
		            dependencies = device_event(arch))

    return event
end

""" Compute the divergence of fluxes Qu and Qv. """
@inline flux_div_xyᶜᶜᶜ(i, j, k, grid, Qu, Qv) = δxᶜᵃᵃ(i, j, k, grid, Qu) + δyᵃᶜᵃ(i, j, k, grid, Qv)

@kernel function implicit_free_surface_right_hand_side!(rhs, grid, g, Δt, ∫ᶻQ, η)
    i, j = @index(Global, NTuple)
    Az = Azᶜᶜᶜ(i, j, 1, grid)
    δ_Q = flux_div_xyᶜᶜᶜ(i, j, 1, grid, ∫ᶻQ.u, ∫ᶻQ.v)
    @inbounds rhs[i, j, 1] = (δ_Q - Az * η[i, j, 1] / Δt) / (g * Δt)
end

"""
Returns `L(ηⁿ)`, where `ηⁿ` is the free surface displacement at time step `n`
and `L` is the linear operator that arises
in an implicit time step for the free surface displacement `η`.

(See the docs section on implicit time stepping.)
"""
function implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)
    grid = L_ηⁿ⁺¹.grid
    arch = architecture(L_ηⁿ⁺¹)
    fill_halo_regions!(ηⁿ⁺¹)

    event = launch!(arch, grid, :xy, _implicit_free_surface_linear_operation!,
                    L_ηⁿ⁺¹, grid,  ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

# Kernels that act on vertically integrated / surface quantities
@inline ∫ᶻ_Ax_∂x_ηᶠᶜᶜ(i, j, k, grid, ∫ᶻ_Axᶠᶜᶜ, η) = @inbounds ∫ᶻ_Axᶠᶜᶜ[i, j, k] * ∂xᶠᶜᶜ(i, j, k, grid, η)
@inline ∫ᶻ_Ay_∂y_ηᶜᶠᶜ(i, j, k, grid, ∫ᶻ_Ayᶜᶠᶜ, η) = @inbounds ∫ᶻ_Ayᶜᶠᶜ[i, j, k] * ∂yᶜᶠᶜ(i, j, k, grid, η)

"""
Compute the horizontal divergence of vertically-uniform quantity using
vertically-integrated face areas `∫ᶻ_Axᶠᶜᶜ` and `∫ᶻ_Ayᶜᶠᶜ`.
"""
@inline Az_∇h²ᶜᶜᶜ(i, j, k, grid, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, η::ZReducedField) =
    (δxᶜᵃᵃ(i, j, k, grid, ∫ᶻ_Ax_∂x_ηᶠᶜᶜ, ∫ᶻ_Axᶠᶜᶜ, η) +
     δyᵃᶜᵃ(i, j, k, grid, ∫ᶻ_Ay_∂y_ηᶜᶠᶜ, ∫ᶻ_Ayᶜᶠᶜ, η))

"""
    _implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, grid, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

Return the left side of the "implicit η equation"

```math
(∇ʰ⋅H∇ʰ - 1 / (g Δt²)) ηⁿ⁺¹ = 1 / (g Δt) ∇ʰ ⋅ Q★ - 1 / (g Δt²) ηⁿ
----------------------
        ≡ L_ηⁿ⁺¹
```

which is derived from the discretely summed barotropic mass conservation equation,
and arranged in a symmetric form by multiplying by horizontal areas Az:

```
δⁱÂʷ∂ˣηⁿ⁺¹ + δʲÂˢ∂ʸηⁿ⁺¹ - 1/gΔt² Az ηⁿ⁺¹ = 1 / (gΔt) (δⁱÂʷu̅ˢᵗᵃʳ + δʲÂˢv̅ˢᵗᵃʳ) - 1 / gΔt² Az ηⁿ
```

where  ̂ indicates a vertical integral, and                   
       ̅ indicates a vertical average                         
"""
@kernel function _implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, grid, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)
    i, j = @index(Global, NTuple)
    Az = Azᶜᶜᶜ(i, j, 1, grid)
    @inbounds L_ηⁿ⁺¹[i, j, 1] = Az_∇h²ᶜᶜᶜ(i, j, 1, grid, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, ηⁿ⁺¹) - Az * ηⁿ⁺¹[i, j, 1] / (g * Δt^2)
end

#####
##### Preconditioners
#####

@inline function precondition!(P_r, preconditioner::FFTImplicitFreeSurfaceSolver, r, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)
    solver = preconditioner.fft_based_poisson_solver
    solver.storage .= interior(r, :, :, 1)
    return solve!(P_r, preconditioner, solver.storage, g, Δt)
end

#####
##### "Asymptotically diagonally-dominant" preconditioner
#####

struct DiagonallyDominantInversePreconditioner end

@inline precondition!(P_r, ::DiagonallyDominantInversePreconditioner, r, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt) =
    diagonally_dominant_precondition!(P_r, r, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

"""
    _diagonally_dominant_precondition!(P_r, grid, r, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

Return the diagonally dominant inverse preconditioner applied to the residuals consistently with
 `M = D⁻¹(I - (A - D)D⁻¹) ≈ A⁻¹` where `I` is the Identity matrix, D is the matrix
containing the diagonal of A, and A is the linear operator applied to η

```math
P_r = M * r
```
which expanded in components is
```math
P_rᵢⱼ = rᵢⱼ / Acᵢⱼ - 1 / Acᵢⱼ ( Ax⁻ / Acᵢ₋₁ rᵢ₋₁ⱼ + Ax⁺ / Acᵢ₊₁ rᵢ₊₁ⱼ + Ay⁻ / Acⱼ₋₁ rᵢⱼ₋₁+ Ay⁺ / Acⱼ₊₁ rᵢⱼ₊₁ )
```

where `Ac`, `Ax⁻`, `Ax⁺`, `Ay⁻` and `Ay⁺` are the coefficients of 
`ηᵢⱼ`, `ηᵢ₋₁ⱼ`, `ηᵢ₊₁ⱼ`, `ηᵢⱼ₋₁` and `ηᵢⱼ₊₁` in `_implicit_free_surface_linear_operation!`
"""
function diagonally_dominant_precondition!(P_r, r, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)
    grid = ∫ᶻ_Axᶠᶜᶜ.grid
    arch = architecture(P_r)

    fill_halo_regions!(r)

    event = launch!(arch, grid, :xy, _diagonally_dominant_precondition!,
                    P_r, grid, r, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

# Kernels that calculate coefficients for the preconditioner
@inline Ax⁻(i, j, grid, ax) = @inbounds   ax[i, j, 1] / Δxᶠᶜᶜ(i, j, 1, grid)
@inline Ay⁻(i, j, grid, ay) = @inbounds   ay[i, j, 1] / Δyᶜᶠᶜ(i, j, 1, grid)
@inline Ax⁺(i, j, grid, ax) = @inbounds ax[i+1, j, 1] / Δxᶠᶜᶜ(i+1, j, 1, grid)
@inline Ay⁺(i, j, grid, ay) = @inbounds ay[i, j+1, 1] / Δyᶜᶠᶜ(i, j+1, 1, grid)

@inline Ac(i, j, grid, g, Δt, ax, ay) = - ( Ax⁻(i, j, grid, ax) 
                                          + Ax⁺(i, j, grid, ax)
                                          + Ay⁻(i, j, grid, ay)
                                          + Ay⁺(i, j, grid, ay)
                                          + Azᶜᶜᶜ(i, j, 1, grid) / (g * Δt^2) )

@inline heuristic_inverse_times_residuals(i, j, r, grid, g, Δt, ax, ay) = @inbounds 1 / Ac(i, j, grid, g, Δt, ax, ay) * ( r[i, j, 1] - 
                            Ax⁻(i, j, grid, ax) / Ac(i-1, j, grid, g, Δt, ax, ay) * r[i-1, j, 1] -
                            Ax⁺(i, j, grid, ax) / Ac(i+1, j, grid, g, Δt, ax, ay) * r[i+1, j, 1] - 
                            Ay⁻(i, j, grid, ay) / Ac(i, j-1, grid, g, Δt, ax, ay) * r[i, j-1, 1] - 
                            Ay⁺(i, j, grid, ay) / Ac(i, j+1, grid, g, Δt, ax, ay) * r[i, j+1, 1] ) 

@kernel function _diagonally_dominant_precondition!(P_r, grid, r, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)
    i, j = @index(Global, NTuple)
    @inbounds P_r[i, j, 1] = heuristic_inverse_times_residuals(i, j, r, grid, g, Δt, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ)
end
