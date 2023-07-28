using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Architectures
using Oceananigans.Grids: with_halo, isrectilinear
using Oceananigans.Architectures: device

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

Return a solver based on a preconditioned conjugate gradient method for
the elliptic equation
    
```math
[∇ ⋅ H ∇ - 1 / (g Δt²)] ηⁿ⁺¹ = (∇ʰ ⋅ Q★ - ηⁿ / Δt) / (g Δt)
```

representing an implicit time discretization of the linear free surface evolution equation
for a fluid with variable depth `H`, horizontal areas `Az`, barotropic volume flux `Q★`, time
step `Δt`, gravitational acceleration `g`, and free surface at time-step `n` `ηⁿ`.
"""
function PCGImplicitFreeSurfaceSolver(grid::AbstractGrid, settings, gravitational_acceleration=nothing)
    
    # Initialize vertically integrated lateral face areas
    ∫ᶻ_Axᶠᶜᶜ = Field((Face, Center, Nothing), with_halo((3, 3, 1), grid))
    ∫ᶻ_Ayᶜᶠᶜ = Field((Center, Face, Nothing), with_halo((3, 3, 1), grid))

    vertically_integrated_lateral_areas = (xᶠᶜᶜ = ∫ᶻ_Axᶠᶜᶜ, yᶜᶠᶜ = ∫ᶻ_Ayᶜᶠᶜ)

    @apply_regionally compute_vertically_integrated_lateral_areas!(vertically_integrated_lateral_areas)
    fill_halo_regions!(vertically_integrated_lateral_areas)

    # Set some defaults
    settings = Dict{Symbol, Any}(settings)
    settings[:maxiter] = get(settings, :maxiter, grid.Nx * grid.Ny)
    settings[:reltol] = get(settings, :reltol, min(1e-7, 10 * sqrt(eps(eltype(grid)))))

    # FFT preconditioner for rectilinear grids, nothing otherwise.
    settings[:preconditioner] = isrectilinear(grid) ?
        get(settings, :preconditioner, FFTImplicitFreeSurfaceSolver(grid)) :
        get(settings, :preconditioner, nothing)

    # TODO: reuse solver.storage for rhs when preconditioner isa FFTImplicitFreeSurfaceSolver?
    right_hand_side = ZFaceField(grid, indices = (:, :, size(grid, 3) + 1))

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
    # Take explicit step first? We haven't found improvement from this yet, but perhaps it will
    # help eventually.
    #explicit_ab2_step_free_surface!(free_surface, model, Δt, χ)
    
    ∫ᶻA = implicit_free_surface_solver.vertically_integrated_lateral_areas
    solver = implicit_free_surface_solver.preconditioned_conjugate_gradient_solver
    
    # solve!(x, solver, b, args...) solves A*x = b for x.
    solve!(η, solver, rhs, ∫ᶻA.xᶠᶜᶜ, ∫ᶻA.yᶜᶠᶜ, g, Δt)

    return nothing
end

function compute_implicit_free_surface_right_hand_side!(rhs, implicit_solver::PCGImplicitFreeSurfaceSolver,
                                                        g, Δt, ∫ᶻQ, η)

    solver = implicit_solver.preconditioned_conjugate_gradient_solver
    arch = architecture(solver)
    grid = solver.grid

    @apply_regionally compute_regional_rhs!(rhs, arch, grid, g, Δt, ∫ᶻQ, η)

    return nothing
end

compute_regional_rhs!(rhs, arch, grid, g, Δt, ∫ᶻQ, η) =
    launch!(arch, grid, :xy,
            implicit_free_surface_right_hand_side!,
            rhs, grid, g, Δt, ∫ᶻQ, η)

""" Compute the divergence of fluxes Qu and Qv. """
@inline flux_div_xyᶜᶜᶠ(i, j, k, grid, Qu, Qv) = δxᶜᵃᵃ(i, j, k, grid, Qu) + δyᵃᶜᵃ(i, j, k, grid, Qv)

@kernel function implicit_free_surface_right_hand_side!(rhs, grid, g, Δt, ∫ᶻQ, η)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz + 1
    Az = Azᶜᶜᶠ(i, j, k_top, grid)
    δ_Q = flux_div_xyᶜᶜᶠ(i, j, k_top, grid, ∫ᶻQ.u, ∫ᶻQ.v)
    @inbounds rhs[i, j, k_top] = (δ_Q - Az * η[i, j, k_top] / Δt) / (g * Δt)
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

    # REMEMBER!!! This is going to create problems!!!!
    fill_halo_regions!(ηⁿ⁺¹)

    launch!(arch, grid, :xy, _implicit_free_surface_linear_operation!,
            L_ηⁿ⁺¹, grid,  ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

    return nothing
end

# Kernels that act on vertically integrated / surface quantities
@inline ∫ᶻ_Ax_∂x_ηᶠᶜᶜ(i, j, k, grid, ∫ᶻ_Axᶠᶜᶜ, η) = @inbounds ∫ᶻ_Axᶠᶜᶜ[i, j, k] * ∂xᶠᶜᶠ(i, j, k, grid, η)
@inline ∫ᶻ_Ay_∂y_ηᶜᶠᶜ(i, j, k, grid, ∫ᶻ_Ayᶜᶠᶜ, η) = @inbounds ∫ᶻ_Ayᶜᶠᶜ[i, j, k] * ∂yᶜᶠᶠ(i, j, k, grid, η)

"""
    _implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, grid, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

Return the left side of the "implicit η equation"

```math
(∇ʰ⋅ H ∇ʰ - 1 / (g Δt²)) ηⁿ⁺¹ = 1 / (g Δt) ∇ʰ ⋅ Q★ - 1 / (g Δt²) ηⁿ
----------------------
        ≡ L_ηⁿ⁺¹
```

which is derived from the discretely summed barotropic mass conservation equation,
and arranged in a symmetric form by multiplying by horizontal areas Az:

```
δⁱÂʷ∂ˣηⁿ⁺¹ + δʲÂˢ∂ʸηⁿ⁺¹ - Az ηⁿ⁺¹ / (g Δt²) = 1 / (g Δt) (δⁱÂʷu̅ˢᵗᵃʳ + δʲÂˢv̅ˢᵗᵃʳ) - Az ηⁿ / (g Δt²) 
```

where  ̂ indicates a vertical integral, and                   
       ̅ indicates a vertical average                         
"""
@kernel function _implicit_free_surface_linear_operation!(L_ηⁿ⁺¹, grid, ηⁿ⁺¹, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz + 1
    Az = Azᶜᶜᶜ(i, j, grid.Nz, grid)
    @inbounds L_ηⁿ⁺¹[i, j, k_top] = Az_∇h²ᶜᶜᶜ(i, j, k_top, grid, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, ηⁿ⁺¹) - Az * ηⁿ⁺¹[i, j, k_top] / (g * Δt^2)
end

#####
##### Preconditioners
#####

"""
add to the rhs - H⁻¹ ∇H ⋅ ∇ηⁿ to the rhs...
"""
@inline function precondition!(P_r, preconditioner::FFTImplicitFreeSurfaceSolver, r, η, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)
    poisson_solver = preconditioner.fft_poisson_solver
    arch = architecture(poisson_solver)
    grid = preconditioner.three_dimensional_grid
    Az = grid.Δxᶜᵃᵃ * grid.Δyᵃᶜᵃ # assume horizontal regularity
    Lz = grid.Lz 

    launch!(arch, grid, :xy,
            fft_preconditioner_right_hand_side!,
            poisson_solver.storage, r, η, grid, Az, Lz)


    return solve!(P_r, preconditioner, poisson_solver.storage, g, Δt)
end

@kernel function fft_preconditioner_right_hand_side!(fft_rhs, pcg_rhs, η, grid, Az, Lz)
    i, j = @index(Global, NTuple)
    @inbounds fft_rhs[i, j, 1] = pcg_rhs[i, j, grid.Nz+1] / (Lz * Az)
end

# TODO: make it so adding this term:
#
#   - ∇H_∇η(i, j, 1, grid, η) / H
#
# speeds up the convergence.
#=
@inline ∇H_∇η(i, j, k, grid, η) = zero(grid) # fallback
@inline depth(i, j, k, grid) = grid.Lz

const GFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBottom}

# Assumes surface is z=0:
@inline depth(i, j, k, ibg::GFBIBG) = @inbounds max(zero(eltype(ibg)), min(ibg.Lz, -ibg.immersed_boundary.bottom[i, j]))
@inline ∂x_H_∂x_η(i, j, k, ibg, η) = ∂xᶠᶜᶜ(i, j, k, ibg, depth) * ∂xᶠᶜᶜ(i, j, k, ibg, η)
@inline ∂y_H_∂y_η(i, j, k, ibg, η) = ∂yᶜᶠᶜ(i, j, k, ibg, depth) * ∂yᶜᶠᶜ(i, j, k, ibg, η)
@inline ∇H_∇η(i, j, k, ibg::GFBIBG, η) = ℑxᶜᵃᵃ(i, j, k, ibg, ∂x_H_∂x_η, η) + ℑyᵃᶜᵃ(i, j, k, ibg, ∂y_H_∂y_η, η)

@inline function H⁻¹_∇H_∇η(i, j, k, ibg::GFBIBG, η)
    H = depth(i, j, k, ibg)
    return ifelse(H == 0, zero(eltype(ibg)), ∇H_∇η(i, j, k, ibg, η) / H)
end

# The rhs below becomes pcg_rhs[i, j, 1] / (H * Az) - ∇H_∇η(i, j, 1, grid, η) / H
=#

#####
##### "Asymptotically diagonally-dominant" preconditioner
#####

struct DiagonallyDominantInversePreconditioner end

@inline precondition!(P_r, ::DiagonallyDominantInversePreconditioner, r, η, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt) =
    diagonally_dominant_precondition!(P_r, r, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

"""
    _diagonally_dominant_precondition!(P_r, grid, r, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

Return the diagonally dominant inverse preconditioner applied to the residuals consistently with
 `M = D⁻¹(I - (A - D)D⁻¹) ≈ A⁻¹` where `I` is the Identity matrix,
A is the linear operator applied to η, and D is the diagonal of A.

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

    launch!(arch, grid, :xy, _diagonally_dominant_precondition!,
            P_r, grid, r, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

    return nothing
end

# Kernels that calculate coefficients for the preconditioner
@inline Ax⁻(i, j, grid, ax) = @inbounds   ax[i, j, 1] / Δxᶠᶜᶠ(i, j, grid.Nz+1, grid)
@inline Ay⁻(i, j, grid, ay) = @inbounds   ay[i, j, 1] / Δyᶜᶠᶠ(i, j, grid.Nz+1, grid)
@inline Ax⁺(i, j, grid, ax) = @inbounds ax[i+1, j, 1] / Δxᶠᶜᶠ(i+1, j, grid.Nz+1, grid)
@inline Ay⁺(i, j, grid, ay) = @inbounds ay[i, j+1, 1] / Δyᶜᶠᶠ(i, j+1, grid.Nz+1, grid)

@inline Ac(i, j, grid, g, Δt, ax, ay) = - Ax⁻(i, j, grid, ax) -
                                          Ax⁺(i, j, grid, ax) -
                                          Ay⁻(i, j, grid, ay) -
                                          Ay⁺(i, j, grid, ay) - 
                                          Azᶜᶜᶜ(i, j, 1, grid) / (g * Δt^2)

@inline heuristic_inverse_times_residuals(i, j, r, grid, g, Δt, ax, ay) =
    @inbounds 1 / Ac(i, j, grid, g, Δt, ax, ay) * (r[i, j, 1] - 2 * Ax⁻(i, j, grid, ax) / (Ac(i-1, j, grid, g, Δt, ax, ay) + Ac(i, j, grid, g, Δt, ax, ay)) * r[i-1, j, grid.Nz+1] -
                                                                2 * Ax⁺(i, j, grid, ax) / (Ac(i+1, j, grid, g, Δt, ax, ay) + Ac(i, j, grid, g, Δt, ax, ay)) * r[i+1, j, grid.Nz+1] - 
                                                                2 * Ay⁻(i, j, grid, ay) / (Ac(i, j-1, grid, g, Δt, ax, ay) + Ac(i, j, grid, g, Δt, ax, ay)) * r[i, j-1, grid.Nz+1] - 
                                                                2 * Ay⁺(i, j, grid, ay) / (Ac(i, j+1, grid, g, Δt, ax, ay) + Ac(i, j, grid, g, Δt, ax, ay)) * r[i, j+1, grid.Nz+1])

@kernel function _diagonally_dominant_precondition!(P_r, grid, r, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)
    i, j = @index(Global, NTuple)
    @inbounds P_r[i, j, grid.Nz+1] = heuristic_inverse_times_residuals(i, j, r, grid, g, Δt, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ)
end
