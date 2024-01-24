using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Grids: topology
using Oceananigans.Utils
using Oceananigans.AbstractOperations: Δz  
using Oceananigans.BoundaryConditions
using Oceananigans.Operators
using Oceananigans.Architectures: convert_args
using Oceananigans.ImmersedBoundaries: peripheral_node, immersed_inactive_node, GFBIBG
using Oceananigans.ImmersedBoundaries: inactive_node, IBG, c, f, ZColumnMap
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, use_only_active_surface_cells, use_only_active_interior_cells
using Oceananigans.ImmersedBoundaries: active_linear_index_to_tuple, ActiveCellsIBG, ActiveZColumnsIBGIBG
using Oceananigans.DistributedComputations: child_architecture
using Oceananigans.DistributedComputations: Distributed
using Printf

# constants for AB3 time stepping scheme (from https://doi.org/10.1016/j.ocemod.2004.08.002)
const β = 0.281105
const α = 1.5 + β
const θ = - 0.5 - 2β

const γ = 0.088
const δ = 0.614
const ϵ = 0.013
const μ = 1.0 - δ - γ - ϵ

# Evolution Kernels
#
# ∂t(η) = -∇⋅U
# ∂t(U) = - gH∇η + f
# 
# the free surface field η and its average η̄ are located on `Face`s at the surface (grid.Nz +1). All other intermediate variables
# (U, V, Ū, V̄) are barotropic fields (`ReducedField`) for which a k index is not defined

# Operators specific to the advancement of the Free surface and the Barotropic velocity. In particular, the base operators follow
# these rules:
#
#   `δxᶠᵃᵃ_η` : Hardcodes Noflux or Periodic boundary conditions for the free surface η in x direction 
#   `δyᵃᶠᵃ_η` : Hardcodes Noflux or Periodic boundary conditions for the free surface η in y direction
#
#   `δxᶜᵃᵃ_U` : Hardcodes NoPenetration or Periodic boundary conditions for the zonal barotropic velocity U in x direction 
#   `δyᵃᶜᵃ_V` : Hardcodes NoPenetration or Periodic boundary conditions for the meridional barotropic velocity V in y direction
#
# The functions `η★` `U★` and `V★` represent the value of free surface, barotropic zonal and meridional velocity at time step m+1/2
@inline δxᶠᵃᵃ_η(i, j, k, grid, T, η★::Function, args...) = δxᶠᵃᵃ(i, j, k, grid, η★, args...)
@inline δyᵃᶠᵃ_η(i, j, k, grid, T, η★::Function, args...) = δyᵃᶠᵃ(i, j, k, grid, η★, args...)
@inline δxᶜᵃᵃ_U(i, j, k, grid, T, U★::Function, args...) = δxᶜᵃᵃ(i, j, k, grid, U★, args...)
@inline δyᵃᶜᵃ_V(i, j, k, grid, T, V★::Function, args...) = δyᵃᶜᵃ(i, j, k, grid, V★, args...)

@inline δxᶠᵃᵃ_η(i, j, k, grid, ::Type{Periodic}, η★::Function, args...) = ifelse(i == 1, η★(1, j, k, grid, args...) - η★(grid.Nx, j, k, grid, args...), δxᶠᵃᵃ(i, j, k, grid, η★, args...))
@inline δyᵃᶠᵃ_η(i, j, k, grid, ::Type{Periodic}, η★::Function, args...) = ifelse(j == 1, η★(i, 1, k, grid, args...) - η★(i, grid.Ny, k, grid, args...), δyᵃᶠᵃ(i, j, k, grid, η★, args...))

@inline δxᶜᵃᵃ_U(i, j, k, grid, ::Type{Periodic}, U★::Function, args...) = ifelse(i == grid.Nx, U★(1, j, k, grid, args...) - U★(grid.Nx, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, U★, args...))
@inline δyᵃᶜᵃ_V(i, j, k, grid, ::Type{Periodic}, V★::Function, args...) = ifelse(j == grid.Ny, V★(i, 1, k, grid, args...) - V★(i, grid.Ny, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, V★, args...))

# Enforce NoFlux conditions for `η★`

@inline δxᶠᵃᵃ_η(i, j, k, grid, ::Type{Bounded},        η★::Function, args...) = ifelse(i == 1, zero(grid), δxᶠᵃᵃ(i, j, k, grid, η★, args...))
@inline δyᵃᶠᵃ_η(i, j, k, grid, ::Type{Bounded},        η★::Function, args...) = ifelse(j == 1, zero(grid), δyᵃᶠᵃ(i, j, k, grid, η★, args...))
@inline δxᶠᵃᵃ_η(i, j, k, grid, ::Type{RightConnected}, η★::Function, args...) = ifelse(i == 1, zero(grid), δxᶠᵃᵃ(i, j, k, grid, η★, args...))
@inline δyᵃᶠᵃ_η(i, j, k, grid, ::Type{RightConnected}, η★::Function, args...) = ifelse(j == 1, zero(grid), δyᵃᶠᵃ(i, j, k, grid, η★, args...))

# Enforce Impenetrability conditions for `U★` and `V★`

@inline δxᶜᵃᵃ_U(i, j, k, grid, ::Type{Bounded},  U★::Function, args...) = ifelse(i == grid.Nx, - U★(i, j, k, grid, args...),
                                                                          ifelse(i == 1, U★(2, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, U★, args...)))
@inline δyᵃᶜᵃ_V(i, j, k, grid, ::Type{Bounded},  V★::Function, args...) = ifelse(j == grid.Ny, - V★(i, j, k, grid, args...),
                                                                          ifelse(j == 1, V★(i, 2, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, V★, args...)))

@inline δxᶜᵃᵃ_U(i, j, k, grid, ::Type{LeftConnected},  U★::Function, args...) = ifelse(i == grid.Nx, - U★(i, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, U★, args...))
@inline δyᵃᶜᵃ_V(i, j, k, grid, ::Type{LeftConnected},  V★::Function, args...) = ifelse(j == grid.Ny, - V★(i, j, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, V★, args...))

@inline δxᶜᵃᵃ_U(i, j, k, grid, ::Type{RightConnected},  U★::Function, args...) = ifelse(i == 1, U★(2, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, U★, args...))
@inline δyᵃᶜᵃ_V(i, j, k, grid, ::Type{RightConnected},  V★::Function, args...) = ifelse(j == 1, V★(i, 2, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, V★, args...))

# Derivative Operators

@inline ∂xᶠᶜᶠ_η(i, j, k, grid, T, η★::Function, args...) = δxᶠᵃᵃ_η(i, j, k, grid, T, η★, args...) / Δxᶠᶜᶠ(i, j, k, grid)
@inline ∂yᶜᶠᶠ_η(i, j, k, grid, T, η★::Function, args...) = δyᵃᶠᵃ_η(i, j, k, grid, T, η★, args...) / Δyᶜᶠᶠ(i, j, k, grid)

@inline div_xᶜᶜᶠ_U(i, j, k, grid, TX, U★, args...) =  1 / Azᶜᶜᶠ(i, j, k, grid) * δxᶜᵃᵃ_U(i, j, k, grid, TX, Δy_qᶠᶜᶠ, U★, args...) 
@inline div_yᶜᶜᶠ_V(i, j, k, grid, TY, V★, args...) =  1 / Azᶜᶜᶠ(i, j, k, grid) * δyᵃᶜᵃ_V(i, j, k, grid, TY, Δx_qᶜᶠᶠ, V★, args...) 

# Immersed Boundary Operators (Velocities are `0` on `peripheral_node`s and the free surface should ensure no-flux on `inactive_node`s)

@inline conditional_U_fcc(i, j, k, grid, ibg::IBG, U★::Function, args...) = ifelse(peripheral_node(i, j, k, ibg, f, c, c), zero(ibg), U★(i, j, k, grid, args...))
@inline conditional_V_cfc(i, j, k, grid, ibg::IBG, V★::Function, args...) = ifelse(peripheral_node(i, j, k, ibg, c, f, c), zero(ibg), V★(i, j, k, grid, args...))

@inline conditional_∂xᶠᶜᶠ_η(i, j, k, ibg::IBG, args...) = ifelse(inactive_node(i, j, k, ibg, c, c, f) | inactive_node(i-1, j, k, ibg, c, c, f), zero(ibg), ∂xᶠᶜᶠ_η(i, j, k, ibg.underlying_grid, args...))
@inline conditional_∂yᶜᶠᶠ_η(i, j, k, ibg::IBG, args...) = ifelse(inactive_node(i, j, k, ibg, c, c, f) | inactive_node(i, j-1, k, ibg, c, c, f), zero(ibg), ∂yᶜᶠᶠ_η(i, j, k, ibg.underlying_grid, args...))

@inline δxᶜᵃᵃ_U(i, j, k, ibg::IBG, T, U★::Function, args...) = δxᶜᵃᵃ_U(i, j, k, ibg.underlying_grid, T, conditional_U_fcc,  ibg, U★, args...)
@inline δyᵃᶜᵃ_V(i, j, k, ibg::IBG, T, V★::Function, args...) = δyᵃᶜᵃ_V(i, j, k, ibg.underlying_grid, T, conditional_V_cfc,  ibg, V★, args...)
@inline ∂xᶠᶜᶠ_η(i, j, k, ibg::IBG, T, η★::Function, args...) = conditional_∂xᶠᶜᶠ_η(i, j, k, ibg, T, η★, args...)
@inline ∂yᶜᶠᶠ_η(i, j, k, ibg::IBG, T, η★::Function, args...) = conditional_∂yᶜᶠᶠ_η(i, j, k, ibg, T, η★, args...)

# Disambiguation
for Topo in [:Periodic, :Bounded, :RightConnected, :LeftConnected]
    @eval begin
        @inline δxᶜᵃᵃ_U(i, j, k, ibg::IBG, T::Type{$Topo}, U★::Function, args...) = δxᶜᵃᵃ_U(i, j, k, ibg.underlying_grid, T, conditional_U_fcc, ibg, U★, args...)
        @inline δyᵃᶜᵃ_V(i, j, k, ibg::IBG, T::Type{$Topo}, V★::Function, args...) = δyᵃᶜᵃ_V(i, j, k, ibg.underlying_grid, T, conditional_V_cfc, ibg, V★, args...)
    end
end

# Time stepping extrapolation U★, and η★

# AB3 step
@inline function U★(i, j, k, grid, ::AdamsBashforth3Scheme, Uᵐ, Uᵐ⁻¹, Uᵐ⁻²)
    FT = eltype(grid)
    return @inbounds FT(α) * Uᵐ[i, j, k] + FT(θ) * Uᵐ⁻¹[i, j, k] + FT(β) * Uᵐ⁻²[i, j, k]
end

@inline function η★(i, j, k, grid, ::AdamsBashforth3Scheme, ηᵐ⁺¹, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)
    FT = eltype(grid)
    return @inbounds FT(δ) * ηᵐ⁺¹[i, j, k] + FT(μ) * ηᵐ[i, j, k] + FT(γ) * ηᵐ⁻¹[i, j, k] + FT(ϵ) * ηᵐ⁻²[i, j, k]
end

# Forward Backward Step
@inline U★(i, j, k, grid, ::ForwardBackwardScheme, U, args...) = @inbounds U[i, j, k]
@inline η★(i, j, k, grid, ::ForwardBackwardScheme, η, args...) = @inbounds η[i, j, k]

@inline advance_previous_velocity!(i, j, k, ::ForwardBackwardScheme, U, Uᵐ⁻¹, Uᵐ⁻²) = nothing

@inline function advance_previous_velocity!(i, j, k, ::AdamsBashforth3Scheme, U, Uᵐ⁻¹, Uᵐ⁻²)
    @inbounds Uᵐ⁻²[i, j, k] = Uᵐ⁻¹[i, j, k]
    @inbounds Uᵐ⁻¹[i, j, k] =    U[i, j, k]

    return nothing
end

@inline advance_previous_free_surface!(i, j, k, ::ForwardBackwardScheme, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) = nothing

@inline function advance_previous_free_surface!(i, j, k, ::AdamsBashforth3Scheme, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)
    @inbounds ηᵐ⁻²[i, j, k] = ηᵐ⁻¹[i, j, k]
    @inbounds ηᵐ⁻¹[i, j, k] =   ηᵐ[i, j, k]
    @inbounds   ηᵐ[i, j, k] =    η[i, j, k]

    return nothing
end

@kernel function _split_explicit_free_surface!(grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², timestepper)
    i, j = @index(Global, NTuple)
    free_surface_evolution!(i, j, grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², timestepper)
end

@inline function free_surface_evolution!(i, j, grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², timestepper)
    k_top = grid.Nz+1
    TX, TY, _ = topology(grid)

    @inbounds begin        
        advance_previous_free_surface!(i, j, k_top, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)

        η[i, j, k_top] -= Δτ * (div_xᶜᶜᶠ_U(i, j, k_top-1, grid, TX, U★, timestepper, U, Uᵐ⁻¹, Uᵐ⁻²) +
                                div_yᶜᶜᶠ_V(i, j, k_top-1, grid, TY, U★, timestepper, V, Vᵐ⁻¹, Vᵐ⁻²))
    end

    return nothing
end

@kernel function _split_explicit_barotropic_velocity!(averaging_weight, grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², 
                                                      U, Uᵐ⁻¹, Uᵐ⁻², V,  Vᵐ⁻¹, Vᵐ⁻²,
                                                      η̅, U̅, V̅, Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, g, 
                                                      timestepper)
    i, j = @index(Global, NTuple)
    velocity_evolution!(i, j, grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², 
                        U, Uᵐ⁻¹, Uᵐ⁻², V,  Vᵐ⁻¹, Vᵐ⁻²,
                        η̅, U̅, V̅, averaging_weight,
                        Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, g, 
                        timestepper)
end

@inline function velocity_evolution!(i, j, grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², 
                                     U, Uᵐ⁻¹, Uᵐ⁻², V,  Vᵐ⁻¹, Vᵐ⁻²,
                                     η̅, U̅, V̅, averaging_weight,
                                     Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, g, 
                                     timestepper)
    k_top = grid.Nz+1
    
    TX, TY, _ = topology(grid)

    @inbounds begin 
        advance_previous_velocity!(i, j, k_top-1, timestepper, U, Uᵐ⁻¹, Uᵐ⁻²)
        advance_previous_velocity!(i, j, k_top-1, timestepper, V, Vᵐ⁻¹, Vᵐ⁻²)

        # ∂τ(U) = - ∇η + G
        U[i, j, k_top-1] +=  Δτ * (- g * Hᶠᶜ[i, j, 1] * ∂xᶠᶜᶠ_η(i, j, k_top, grid, TX, η★, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) + Gᵁ[i, j, k_top-1])
        V[i, j, k_top-1] +=  Δτ * (- g * Hᶜᶠ[i, j, 1] * ∂yᶜᶠᶠ_η(i, j, k_top, grid, TY, η★, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) + Gⱽ[i, j, k_top-1])
                          
        # time-averaging
        η̅[i, j, k_top]   += averaging_weight * η[i, j, k_top]
        U̅[i, j, k_top-1] += averaging_weight * U[i, j, k_top-1]
        V̅[i, j, k_top-1] += averaging_weight * V[i, j, k_top-1]
    end
end

# Barotropic Model Kernels
# u_Δz = u * Δz
@kernel function _barotropic_mode_kernel!(U, V, grid, u, v)
    i, j  = @index(Global, NTuple)	
    k_top = grid.Nz+1

    # hand unroll first loop
    @inbounds U[i, j, k_top-1] = Δzᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1]
    @inbounds V[i, j, k_top-1] = Δzᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1]

    @unroll for k in 2:grid.Nz
        @inbounds U[i, j, k_top-1] += Δzᶠᶜᶜ(i, j, k, grid) * u[i, j, k]
        @inbounds V[i, j, k_top-1] += Δzᶜᶠᶜ(i, j, k, grid) * v[i, j, k]
    end
end

# Barotropic Model Kernels
# u_Δz = u * Δz
@kernel function _barotropic_mode_kernel!(U, V, grid::ActiveZColumnsIBGIBG, u, v)
    idx = @index(Global, Linear)
    i, j = active_linear_index_to_tuple(idx, ZColumnMap(), grid)
    k_top = grid.Nz+1

    # hand unroll first loop
    @inbounds U[i, j, k_top-1] = Δzᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1]
    @inbounds V[i, j, k_top-1] = Δzᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1]

    @unroll for k in 2:grid.Nz
        @inbounds U[i, j, k_top-1] += Δzᶠᶜᶜ(i, j, k, grid) * u[i, j, k]
        @inbounds V[i, j, k_top-1] += Δzᶜᶠᶜ(i, j, k, grid) * v[i, j, k]
    end
end

compute_barotropic_mode!(U, V, grid, u, v) = 
    launch!(architecture(grid), grid, :xy, _barotropic_mode_kernel!, U, V, grid, u, v; 
            only_active_cells = use_only_active_surface_cells(grid))

function initialize_free_surface_state!(state, η, timestepper)

    parent(state.U) .= parent(state.U̅)
    parent(state.V) .= parent(state.V̅)

    initialize_auxiliary_state!(state, η, timestepper)

    fill!(state.η̅, 0)
    fill!(state.U̅, 0)
    fill!(state.V̅, 0)

    return nothing
end

initialize_auxiliary_state!(state, η, ::ForwardBackwardScheme) = nothing

function initialize_auxiliary_state!(state, η, timestepper)
    parent(state.Uᵐ⁻¹) .= parent(state.U̅)
    parent(state.Vᵐ⁻¹) .= parent(state.V̅)

    parent(state.Uᵐ⁻²) .= parent(state.U̅)
    parent(state.Vᵐ⁻²) .= parent(state.V̅)

    parent(state.ηᵐ)   .= parent(η)
    parent(state.ηᵐ⁻¹) .= parent(η)
    parent(state.ηᵐ⁻²) .= parent(η)

    return nothing
end

@kernel function _barotropic_split_explicit_corrector!(u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ, grid)
    i, j, k = @index(Global, NTuple)
    k_top = grid.Nz+1

    @inbounds begin
        u[i, j, k] = u[i, j, k] + (U̅[i, j, k_top-1] - U[i, j, k_top-1]) / Hᶠᶜ[i, j, 1]
        v[i, j, k] = v[i, j, k] + (V̅[i, j, k_top-1] - V[i, j, k_top-1]) / Hᶜᶠ[i, j, 1]
    end
end

# may need to do Val(Nk) since it may not be known at compile. Also figure out where to put H
function barotropic_split_explicit_corrector!(u, v, free_surface, grid)
    sefs       = free_surface.state
    U, V, U̅, V̅ = sefs.U, sefs.V, sefs.U̅, sefs.V̅
    Hᶠᶜ, Hᶜᶠ   = free_surface.auxiliary.Hᶠᶜ, free_surface.auxiliary.Hᶜᶠ
    arch       = architecture(grid)


    # take out "bad" barotropic mode, 
    # !!!! reusing U and V for this storage since last timestep doesn't matter
    compute_barotropic_mode!(U, V, grid, u, v)
    # add in "good" barotropic mode
    launch!(arch, grid, :xyz, _barotropic_split_explicit_corrector!,
            u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ, grid)

    return nothing
end

"""
Explicitly step forward η in substeps.
"""
ab2_step_free_surface!(free_surface::SplitExplicitFreeSurface, model, Δt, χ) =
    split_explicit_free_surface_step!(free_surface, model, Δt, χ)

function initialize_free_surface!(sefs::SplitExplicitFreeSurface, grid, velocities)
    @apply_regionally compute_barotropic_mode!(sefs.state.U̅, sefs.state.V̅, grid, velocities.u, velocities.v)
    fill_halo_regions!((sefs.state.U̅, sefs.state.V̅, sefs.η))
end

function split_explicit_free_surface_step!(free_surface::SplitExplicitFreeSurface, model, Δt, χ)

    # Note: free_surface.η.grid != model.grid for DistributedSplitExplicitFreeSurface
    # since halo_size(free_surface.η.grid) != halo_size(model.grid)
    free_surface_grid = free_surface.η.grid

    # Wait for previous set up
    wait_free_surface_communication!(free_surface, architecture(free_surface_grid))

    # Calculate the substepping parameterers
    settings = free_surface.settings 
    Nsubsteps = calculate_substeps(settings.substepping, Δt)
    
    # barotropic time step as fraction of baroclinic step and averaging weights
    fractional_Δt, weights = calculate_adaptive_settings(settings.substepping, Nsubsteps) 
    Nsubsteps = length(weights)

    # barotropic time step in seconds
    Δτᴮ = fractional_Δt * Δt  
    
    # reset free surface averages
    @apply_regionally begin 
        initialize_free_surface_state!(free_surface.state, free_surface.η, settings.timestepper)
        
        # Solve for the free surface at tⁿ⁺¹
        iterate_split_explicit!(free_surface, free_surface_grid, Δτᴮ, weights, Val(Nsubsteps))
        
        # Reset eta for the next timestep
        set!(free_surface.η, free_surface.state.η̅)
    end

    fields_to_fill = (free_surface.state.U̅, free_surface.state.V̅)
    fill_halo_regions!(fields_to_fill; async = true)

    # Preparing velocities for the barotropic correction
    @apply_regionally begin 
        mask_immersed_field!(model.velocities.u)
        mask_immersed_field!(model.velocities.v)
    end

    return nothing
end

# Change name
const FNS = FixedSubstepNumber
const FTS = FixedTimeStepSize

# since weights can be negative in the first few substeps (as in the default averaging kernel), 
# we set a minimum number of substeps to execute to avoid numerical issues
const MINIMUM_SUBSTEPS = 5

@inline calculate_substeps(substepping::FNS, Δt) = length(substepping.averaging_weights)
@inline calculate_substeps(substepping::FTS, Δt) = max(MINIMUM_SUBSTEPS, ceil(Int, 2 * Δt / substepping.Δt_barotropic))

@inline calculate_adaptive_settings(substepping::FNS, substeps) = substepping.fractional_step_size, substepping.averaging_weights
@inline calculate_adaptive_settings(substepping::FTS, substeps) = weights_from_substeps(eltype(substepping.Δt_barotropic),
                                                                                        substeps, substepping.averaging_kernel)

const FixedSubstepsSetting{N} = SplitExplicitSettings{<:FixedSubstepNumber{<:Any, <:NTuple{N, <:Any}}} where N
const FixedSubstepsSplitExplicit{F} = SplitExplicitFreeSurface{<:Any, <:Any, <:Any, <:Any, <:FixedSubstepsSetting{N}} where N

function iterate_split_explicit!(free_surface, grid, Δτᴮ, weights, ::Val{Nsubsteps}) where Nsubsteps
    arch = architecture(grid)

    η         = free_surface.η
    state     = free_surface.state
    auxiliary = free_surface.auxiliary
    settings  = free_surface.settings
    g         = free_surface.gravitational_acceleration
    
    # unpack state quantities, parameters and forcing terms 
    U, V             = state.U,    state.V
    Uᵐ⁻¹, Uᵐ⁻²       = state.Uᵐ⁻¹, state.Uᵐ⁻²
    Vᵐ⁻¹, Vᵐ⁻²       = state.Vᵐ⁻¹, state.Vᵐ⁻²
    ηᵐ, ηᵐ⁻¹, ηᵐ⁻²   = state.ηᵐ,   state.ηᵐ⁻¹, state.ηᵐ⁻²
    η̅, U̅, V̅          = state.η̅, state.U̅, state.V̅
    Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ = auxiliary.Gᵁ, auxiliary.Gⱽ, auxiliary.Hᶠᶜ, auxiliary.Hᶜᶠ
    
    timestepper      = settings.timestepper
    
    parameters = auxiliary.kernel_parameters
    
    free_surface_kernel! = configured_kernel(arch, grid, parameters, _split_explicit_free_surface!)
    barotropic_velocity_kernel! = configured_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!)

    η_args = (grid, Δτᴮ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², 
              U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², 
              timestepper)

    U_args = (grid, Δτᴮ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², 
              U, Uᵐ⁻¹, Uᵐ⁻², V,  Vᵐ⁻¹, Vᵐ⁻²,
              η̅, U̅, V̅, Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, g, 
              timestepper)

    GC.@preserve η_args U_args begin

        # We need to perform ~50 time-steps which means
        # launching ~100 very small kernels: we are limited by
        # latency of argument conversion to GPU-compatible values.
        # To alleviate this penalty we convert first and then we substep!
        converted_η_args = convert_args(arch, η_args)
        converted_U_args = convert_args(arch, U_args)
            
        @unroll for substep in 1:Nsubsteps
            Base.@_inline_meta

            averaging_weight = weights[substep]

            free_surface_kernel!(converted_η_args...)
            barotropic_velocity_kernel!(averaging_weight, converted_U_args...)
        end
    end

    return nothing
end

# Calculate RHS for the barotopic time step. 
@kernel function _compute_integrated_ab2_tendencies!(Gᵁ, Gⱽ, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    i, j  = @index(Global, NTuple)	
    k_top = grid.Nz+1

    # hand unroll first loop 	
    @inbounds Gᵁ[i, j, k_top-1] = Δzᶠᶜᶜ(i, j, 1, grid) * ab2_step_Gu(i, j, 1, grid, Gu⁻, Guⁿ, χ)
    @inbounds Gⱽ[i, j, k_top-1] = Δzᶜᶠᶜ(i, j, 1, grid) * ab2_step_Gv(i, j, 1, grid, Gv⁻, Gvⁿ, χ)

    @unroll for k in 2:grid.Nz	
        @inbounds Gᵁ[i, j, k_top-1] += Δzᶠᶜᶜ(i, j, k, grid) * ab2_step_Gu(i, j, k, grid, Gu⁻, Guⁿ, χ)
        @inbounds Gⱽ[i, j, k_top-1] += Δzᶜᶠᶜ(i, j, k, grid) * ab2_step_Gv(i, j, k, grid, Gv⁻, Gvⁿ, χ)
    end	
end

# Calculate RHS for the barotopic time step. 
@kernel function _compute_integrated_ab2_tendencies!(Gᵁ, Gⱽ, grid::ActiveZColumnsIBGIBG, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    idx = @index(Global, Linear)
    i, j = active_linear_index_to_tuple(idx, ZColumnMap(), grid)
    k_top = grid.Nz+1

    # hand unroll first loop 	
    @inbounds Gᵁ[i, j, k_top-1] = Δzᶠᶜᶜ(i, j, 1, grid) * ab2_step_Gu(i, j, 1, grid, Gu⁻, Guⁿ, χ)
    @inbounds Gⱽ[i, j, k_top-1] = Δzᶜᶠᶜ(i, j, 1, grid) * ab2_step_Gv(i, j, 1, grid, Gv⁻, Gvⁿ, χ)

    @unroll for k in 2:grid.Nz	
        @inbounds Gᵁ[i, j, k_top-1] += Δzᶠᶜᶜ(i, j, k, grid) * ab2_step_Gu(i, j, k, grid, Gu⁻, Guⁿ, χ)
        @inbounds Gⱽ[i, j, k_top-1] += Δzᶜᶠᶜ(i, j, k, grid) * ab2_step_Gv(i, j, k, grid, Gv⁻, Gvⁿ, χ)
    end	
end

@inline ab2_step_Gu(i, j, k, grid, G⁻, Gⁿ, χ::FT) where FT =
    @inbounds ifelse(peripheral_node(i, j, k, grid, f, c, c), zero(grid), (convert(FT, 1.5) + χ) *  Gⁿ[i, j, k] - G⁻[i, j, k] * (convert(FT, 0.5) + χ))

@inline ab2_step_Gv(i, j, k, grid, G⁻, Gⁿ, χ::FT) where FT =
    @inbounds ifelse(peripheral_node(i, j, k, grid, c, f, c), zero(grid), (convert(FT, 1.5) + χ) *  Gⁿ[i, j, k] - G⁻[i, j, k] * (convert(FT, 0.5) + χ))

# Setting up the RHS for the barotropic step (tendencies of the barotopic velocity components)
# This function is called after `calculate_tendency` and before `ab2_step_velocities!`
function setup_free_surface!(model, free_surface::SplitExplicitFreeSurface, χ)
    
    # we start the time integration of η from the average ηⁿ     
    Gu⁻ = model.timestepper.G⁻.u
    Gv⁻ = model.timestepper.G⁻.v
    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v

    auxiliary = free_surface.auxiliary

    @apply_regionally setup_split_explicit_tendency!(auxiliary, model.grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)

    fields_to_fill = (auxiliary.Gᵁ, auxiliary.Gⱽ)
    fill_halo_regions!(fields_to_fill; async = true)

    return nothing
end

setup_split_explicit_tendency!(auxiliary, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ) =
    launch!(architecture(grid), grid, :xy, _compute_integrated_ab2_tendencies!, auxiliary.Gᵁ, auxiliary.Gⱽ, grid, 
            Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ; only_active_cells = use_only_active_surface_cells(grid))
            
wait_free_surface_communication!(free_surface, arch) = nothing

