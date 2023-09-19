using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Grids: topology
using Oceananigans.Utils
using Oceananigans.AbstractOperations: Δz  
using Oceananigans.BoundaryConditions
using Oceananigans.Operators
using Oceananigans.ImmersedBoundaries: peripheral_node, immersed_inactive_node,
                                       inactive_node, IBG, c, f

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
                               
# Special ``partial'' divergence for free surface evolution
@inline div_xᶜᶜᶠ_U(i, j, k, grid, U★::Function, args...) =  1 / Azᶜᶜᶠ(i, j, k, grid) * δxᶜᵃᵃ_U(i, j, k, grid, Δy_qᶠᶜᶠ, U★, args...) 
@inline div_yᶜᶜᶠ_V(i, j, k, grid, V★::Function, args...) =  1 / Azᶜᶜᶠ(i, j, k, grid) * δyᵃᶜᵃ_V(i, j, k, grid, Δx_qᶜᶠᶠ, V★, args...) 

# The functions `η★` `U★` and `V★` represent the value of free surface, barotropic zonal and meridional velocity at time step m+1/2

# Time stepping extrapolation U★, and η★

# AB3 step
@inline function U★(i, j, k, grid, ::AdamsBashforth3Scheme, ϕᵐ, ϕᵐ⁻¹, ϕᵐ⁻²)
    FT = eltype(grid)
    return FT(α) * ϕᵐ[i, j, k]   + FT(θ) * ϕᵐ⁻¹[i, j, k] + FT(β) * ϕᵐ⁻²[i, j, k]
end

@inline function η★(i, j, k, grid, ::AdamsBashforth3Scheme, ηᵐ⁺¹, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)
    FT = eltype(grid)
    return FT(δ) * ηᵐ⁺¹[i, j, k] + FT(μ) * ηᵐ[i, j, k]   + FT(γ) * ηᵐ⁻¹[i, j, k] + FT(ϵ) * ηᵐ⁻²[i, j, k]
end

# Forward Backward Step
@inline U★(i, j, k, grid, ::ForwardBackwardScheme, ϕ, args...) = ϕ[i, j, k] 
@inline η★(i, j, k, grid, ::ForwardBackwardScheme, η, args...) = η[i, j, k] 

@inline advance_previous_velocity!(i, j, k, ::ForwardBackwardScheme, U, Uᵐ⁻¹, Uᵐ⁻²) = nothing

@inline function advance_previous_velocity!(i, j, k, ::AdamsBashforth3Scheme, U, Uᵐ⁻¹, Uᵐ⁻²)
    @inbounds Uᵐ⁻²[i, j, k] = Uᵐ⁻¹[i, j, k] 
    @inbounds Uᵐ⁻¹[i, j, k] =    U[i, j, k] 
end

@inline advance_previous_free_surface!(i, j, k, ::ForwardBackwardScheme, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) = nothing

@inline function advance_previous_free_surface!(i, j, k, ::AdamsBashforth3Scheme, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)
    @inbounds ηᵐ⁻²[i, j, k] = ηᵐ⁻¹[i, j, k]
    @inbounds ηᵐ⁻¹[i, j, k] =   ηᵐ[i, j, k]
    @inbounds   ηᵐ[i, j, k] =    η[i, j, k]
end

@kernel function split_explicit_free_surface_evolution_kernel!(grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², 
                                                               η̅, U̅, V̅, averaging_weight, 
                                                               Gᵁ, Gⱽ, g, Hᶠᶜ, Hᶜᶠ,
                                                               timestepper, offsets)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    i′ = i - offsets[1]
    j′ = j - offsets[2]

    @inbounds begin        
        advance_previous_free_surface!(i′, j′, k_top, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)

        # ∂τ(η) = - ∇ ⋅ U. 
        # `k_top - 1` is used here to allow `immersed_peripheral_node` to be true 
        # NOTE: `immersed_peripheral_node` is _always_ false on `Nz+1` `Face`s because `peripheral_node` is always true
        η[i′, j′, k_top] -= Δτ * (div_xᶜᶜᶠ_U(i′, j′, k_top-1, grid, U★, timestepper, U, Uᵐ⁻¹, Uᵐ⁻²) +
                                  div_yᶜᶜᶠ_V(i′, j′, k_top-1, grid, U★, timestepper, V, Vᵐ⁻¹, Vᵐ⁻²))
    end
end

@kernel function split_explicit_barotropic_velocity_evolution_kernel!(grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², 
                                                                      η̅, U̅, V̅, averaging_weight, 
                                                                      Gᵁ, Gⱽ, g, Hᶠᶜ, Hᶜᶠ,
                                                                      timestepper, offsets)
    i, j  = @index(Global, NTuple)
    k_top = grid.Nz+1
    
    i′ = i - offsets[1]
    j′ = j - offsets[2]

    @inbounds begin 
        advance_previous_velocity!(i′, j′, 1, timestepper, U, Uᵐ⁻¹, Uᵐ⁻²)
        advance_previous_velocity!(i′, j′, 1, timestepper, V, Vᵐ⁻¹, Vᵐ⁻²)

        # ∂τ(U) = - ∇η + G
        U[i′, j′, 1] +=  Δτ * (- g * Hᶠᶜ[i′, j′] * ∂xᶠᶜᶠ_c(i′, j′, k_top, grid, η★, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) + Gᵁ[i′, j′, 1])
        V[i′, j′, 1] +=  Δτ * (- g * Hᶜᶠ[i′, j′] * ∂yᶜᶠᶠ_c(i′, j′, k_top, grid, η★, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) + Gⱽ[i′, j′, 1])
                          
        # time-averaging
        η̅[i′, j′, k_top] +=  averaging_weight * η[i′, j′, k_top]
        U̅[i′, j′, 1]     +=  averaging_weight * U[i′, j′, 1]
        V̅[i′, j′, 1]     +=  averaging_weight * V[i′, j′, 1]
    end
end

function split_explicit_free_surface_substep!(η, state, auxiliary, settings, weights, arch, grid, g, Δτ, substep_index)
    # unpack state quantities, parameters and forcing terms 
    U, V             = state.U,    state.V
    Uᵐ⁻¹, Uᵐ⁻²       = state.Uᵐ⁻¹, state.Uᵐ⁻²
    Vᵐ⁻¹, Vᵐ⁻²       = state.Vᵐ⁻¹, state.Vᵐ⁻²
    ηᵐ, ηᵐ⁻¹, ηᵐ⁻²   = state.ηᵐ,   state.ηᵐ⁻¹, state.ηᵐ⁻²
    η̅, U̅, V̅          = state.η̅, state.U̅, state.V̅
    Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ = auxiliary.Gᵁ, auxiliary.Gⱽ, auxiliary.Hᶠᶜ, auxiliary.Hᶜᶠ
    
    timestepper      = settings.timestepper
    averaging_weight = weights[substep_index]
    
    offsets     = auxiliary.kernel_offsets
    kernel_size = auxiliary.kernel_size

    args = (grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², 
            η̅, U̅, V̅, averaging_weight, 
            Gᵁ, Gⱽ, g, Hᶠᶜ, Hᶜᶠ, timestepper, offsets)

    launch!(arch, grid, kernel_size, split_explicit_free_surface_evolution_kernel!, args...)

    launch!(arch, grid, kernel_size, split_explicit_barotropic_velocity_evolution_kernel!, args...)

    return nothing
end

# Barotropic Model Kernels
# u_Δz = u * Δz
@kernel function _barotropic_mode_kernel!(U, V, grid, u, v)
    i, j  = @index(Global, NTuple)	

    # hand unroll first loop
    @inbounds U[i, j, 1] = Δzᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1]
    @inbounds V[i, j, 1] = Δzᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1]

    @unroll for k in 2:grid.Nz
        @inbounds U[i, j, 1] += Δzᶠᶜᶜ(i, j, k, grid) * u[i, j, k]
        @inbounds V[i, j, 1] += Δzᶜᶠᶜ(i, j, k, grid) * v[i, j, k]
    end
end

# may need to do Val(Nk) since it may not be known at compile
compute_barotropic_mode!(U, V, grid, u, v) = 
    launch!(architecture(grid), grid, :xy, _barotropic_mode_kernel!, U, V, grid, u, v)

function initialize_free_surface_state!(free_surface_state, η)
    state = free_surface_state

    parent(state.U) .= parent(state.U̅)
    parent(state.V) .= parent(state.V̅)
    
    parent(state.Uᵐ⁻¹) .= parent(state.U̅)
    parent(state.Vᵐ⁻¹) .= parent(state.V̅)

    parent(state.Uᵐ⁻²) .= parent(state.U̅)
    parent(state.Vᵐ⁻²) .= parent(state.V̅)

    parent(state.ηᵐ)   .= parent(η)
    parent(state.ηᵐ⁻¹) .= parent(η)
    parent(state.ηᵐ⁻²) .= parent(η)

    fill!(state.η̅, 0.0)
    fill!(state.U̅, 0.0)
    fill!(state.V̅, 0.0)
end

@kernel function barotropic_split_explicit_corrector_kernel!(u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        u[i, j, k] = u[i, j, k] + (U̅[i, j] - U[i, j]) / Hᶠᶜ[i, j] 
        v[i, j, k] = v[i, j, k] + (V̅[i, j] - V[i, j]) / Hᶜᶠ[i, j]
    end
end

# may need to do Val(Nk) since it may not be known at compile. Also figure out where to put H
function barotropic_split_explicit_corrector!(u, v, free_surface, grid)
    sefs       = free_surface.state
    Hᶠᶜ, Hᶜᶠ   = free_surface.auxiliary.Hᶠᶜ, free_surface.auxiliary.Hᶜᶠ
    U, V, U̅, V̅ = sefs.U, sefs.V, sefs.U̅, sefs.V̅
    arch       = architecture(grid)

    # take out "bad" barotropic mode, 
    # !!!! reusing U and V for this storage since last timestep doesn't matter
    compute_barotropic_mode!(U, V, grid, u, v)
    # add in "good" barotropic mode

    launch!(arch, grid, :xyz, barotropic_split_explicit_corrector_kernel!,
        u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ)
end

@kernel function _calc_ab2_tendencies!(G⁻, Gⁿ, χ::FT) where FT
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻[i, j, k] = (FT(1.5) + χ) *  Gⁿ[i, j, k] - G⁻[i, j, k] * (FT(0.5) + χ)
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

    grid = free_surface.η.grid

    # we start the time integration of η from the average ηⁿ     
    Gu  = model.timestepper.G⁻.u
    Gv  = model.timestepper.G⁻.v
    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v

    fill_halo_regions!((free_surface.state.U̅, free_surface.state.V̅))
    
    @apply_regionally setup_split_explicit!(free_surface.auxiliary, free_surface.state, free_surface.η, grid, Gu, Gv, Guⁿ, Gvⁿ, χ)

    fill_halo_regions!((free_surface.auxiliary.Gᵁ, free_surface.auxiliary.Gⱽ))
    
    @apply_regionally begin 
        # Solve for the free surface at tⁿ⁺¹
        iterate_split_explicit!(free_surface, grid, Δt)
        # this is the only way in which η̅ is used: as a smoother for the substepped η field
        set!(free_surface.η, free_surface.state.η̅)

        # Wait for predictor velocity update step to complete and mask it if immersed boundary.
        mask_immersed_field!(model.velocities.u)
        mask_immersed_field!(model.velocities.v)
    end

    fill_halo_regions!(free_surface.η)

    return nothing
end

# Change name
const FNS = FixedSubstepNumber
const FTS = FixedTimeStepSize

@inline calculate_substeps(substepping::FNS, Δt) = length(substepping.averaging_weights)
@inline calculate_substeps(substepping::FTS, Δt) = ceil(Int, 2 * Δt / settings.Δtᴮ)

@inline calculate_adaptive_settings(substepping::FNS, substeps) = substepping.fractional_step_size, substepping.averaging_weights
@inline calculate_adaptive_settings(substepping::FTS, substeps) = weights_from_substeps(eltype(substepping.Δt_barotopic), 
                                                                                     substeps, substepping.averaging_kernel)

function iterate_split_explicit!(free_surface, grid, Δt)
    arch = architecture(grid)

    η         = free_surface.η
    state     = free_surface.state
    auxiliary = free_surface.auxiliary
    settings  = free_surface.settings
    g         = free_surface.gravitational_acceleration

    Nsubsteps  = calculate_substeps(settings.substepping, Δt)
    fractional_Δt, weights = calculate_adaptive_settings(settings.substepping, Nsubsteps) # barotropic time step in fraction of baroclinic step and averaging weights
    
    Nsubsteps = length(weights)

    Δτᴮ = fractional_Δt * Δt  # barotropic time step in seconds

    for substep in 1:Nsubsteps
        split_explicit_free_surface_substep!(η, state, auxiliary, settings, weights, arch, grid, g, Δτᴮ, substep)
    end
end

function setup_split_explicit!(auxiliary, state, η, grid, Gu, Gv, Guⁿ, Gvⁿ, χ)
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _calc_ab2_tendencies!, Gu, Guⁿ, χ)
    launch!(arch, grid, :xyz, _calc_ab2_tendencies!, Gv, Gvⁿ, χ)

    # reset free surface averages
    initialize_free_surface_state!(state, η)

    # Wait for predictor velocity update step to complete and mask it if immersed boundary.
    mask_immersed_field!(Gu)
    mask_immersed_field!(Gv)

    # Compute barotropic mode of tendency fields
    compute_barotropic_mode!(auxiliary.Gᵁ, auxiliary.Gⱽ, grid, Gu, Gv)

    return nothing
end

