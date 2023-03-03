using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Grids: topology
using Oceananigans.Utils
using Oceananigans.AbstractOperations: Δz  
using Oceananigans.BoundaryConditions
using Oceananigans.Operators
using Oceananigans.ImmersedBoundaries: peripheral_node, immersed_inactive_node,
                                       inactive_node, IBG, c, f

const β = 0.281105
const α = 1.5 + β
const θ = - 0.5 - 2β

const γ = 0.088
const δ = 0.614
const ϵ = 0.013
const μ = 1.0 - δ - γ - ϵ

# Evolution Kernels
#=
∂t(η) = -∇⋅U
∂t(U) = - gH∇η + f
=#

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

# functions `η★` `U★` and `V★` represent the value of free surface, barotropic zonal and meridional velocity at time step m+1/2
@inline δxᶠᵃᵃ_η(i, j, k, grid, T, η★::Function, args...) = δxᶠᵃᵃ(i, j, k, grid, η★, args...)
@inline δyᵃᶠᵃ_η(i, j, k, grid, T, η★::Function, args...) = δyᵃᶠᵃ(i, j, k, grid, η★, args...)
@inline δxᶜᵃᵃ_U(i, j, k, grid, T, U★::Function, args...) = δxᶜᵃᵃ(i, j, k, grid, U★, args...)
@inline δyᵃᶜᵃ_V(i, j, k, grid, T, V★::Function, args...) = δyᵃᶜᵃ(i, j, k, grid, V★, args...)

@inline δxᶠᵃᵃ_η(i, j, k, grid, ::Type{Periodic}, η★::Function, args...) = ifelse(i == 1, η★(1, j, k, grid, args...) - η★(grid.Nx, j, k, grid, args...), δxᶠᵃᵃ(i, j, k, grid, η★, args...))
@inline δyᵃᶠᵃ_η(i, j, k, grid, ::Type{Periodic}, η★::Function, args...) = ifelse(j == 1, η★(i, 1, k, grid, args...) - η★(i, grid.Ny, k, grid, args...), δyᵃᶠᵃ(i, j, k, grid, η★, args...))

@inline δxᶜᵃᵃ_U(i, j, k, grid, ::Type{Periodic}, U★::Function, args...) = ifelse(i == grid.Nx, U★(1, j, k, grid, args...) - U★(grid.Nx, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, U★, args...))
@inline δyᵃᶜᵃ_V(i, j, k, grid, ::Type{Periodic}, V★::Function, args...) = ifelse(j == grid.Ny, V★(i, 1, k, grid, args...) - V★(i, grid.Ny, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, V★, args...))

# Enforce NoFlux conditions for `η★`

@inline δxᶠᵃᵃ_η(i, j, k, grid, ::Type{Bounded},        η★::Function, args...) = ifelse(i == 1, 0.0, δxᶠᵃᵃ(i, j, k, grid, η★, args...))
@inline δyᵃᶠᵃ_η(i, j, k, grid, ::Type{Bounded},        η★::Function, args...) = ifelse(j == 1, 0.0, δyᵃᶠᵃ(i, j, k, grid, η★, args...))
@inline δxᶠᵃᵃ_η(i, j, k, grid, ::Type{RightConnected}, η★::Function, args...) = ifelse(i == 1, 0.0, δxᶠᵃᵃ(i, j, k, grid, η★, args...))
@inline δyᵃᶠᵃ_η(i, j, k, grid, ::Type{RightConnected}, η★::Function, args...) = ifelse(j == 1, 0.0, δyᵃᶠᵃ(i, j, k, grid, η★, args...))

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
                                                                                                                                           
@inline div_xᶜᶜᶠ_U(i, j, k, grid, TX, U★, args...) = 
    1 / Azᶜᶜᶠ(i, j, k, grid) * δxᶜᵃᵃ_U(i, j, k, grid, TX, Δy_qᶠᶜᶠ, U★, args...) 

@inline div_yᶜᶜᶠ_V(i, j, k, grid, TY, V★, args...) = 
    1 / Azᶜᶜᶠ(i, j, k, grid) * δyᵃᶜᵃ_V(i, j, k, grid, TY, Δx_qᶜᶠᶠ, V★, args...) 

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
@inline U★(i, j, k, grid, ::AdamsBashforth3Scheme, ϕᵐ, ϕᵐ⁻¹, ϕᵐ⁻²)       = α * ϕᵐ[i, j, k]   + θ * ϕᵐ⁻¹[i, j, k] + β * ϕᵐ⁻²[i, j, k]
@inline η★(i, j, k, grid, ::AdamsBashforth3Scheme, ηᵐ⁺¹, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) = δ * ηᵐ⁺¹[i, j, k] + μ * ηᵐ[i, j, k]   + γ * ηᵐ⁻¹[i, j, k] + ϵ * ηᵐ⁻²[i, j, k]

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

    TX, TY, _ = topology(grid)

    @inbounds begin        
        advance_previous_free_surface!(i′, j′, k_top, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)

        # ∂τ(η) = - ∇ ⋅ U
        η[i′, j′, k_top] -= Δτ * (div_xᶜᶜᶠ_U(i′, j′, k_top-1, grid, TX, U★, timestepper, U, Uᵐ⁻¹, Uᵐ⁻²) +
                                  div_yᶜᶜᶠ_V(i′, j′, k_top-1, grid, TY, U★, timestepper, V, Vᵐ⁻¹, Vᵐ⁻²))
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

    TX, TY, _ = topology(grid)

    @inbounds begin 
        advance_previous_velocity!(i′, j′, 1, timestepper, U, Uᵐ⁻¹, Uᵐ⁻²)
        advance_previous_velocity!(i′, j′, 1, timestepper, V, Vᵐ⁻¹, Vᵐ⁻²)

        # ∂τ(U) = - ∇η + G
        U[i′, j′, 1] +=  Δτ * (- g * Hᶠᶜ[i′, j′] * ∂xᶠᶜᶠ_η(i′, j′, k_top, grid, TX, η★, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) + Gᵁ[i′, j′, 1])
        V[i′, j′, 1] +=  Δτ * (- g * Hᶜᶠ[i′, j′] * ∂yᶜᶠᶠ_η(i′, j′, k_top, grid, TY, η★, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) + Gⱽ[i′, j′, 1])
                          
        # time-averaging
        η̅[i′, j′, k_top] +=  averaging_weight * η[i′, j′, k_top]
        U̅[i′, j′, 1]     +=  averaging_weight * U[i′, j′, 1]
        V̅[i′, j′, 1]     +=  averaging_weight * V[i′, j′, 1]
    end
end

function split_explicit_free_surface_substep!(η, state, auxiliary, settings, arch, grid, g, Δτ, substep_index)
    # unpack state quantities, parameters and forcing terms 
    U, V             = state.U,    state.V
    Uᵐ⁻¹, Uᵐ⁻²       = state.Uᵐ⁻¹, state.Uᵐ⁻²
    Vᵐ⁻¹, Vᵐ⁻²       = state.Vᵐ⁻¹, state.Vᵐ⁻²
    ηᵐ, ηᵐ⁻¹, ηᵐ⁻²   = state.ηᵐ,   state.ηᵐ⁻¹, state.ηᵐ⁻²
    η̅, U̅, V̅          = state.η̅, state.U̅, state.V̅
    Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ = auxiliary.Gᵁ, auxiliary.Gⱽ, auxiliary.Hᶠᶜ, auxiliary.Hᶜᶠ
    
    timestepper      = settings.timestepper
    averaging_weight = settings.averaging_weights[substep_index]
    
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
@kernel function barotropic_mode_kernel!(U, V, grid, u, v)
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
barotropic_mode!(U, V, grid, u, v) = 
    launch!(architecture(grid), grid, :xy, barotropic_mode_kernel!, U, V, grid, u, v)

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
    barotropic_mode!(U, V, grid, u, v)
    # add in "good" barotropic mode

    launch!(arch, grid, :xyz, barotropic_split_explicit_corrector_kernel!,
        u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ)

    return nothing
end

@kernel function _calc_ab2_tendencies!(G⁻, Gⁿ, χ)
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻[i, j, k] = (1.5 + χ) *  Gⁿ[i, j, k] - G⁻[i, j, k] * (0.5 + χ)
end

"""
Explicitly step forward η in substeps.
"""
ab2_step_free_surface!(free_surface::SplitExplicitFreeSurface, model, Δt, χ) =
    split_explicit_free_surface_step!(free_surface, model, Δt, χ)
    
function initialize_free_surface!(sefs::SplitExplicitFreeSurface, grid, velocities)
    @apply_regionally barotropic_mode!(sefs.state.U̅, sefs.state.V̅, grid, velocities.u, velocities.v)
    fill_halo_regions!((sefs.state.U̅, sefs.state.V̅))
end

function split_explicit_free_surface_step!(free_surface::SplitExplicitFreeSurface, model, Δt, χ)

    grid = free_surface.η.grid

    # Wait for previous set up
    wait_free_surface_communication!(free_surface)

    # reset free surface averages
    @apply_regionally initialize_free_surface_state!(free_surface.state, free_surface.η)

    # Solve for the free surface at tⁿ⁺¹
    @apply_regionally iterate_split_explicit!(free_surface, grid, Δt)
    
    # Reset eta for the next timestep
    # this is the only way in which η̅ is used: as a smoother for the 
    # substepped η field
    @apply_regionally set!(free_surface.η, free_surface.state.η̅)

    fields_to_fill = (free_surface.state.U̅, free_surface.state.V̅)
    fill_halo_regions!(fields_to_fill; blocking = true)

    return nothing
end

function iterate_split_explicit!(free_surface, grid, Δt)
    arch = architecture(grid)

    η         = free_surface.η
    state     = free_surface.state
    auxiliary = free_surface.auxiliary
    settings  = free_surface.settings
    g         = free_surface.gravitational_acceleration

    Δτ = settings.Δτ * Δt  # we evolve for two times the Δt 

    for substep in 1:settings.substeps
        split_explicit_free_surface_substep!(η, state, auxiliary, settings, arch, grid, g, Δτ, substep)
    end

    return nothing
end

# Setting up the tendencies and the communicating the barotopic velocity components
# This function is called after `calculate_tendency` and before `ab2_step!`
function setup_free_surface!(model, free_surface::SplitExplicitFreeSurface, χ)

    grid = free_surface.η.grid
    
    # we start the time integration of η from the average ηⁿ     
    Gu  = model.timestepper.G⁻.u
    Gv  = model.timestepper.G⁻.v
    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v
    
    auxiliary = free_surface.auxiliary

    @apply_regionally setup_split_explicit_tendency!(auxiliary, grid, Gu, Gv, Guⁿ, Gvⁿ, χ)

    fields_to_fill = (auxiliary.Gᵁ, auxiliary.Gⱽ)
    fill_halo_regions!(fields_to_fill; blocking = true)

    return nothing
end

function setup_split_explicit_tendency!(auxiliary, grid, Gu, Gv, Guⁿ, Gvⁿ, χ)
    arch = architecture(grid)

    launch!(arch, grid, :xyz, _calc_ab2_tendencies!, Gu, Guⁿ, χ)
    launch!(arch, grid, :xyz, _calc_ab2_tendencies!, Gv, Gvⁿ, χ)
    
    mask_immersed_field!(Gu)
    mask_immersed_field!(Gv)

    barotropic_mode!(auxiliary.Gᵁ, auxiliary.Gⱽ, grid, Gu, Gv)

    return nothing
end

wait_free_surface_communication!(free_surface) = nothing