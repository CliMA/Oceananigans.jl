using KernelAbstractions: @index, @kernel, Event
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Grids: topology
using Oceananigans.Utils
using Oceananigans.AbstractOperations: Δz  
using Oceananigans.BoundaryConditions
using Oceananigans.Operators

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

"""
These operators hardcode the boundary conditions in the difference, 
for `Periodic` domains a periodic boundary condition is assumed, 
for `Bounded` domains, a `FluxBoundaryCondition(0.0)` is assumed for
the free surface and a `NoPenetration` boundary condition for velocity
"""
@inline ∂xᶠᶜᶠ_bound(i, j, k, grid, T, f::Function, args...) = δxᶠᵃᵃ_bound(i, j, k, grid, T, f, args...) / Δxᶠᶜᶠ(i, j, k, grid)
@inline ∂yᶜᶠᶠ_bound(i, j, k, grid, T, f::Function, args...) = δyᵃᶠᵃ_bound(i, j, k, grid, T, f, args...) / Δyᶜᶠᶠ(i, j, k, grid)

@inline δxᶠᵃᵃ_bound(i, j, k, grid, ::Type{Periodic}, f::Function, args...) = ifelse(i == 1, f(1, j, k, grid, args...) - f(grid.Nx, j, k, grid, args...), δxᶠᵃᵃ(i, j, k, grid, f, args...))
@inline δyᵃᶠᵃ_bound(i, j, k, grid, ::Type{Periodic}, f::Function, args...) = ifelse(j == 1, f(i, 1, k, grid, args...) - f(i, grid.Ny, k, grid, args...), δyᵃᶠᵃ(i, j, k, grid, f, args...))

@inline δxᶠᵃᵃ_bound(i, j, k, grid, ::Type{Bounded},  f::Function, args...) = ifelse(i == 1, 0.0, δxᶠᵃᵃ(i, j, k, grid, f, args...))
@inline δyᵃᶠᵃ_bound(i, j, k, grid, ::Type{Bounded},  f::Function, args...) = ifelse(j == 1, 0.0, δyᵃᶠᵃ(i, j, k, grid, f, args...))

@inline δxᶜᵃᵃ_bound(i, j, k, grid, ::Type{Periodic}, f::Function, args...) = ifelse(i == grid.Nx, f(1, j, k, grid, args...) - f(grid.Nx, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, f, args...))
@inline δyᵃᶜᵃ_bound(i, j, k, grid, ::Type{Periodic}, f::Function, args...) = ifelse(j == grid.Ny, f(i, 1, k, grid, args...) - f(i, grid.Ny, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, f, args...))

# Enforce Impenetrability conditions
@inline δxᶜᵃᵃ_bound(i, j, k, grid, ::Type{Bounded},  f::Function, args...) = ifelse(i == grid.Nx, - f(i, j, k, grid, args...),
                                                                             ifelse(i == 1, f(2, j, k, grid, args...), δxᶜᵃᵃ(i, j, k, grid, f, args...)))
@inline δyᵃᶜᵃ_bound(i, j, k, grid, ::Type{Bounded},  f::Function, args...) = ifelse(j == grid.Ny, - f(i, j, k, grid, args...), 
                                                                             ifelse(j == 1, f(i, 2, k, grid, args...), δyᵃᶜᵃ(i, j, k, grid, f, args...)))

@inline div_xᶜᶜᶠ_bound(i, j, k, grid, TX, f, args...) = 
    1 / Azᶜᶜᶠ(i, j, k, grid) * δxᶜᵃᵃ_bound(i, j, k, grid, TX, Δy_qᶠᶜᶠ, f, args...) 

@inline div_yᶜᶜᶠ_bound(i, j, k, grid, TY, f, args...) = 
    1 / Azᶜᶜᶠ(i, j, k, grid) * δyᵃᶜᵃ_bound(i, j, k, grid, TY, Δx_qᶜᶠᶠ, f, args...) 

using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, inactive_node, IBG, c, f

@inline immersed_inactive_node(i, j, k, ibg::IBG, LX, LY, LZ) =  inactive_node(i, j, k, ibg, LX, LY, LZ) &
                                                                !inactive_node(i, j, k, ibg.underlying_grid, LX, LY, LZ)

@inline conditional_value_fcf(i, j, k, grid, ibg, f::Function, args...) = ifelse(immersed_inactive_node(i, j, k, ibg, f, c, f), zero(ibg), f(i, j, k, grid, args...))
@inline conditional_value_cff(i, j, k, grid, ibg, f::Function, args...) = ifelse(immersed_inactive_node(i, j, k, ibg, c, f, f), zero(ibg), f(i, j, k, grid, args...))

@inline conditional_∂x_bound_f(LY, LZ, i, j, k, ibg::IBG{FT}, ∂x, args...) where FT = ifelse(immersed_inactive_node(i, j, k, ibg, c, LY, LZ) | immersed_inactive_node(i+1, j, k, ibg, c, LY, LZ), zero(ibg), ∂x(i, j, k, ibg.underlying_grid, args...))
@inline conditional_∂y_bound_f(LY, LZ, i, j, k, ibg::IBG{FT}, ∂y, args...) where FT = ifelse(immersed_inactive_node(i, j, k, ibg, f, LY, LZ) | immersed_inactive_node(i, j+1, k, ibg, f, LY, LZ), zero(ibg), ∂y(i, j, k, ibg.underlying_grid, args...))

for Topo in [:Periodic, :Bounded]
    @eval begin
        @inline δxᶜᵃᵃ_bound(i, j, k, ibg::IBG, T::Type{$Topo}, f::Function, args...) = δxᶜᵃᵃ_bound(i, j, k, ibg.underlying_grid, T, conditional_value_fcf, ibg, f, args...)
        @inline δyᵃᶜᵃ_bound(i, j, k, ibg::IBG, T::Type{$Topo}, f::Function, args...) = δyᵃᶜᵃ_bound(i, j, k, ibg.underlying_grid, T, conditional_value_cff, ibg, f, args...)

        @inline ∂xᶠᶜᶠ_bound(i, j, k, ibg::IBG, T::Type{$Topo}, f::Function, args...) = conditional_∂x_bound_f(c, f, i, j, k, ibg, ∂xᶠᶜᶠ_bound, T, f, args...)
        @inline ∂yᶜᶠᶠ_bound(i, j, k, ibg::IBG, T::Type{$Topo}, f::Function, args...) = conditional_∂y_bound_f(c, f, i, j, k, ibg, ∂yᶜᶠᶠ_bound, T, f, args...)        
    end
end

@inline U★(i, j, k, grid, ϕᵐ, ϕᵐ⁻¹, ϕᵐ⁻²)       =                     α * ϕᵐ[i, j, k] + θ * ϕᵐ⁻¹[i, j, k] + β * ϕᵐ⁻²[i, j, k]
@inline η★(i, j, k, grid, ηᵐ⁺¹, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) = δ * ηᵐ⁺¹[i, j, k] + μ * ηᵐ[i, j, k] + γ * ηᵐ⁻¹[i, j, k] + ϵ * ηᵐ⁻²[i, j, k]

@kernel function split_explicit_free_surface_substep_kernel_1!(grid, Δτ, η, U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², Ũ, Ṽ, mass_flux_weight)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    TX, TY, _ = topology(grid)

    @inbounds begin        
        η[i, j, k_top] -= Δτ * (div_xᶜᶜᶠ_bound(i, j, k_top, grid, TX, U★, U, Uᵐ⁻¹, Uᵐ⁻²) +
                                div_yᶜᶜᶠ_bound(i, j, k_top, grid, TY, U★, V, Vᵐ⁻¹, Vᵐ⁻²))
        
        Ũ[i, j, 1] +=  mass_flux_weight * U★(i, j, 1, grid, U, Uᵐ⁻¹, Uᵐ⁻²)
        Ṽ[i, j, 1] +=  mass_flux_weight * U★(i, j, 1, grid, V, Vᵐ⁻¹, Vᵐ⁻²)
    end
end

@kernel function split_explicit_free_surface_substep_kernel_2!(grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, Uᵐ⁻¹, Uᵐ⁻², V, Vᵐ⁻¹, Vᵐ⁻², 
                                                                         η̅, U̅, V̅, averaging_weight, Gᵁ, Gⱽ, g, Hᶠᶜ, Hᶜᶠ)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1
    
    TX, TY, _ = topology(grid)

    @inbounds begin 
        # ∂τ(U) = - ∇η + G
        Uᵐ⁻²[i, j, 1] = Uᵐ⁻¹[i, j, 1] 
        Uᵐ⁻¹[i, j, 1] =    U[i, j, 1] 
        Vᵐ⁻²[i, j, 1] = Vᵐ⁻¹[i, j, 1] 
        Vᵐ⁻¹[i, j, 1] =    V[i, j, 1] 

        U[i, j, 1] +=  Δτ * (- g * Hᶠᶜ[i, j] * ∂xᶠᶜᶠ_bound(i, j, k_top, grid, TX, η★, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) + Gᵁ[i, j, 1])
        V[i, j, 1] +=  Δτ * (- g * Hᶜᶠ[i, j] * ∂yᶜᶠᶠ_bound(i, j, k_top, grid, TY, η★, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) + Gⱽ[i, j, 1])

        ηᵐ⁻²[i, j, k_top] = ηᵐ⁻¹[i, j, k_top]
        ηᵐ⁻¹[i, j, k_top] =   ηᵐ[i, j, k_top]
        ηᵐ[i, j, k_top]   =    η[i, j, k_top]
        
        # ∂τ(η) = - ∇⋅U
        # time-averaging
        η̅[i, j, k_top] +=  averaging_weight * η[i, j, k_top]
        U̅[i, j, 1]     +=  averaging_weight * U[i, j, 1]
        V̅[i, j, 1]     +=  averaging_weight * V[i, j, 1]
    end
end

function split_explicit_free_surface_substep!(η, state, auxiliary, settings, arch, grid, g, Δτ, substep_index)
    # unpack state quantities, parameters and forcing terms 
    U, V             = state.U,    state.V
    Uᵐ⁻¹, Uᵐ⁻²       = state.Uᵐ⁻¹, state.Uᵐ⁻²
    Vᵐ⁻¹, Vᵐ⁻²       = state.Vᵐ⁻¹, state.Vᵐ⁻²
    ηᵐ, ηᵐ⁻¹, ηᵐ⁻²   = state.ηᵐ,   state.ηᵐ⁻¹, state.ηᵐ⁻²
    η̅, U̅, V̅, Ũ, Ṽ    = state.η̅, state.U̅, state.V̅, state.Ũ, state.Ṽ
    Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ = auxiliary.Gᵁ, auxiliary.Gⱽ, auxiliary.Hᶠᶜ, auxiliary.Hᶜᶠ
    
    averaging_weight = settings.averaging_weights[substep_index]
    mass_flux_weight = settings.mass_flux_weights[substep_index]

    event = launch!(arch, grid, :xy, split_explicit_free_surface_substep_kernel_1!, 
            grid, Δτ, η, U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², Ũ, Ṽ, mass_flux_weight,
            dependencies=Event(device(arch)))

    wait(device(arch), event)

    event = launch!(arch, grid, :xy, split_explicit_free_surface_substep_kernel_2!,
            grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, Uᵐ⁻¹, Uᵐ⁻¹, V, Vᵐ⁻¹, Vᵐ⁻², 
            η̅, U̅, V̅, averaging_weight, Gᵁ, Gⱽ, g, Hᶠᶜ, Hᶜᶠ,
            dependencies=Event(device(arch)))

    wait(device(arch), event)
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
function barotropic_mode!(U, V, grid, u, v)

    arch  = architecture(grid)
    event = launch!(arch, grid, :xy, barotropic_mode_kernel!, U, V, grid, u, v,
                   dependencies=Event(device(arch)))

    wait(device(arch), event)
    
    fill_halo_regions!((U, V))
end

function initialize_free_surface_state!(free_surface_state, η)
    state = free_surface_state

    set!(state.U, state.Ũ)
    set!(state.V, state.Ṽ)
    
    set!(state.Uᵐ⁻¹, state.Ũ)
    set!(state.Vᵐ⁻¹, state.Ṽ)

    set!(state.Uᵐ⁻², state.Ũ)
    set!(state.Vᵐ⁻², state.Ṽ)

    set!(state.ηᵐ,   η)
    set!(state.ηᵐ⁻¹, η)
    set!(state.ηᵐ⁻², η)

    fill!(state.η̅, 0.0)
    fill!(state.Ũ, 0.0)
    fill!(state.Ṽ, 0.0)
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

    event = launch!(arch, grid, :xyz, barotropic_split_explicit_corrector_kernel!,
        u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ,
        dependencies = Event(device(arch)))

    wait(device(arch), event)
end

@kernel function _calc_ab2_tendencies!(G⁻, Gⁿ, χ)
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻[i, j, k] = (1.5 + χ) *  Gⁿ[i, j, k] - G⁻[i, j, k] * (0.5 + χ)
end

"""
Explicitly step forward η in substeps.
"""
ab2_step_free_surface!(free_surface::SplitExplicitFreeSurface, model, Δt, χ, prognostic_field_events) =
    split_explicit_free_surface_step!(free_surface, model, Δt, χ, prognostic_field_events)
    
function initialize_free_surface_state!(sefs::SplitExplicitFreeSurface, grid, velocities)
    barotropic_mode!(sefs.state.Ũ, sefs.state.Ṽ, grid, velocities.u, velocities.v)
    fill_halo_regions!((sefs.state.Ũ, sefs.state.Ṽ))
    masking_events_U = mask_immersed_reduced_field_xy!(sefs.state.Ũ, k = 1) 
    masking_events_V = mask_immersed_reduced_field_xy!(sefs.state.Ṽ, k = 1) 
    wait(device(architecture(grid)), MultiEvent((masking_events_U, masking_events_V)))
end

function split_explicit_free_surface_step!(free_surface::SplitExplicitFreeSurface, model, Δt, χ, prognostic_field_events)

    grid = model.grid
    arch = architecture(grid)

    # we start the time integration of η from the average ηⁿ     
    η         = free_surface.η
    state     = free_surface.state
    auxiliary = free_surface.auxiliary
    settings  = free_surface.settings
    g         = free_surface.gravitational_acceleration

    Gu⁻ = model.timestepper.G⁻.u
    Gv⁻ = model.timestepper.G⁻.v

    Δτ = Δt * settings.Δτ  # we evolve for two times the Δt 

    event_Gu = launch!(arch, grid, :xyz, _calc_ab2_tendencies!, Gu⁻, model.timestepper.Gⁿ.u, χ)
    event_Gv = launch!(arch, grid, :xyz, _calc_ab2_tendencies!, Gv⁻, model.timestepper.Gⁿ.v, χ)

    # Wait for predictor velocity update step to complete and mask it if immersed boundary.
    @apply_regionally prognostic_field_events = wait_velocity_event(arch,  prognostic_field_events)

    masking_events = Tuple(mask_immersed_field!(q) for q in model.velocities)
    masking_events_U = mask_immersed_reduced_field_xy!(state.Ũ, k = 1) 
    masking_events_V = mask_immersed_reduced_field_xy!(state.Ṽ, k = 1) 
    wait(device(arch), MultiEvent(tuple(masking_events_U, masking_events_V, masking_events..., event_Gu, event_Gv)))

    barotropic_mode!(auxiliary.Gᵁ, auxiliary.Gⱽ, grid, Gu⁻, Gv⁻)

    # reset free surface averages
    initialize_free_surface_state!(state, η)

    # Solve for the free surface at tⁿ⁺¹

    for substep in 1:settings.substeps
        split_explicit_free_surface_substep!(η, state, auxiliary, settings, arch, grid, g, Δτ, substep)
    end
        
    # Reset eta for the next timestep
    # this is the only way in which η̅ is used: as a smoother for the 
    # substepped η field
    set!(η, free_surface.state.η̅)

    fill_halo_regions!(η)

    return prognostic_field_events
end