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

@inline δxᶠᵃᵃ_bound(i, j, k, grid, T, f::Function, args...) = δxᶠᵃᵃ(i, j, k, grid, f, args...)
@inline δyᵃᶠᵃ_bound(i, j, k, grid, T, f::Function, args...) = δyᵃᶠᵃ(i, j, k, grid, f, args...)
@inline δxᶜᵃᵃ_bound(i, j, k, grid, T, f::Function, args...) = δxᶜᵃᵃ(i, j, k, grid, f, args...)
@inline δyᵃᶜᵃ_bound(i, j, k, grid, T, f::Function, args...) = δyᵃᶜᵃ(i, j, k, grid, f, args...)

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

@inline conditional_value_fcf(i, j, k, grid, ibg::IBG, f::Function, args...) = ifelse(immersed_peripheral_node(i, j, k, ibg, f, c, f), zero(ibg), f(i, j, k, grid, args...))
@inline conditional_value_cff(i, j, k, grid, ibg::IBG, f::Function, args...) = ifelse(immersed_peripheral_node(i, j, k, ibg, c, f, f), zero(ibg), f(i, j, k, grid, args...))

@inline conditional_∂x_bound_f(LY, LZ, i, j, k, ibg::IBG{FT}, ∂x, args...) where FT = ifelse(immersed_inactive_node(i, j, k, ibg, c, LY, LZ) | immersed_inactive_node(i+1, j, k, ibg, c, LY, LZ), zero(ibg), ∂x(i, j, k, ibg.underlying_grid, args...))
@inline conditional_∂y_bound_f(LY, LZ, i, j, k, ibg::IBG{FT}, ∂y, args...) where FT = ifelse(immersed_inactive_node(i, j, k, ibg, f, LY, LZ) | immersed_inactive_node(i, j+1, k, ibg, f, LY, LZ), zero(ibg), ∂y(i, j, k, ibg.underlying_grid, args...))

@inline δxᶜᵃᵃ_bound(i, j, k, ibg::IBG, T, f::Function, args...) = δxᶜᵃᵃ_bound(i, j, k, ibg.underlying_grid, T, conditional_value_fcf, ibg, f, args...)
@inline δyᵃᶜᵃ_bound(i, j, k, ibg::IBG, T, f::Function, args...) = δyᵃᶜᵃ_bound(i, j, k, ibg.underlying_grid, T, conditional_value_cff, ibg, f, args...)

@inline ∂xᶠᶜᶠ_bound(i, j, k, ibg::IBG, T, η) = conditional_∂x_bound_f(c, f, i, j, k, ibg, ∂xᶠᶜᶠ_bound, T, η)
@inline ∂yᶜᶠᶠ_bound(i, j, k, ibg::IBG, T, η) = conditional_∂y_bound_f(c, f, i, j, k, ibg, ∂yᶜᶠᶠ_bound, T, η)        

for Topo in [:Periodic, :Bounded]
    @eval begin
        @inline δxᶜᵃᵃ_bound(i, j, k, ibg::IBG, T::Type{$Topo}, f::Function, args...) = δxᶜᵃᵃ_bound(i, j, k, ibg.underlying_grid, T, conditional_value_fcf, ibg, f, args...)
        @inline δyᵃᶜᵃ_bound(i, j, k, ibg::IBG, T::Type{$Topo}, f::Function, args...) = δyᵃᶜᵃ_bound(i, j, k, ibg.underlying_grid, T, conditional_value_cff, ibg, f, args...)

        @inline ∂xᶠᶜᶠ_bound(i, j, k, ibg::IBG, T::Type{$Topo}, η) = conditional_∂x_bound_f(c, f, i, j, k, ibg, ∂xᶠᶜᶠ_bound, T, η)
        @inline ∂yᶜᶠᶠ_bound(i, j, k, ibg::IBG, T::Type{$Topo}, η) = conditional_∂y_bound_f(c, f, i, j, k, ibg, ∂yᶜᶠᶠ_bound, T, η)        
    end
end

# AB3 step
@inline U★(i, j, k, grid, ::AdamsBashforth3Scheme, ϕᵐ, ϕᵐ⁻¹, ϕᵐ⁻²)       = α * ϕᵐ[i, j, k]   + θ * ϕᵐ⁻¹[i, j, k] + β * ϕᵐ⁻²[i, j, k]
@inline η★(i, j, k, grid, ::AdamsBashforth3Scheme, ηᵐ⁺¹, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) = δ * ηᵐ⁺¹[i, j, k] + μ * ηᵐ[i, j, k]   + γ * ηᵐ⁻¹[i, j, k] + ϵ * ηᵐ⁻²[i, j, k]

# Forward Backward Step
@inline U★(i, j, k, grid, ::ForwardBackwardScheme, ϕ, args...) = ϕ[i, j, k] 
@inline η★(i, j, k, grid, ::ForwardBackwardScheme, η, args...) = η[i, j, k] 

@inline advance_previous_velocity!(i, j, k, ::ForwardBackwardScheme, U, Uᵐ⁻¹, Uᵐ⁻²) = nothing

@inline function advance_previous_velocity!(i, j, k, ::AdamsBashforth3Scheme, U, Uᵐ⁻¹, Uᵐ⁻²)
    Uᵐ⁻²[i, j, k] = Uᵐ⁻¹[i, j, k] 
    Uᵐ⁻¹[i, j, k] =    U[i, j, k] 
end

@inline advance_previous_free_surface!(i, j, k, ::ForwardBackwardScheme, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) = nothing

@inline function advance_previous_free_surface!(i, j, k, ::AdamsBashforth3Scheme, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)
    ηᵐ⁻²[i, j, k] = ηᵐ⁻¹[i, j, k]
    ηᵐ⁻¹[i, j, k] =   ηᵐ[i, j, k]
      ηᵐ[i, j, k] =    η[i, j, k]
end

@kernel function split_explicit_free_surface_substep_kernel_1!(grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², 
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
        η[i′, j′, k_top] -= Δτ * (div_xᶜᶜᶠ_bound(i′, j′, k_top, grid, TX, U★, timestepper, U, Uᵐ⁻¹, Uᵐ⁻²) +
                                  div_yᶜᶜᶠ_bound(i′, j′, k_top, grid, TY, U★, timestepper, V, Vᵐ⁻¹, Vᵐ⁻²))
    end
end

@kernel function split_explicit_free_surface_substep_kernel_2!(grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², 
                                                               η̅, U̅, V̅, averaging_weight, 
                                                               Gᵁ, Gⱽ, g, Hᶠᶜ, Hᶜᶠ,
                                                               timestepper, offsets)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1
    
    i′ = i - offsets[1]
    j′ = j - offsets[2]

    TX, TY, _ = topology(grid)

    @inbounds begin 
        advance_previous_velocity!(i′, j′, 1, timestepper, U, Uᵐ⁻¹, Uᵐ⁻²)
        advance_previous_velocity!(i′, j′, 1, timestepper, V, Vᵐ⁻¹, Vᵐ⁻²)

        # ∂τ(U) = - ∇η + G
        U[i′, j′, 1] +=  Δτ * (- g * Hᶠᶜ[i′, j′] * ∂xᶠᶜᶠ_bound(i′, j′, k_top, grid, TX, η★, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) + Gᵁ[i′, j′, 1])
        V[i′, j′, 1] +=  Δτ * (- g * Hᶜᶠ[i′, j′] * ∂yᶜᶠᶠ_bound(i′, j′, k_top, grid, TY, η★, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) + Gⱽ[i′, j′, 1])
                          
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

    event = launch!(arch, grid, kernel_size, split_explicit_free_surface_substep_kernel_1!, args...;
            dependencies=Event(device(arch)))

    event = launch!(arch, grid, kernel_size, split_explicit_free_surface_substep_kernel_2!, args...;
            dependencies=event)

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
end

function initialize_free_surface!(free_surface_state, η)
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
    @apply_regionally barotropic_mode!(sefs.state.U̅, sefs.state.V̅, grid, velocities.u, velocities.v)
    fill_halo_regions!((sefs.state.U̅, sefs.state.V̅))
end

function split_explicit_free_surface_step!(free_surface::SplitExplicitFreeSurface, model, Δt, χ, velocities_update)

    grid = free_surface.η.grid

    # we start the time integration of η from the average ηⁿ     
    Gu  = model.timestepper.G⁻.u
    Gv  = model.timestepper.G⁻.v
    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v

    velocities = model.velocities

    fill_halo_regions!((free_surface.state.U̅, free_surface.state.V̅))
    
    @apply_regionally velocities_update = setup_split_explicit!(free_surface.auxiliary, free_surface.state, free_surface.η, 
                                                                grid, Gu, Gv, Guⁿ, Gvⁿ, χ, velocities, velocities_update)

    fill_halo_regions!((free_surface.auxiliary.Gᵁ, free_surface.auxiliary.Gⱽ))

    # Solve for the free surface at tⁿ⁺¹
    @apply_regionally iterate_split_explicit!(free_surface, grid, Δt)
    
    # Reset eta for the next timestep
    # this is the only way in which η̅ is used: as a smoother for the 
    # substepped η field
    @apply_regionally set!(free_surface.η, free_surface.state.η̅)

    fill_halo_regions!(free_surface.η)

    return velocities_update
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
end

function setup_split_explicit!(auxiliary, state, η, grid, Gu, Gv, Guⁿ, Gvⁿ, χ, velocities, velocities_update)
    arch = architecture(grid)

    event_Gu = launch!(arch, grid, :xyz, _calc_ab2_tendencies!, Gu, Guⁿ, χ)
    event_Gv = launch!(arch, grid, :xyz, _calc_ab2_tendencies!, Gv, Gvⁿ, χ)

    # reset free surface averages
    initialize_free_surface!(state, η)

    # Wait for predictor velocity update step to complete and mask it if immersed boundary.
    wait(device(arch), MultiEvent(tuple(velocities_update[1]...)))

    masking_events = [mask_immersed_field!(q) for q in velocities]
    push!(masking_events, mask_immersed_field!(Gu))
    push!(masking_events, mask_immersed_field!(Gv))
    wait(device(arch), MultiEvent(tuple(masking_events..., event_Gu, event_Gv)))

    # Compute barotropic mode of tendency fields
    barotropic_mode!(auxiliary.Gᵁ, auxiliary.Gⱽ, grid, Gu, Gv)

    return MultiEvent(tuple(velocities_update[2]...))
end