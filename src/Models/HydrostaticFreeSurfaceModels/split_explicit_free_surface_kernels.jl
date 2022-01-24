using KernelAbstractions: @index, @kernel, Event
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Utils
using Oceananigans.AbstractOperations: Δz  
using Oceananigans.BoundaryConditions
using Oceananigans.Operators

# Evolution Kernels
#=
∂t(η) = -∇⋅U
∂t(U) = - gH∇η + f
=#

@kernel function split_explicit_free_surface_substep_kernel_1!(grid, Δτ, η, U, V, Gᵁ, Gⱽ, g, Hᶠᶜ, Hᶜᶠ)
    i, j = @index(Global, NTuple)
    # ∂τ(U) = - ∇η + G
    @inbounds U[i, j, 1] +=  Δτ * (-g * Hᶠᶜ[i, j] * ∂xᶠᶜᵃ(i, j, 1, grid, η) + Gᵁ[i, j, 1])
    @inbounds V[i, j, 1] +=  Δτ * (-g * Hᶜᶠ[i, j] * ∂yᶜᶠᵃ(i, j, 1, grid, η) + Gⱽ[i, j, 1])
end

@kernel function split_explicit_free_surface_substep_kernel_2!(grid, Δτ, η, U, V, η̅, U̅, V̅, velocity_weight, free_surface_weight)
    i, j = @index(Global, NTuple)
    # ∂τ(η) = - ∇⋅U
    @inbounds η[i, j, 1] -=  Δτ * div_xyᶜᶜᵃ(i, j, 1, grid, U, V)
    # time-averaging
    @inbounds U̅[i, j, 1] +=  velocity_weight * U[i, j, 1]
    @inbounds V̅[i, j, 1] +=  velocity_weight * V[i, j, 1]
    @inbounds η̅[i, j, 1] +=  free_surface_weight * η[i, j, 1]
end

function split_explicit_free_surface_substep!(η, state, auxiliary, settings, arch, grid, g, Δτ, substep_index)
    # unpack state quantities, parameters and forcing terms 
    U, V, η̅, U̅, V̅     = state.U, state.V, state.η̅, state.U̅, state.V̅
    Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ  = auxiliary.Gᵁ, auxiliary.Gⱽ, auxiliary.Hᶠᶜ, auxiliary.Hᶜᶠ

    vel_weight = settings.velocity_weights[substep_index]
    η_weight   = settings.free_surface_weights[substep_index]

    fill_halo_regions!(η, arch)

    event = launch!(arch, grid, :xy, split_explicit_free_surface_substep_kernel_1!, 
            grid, Δτ, η, U, V, Gᵁ, Gⱽ, g, Hᶠᶜ, Hᶜᶠ,
            dependencies=Event(device(arch)))

    wait(device(arch), event)

    # U, V has been updated thus need to refill halo
    fill_halo_regions!(U, arch)
    fill_halo_regions!(V, arch)

    event = launch!(arch, grid, :xy, split_explicit_free_surface_substep_kernel_2!, 
            grid, Δτ, η, U, V, η̅, U̅, V̅, vel_weight, η_weight,
            dependencies=Event(device(arch)))

    wait(device(arch), event)
            
end

# Barotropic Model Kernels
# u_Δz = u * Δz

@kernel function barotropic_mode_kernel!(U, V, u, v, grid)
    i, j = @index(Global, NTuple)

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
    sum!(U, u * Δz)
    sum!(V, v * Δz)

    arch = architecture(grid)
    fill_halo_regions!(U, arch)
    fill_halo_regions!(V, arch)
end

function set_average_to_zero!(free_surface_state)
    fill!(free_surface_state.η̅, 0.0)
    fill!(free_surface_state.U̅, 0.0)
    fill!(free_surface_state.V̅, 0.0)     
end

@kernel function barotropic_split_explicit_corrector_kernel!(u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        u[i, j, k] = u[i, j, k] + (-U[i, j] + U̅[i, j]) / Hᶠᶜ[i, j]
        v[i, j, k] = v[i, j, k] + (-V[i, j] + V̅[i, j]) / Hᶜᶠ[i, j]
    end
end

# may need to do Val(Nk) since it may not be known at compile. Also figure out where to put H
function barotropic_split_explicit_corrector!(u, v, free_surface, grid)
    sefs = free_surface.state
    U, V, U̅, V̅ = sefs.U, sefs.V, sefs.U̅, sefs.V̅
    Hᶠᶜ, Hᶜᶠ = free_surface.auxiliary.Hᶠᶜ, free_surface.auxiliary.Hᶜᶠ
    arch = architecture(grid)

    # take out "bad" barotropic mode, 
    # !!!! reusing U and V for this storage since last timestep doesn't matter
    barotropic_mode!(U, V, grid, u, v)
    # add in "good" barotropic mode

    event = launch!(arch, grid, :xyz, barotropic_split_explicit_corrector_kernel!,
        u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ,
        dependencies = Event(device(arch)))

    wait(device(arch), event)
end

@inline calc_ab2_tendencies(Gⁿ, G⁻, χ) = (convert(eltype(Gⁿ), (1.5)) + χ) * Gⁿ - (convert(eltype(Gⁿ), (0.5)) + χ) * G⁻

"""
Explicitly step forward η in substeps.
"""
ab2_step_free_surface!(free_surface::SplitExplicitFreeSurface, model, Δt, χ, velocities_update) =
    split_explicit_free_surface_step!(free_surface, model, Δt, χ, velocities_update)

function split_explicit_free_surface_step!(free_surface::SplitExplicitFreeSurface, model, Δt, χ, velocities_update)

    grid = model.grid
    arch = architecture(grid)

    # we start the time integration of η from the average ηⁿ     
    η = free_surface.η
    state = free_surface.state
    auxiliary = free_surface.auxiliary
    settings = free_surface.settings
    g = free_surface.gravitational_acceleration

    U, V = (state.U, state.V)
    Δτ = 2 * Δt / settings.substeps  # we evolve for two times the Δt 

    u, v, _ = model.velocities # this is u⋆

    Gu = calc_ab2_tendencies(model.timestepper.Gⁿ.u, model.timestepper.G⁻.u, χ)
    Gv = calc_ab2_tendencies(model.timestepper.Gⁿ.v, model.timestepper.G⁻.v, χ)

    # reset free surface averages
    set_average_to_zero!(state)

    # Wait for predictor velocity update step to complete and mask it if immersed boundary.
    wait(device(arch), velocities_update)
    masking_events = Tuple(mask_immersed_field!(q) for q in model.velocities)
    wait(device(arch), MultiEvent(masking_events))

    # Compute barotropic mode of tendency fields
    barotropic_mode!(auxiliary.Gᵁ, auxiliary.Gⱽ, grid, Gu, Gv)

    # Solve for the free surface at tⁿ⁺¹
    start_time = time_ns()

    for substep in 1:settings.substeps
        split_explicit_free_surface_substep!(η, state, auxiliary, settings, arch, grid, g, Δτ, substep)
    end
        
    # Reset eta for the next timestep
    # this is the only way in which η̅ is used: as a smoother for the 
    # substepped η field
    set!(η, free_surface.state.η̅)

    @debug "Split explicit step solve took $(prettytime((time_ns() - start_time) * 1e-9))."

    fill_halo_regions!(η, arch)

    return NoneEvent()
end