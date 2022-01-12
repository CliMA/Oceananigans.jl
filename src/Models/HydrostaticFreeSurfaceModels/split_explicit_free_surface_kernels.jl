using KernelAbstractions: @index, @kernel, Event
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Utils
using Oceananigans.BoundaryConditions
using Oceananigans.Operators

# Evolution Kernels
#=
∂t(η) = -∇⋅U⃗ 
∂t(U⃗) = - gH∇η + f⃗
=#

@kernel function free_surface_substep_kernel_1!(grid, Δτ, η, U, V, Gᵁ, Gⱽ, g, Hᶠᶜ, Hᶜᶠ)
    i, j = @index(Global, NTuple)
    # ∂τ(U⃗) = - ∇η + G⃗
    @inbounds U[i, j, 1] +=  Δτ * (-g * Hᶠᶜ[i, j] * ∂xᶠᶜᵃ(i, j, 1, grid, η) + Gᵁ[i, j, 1])
    @inbounds V[i, j, 1] +=  Δτ * (-g * Hᶜᶠ[i, j] * ∂yᶜᶠᵃ(i, j, 1, grid, η) + Gⱽ[i, j, 1])
end

@kernel function free_surface_substep_kernel_2!(grid, Δτ, η, U, V, η̅, U̅, V̅, velocity_weight, free_surface_weight)
    i, j = @index(Global, NTuple)
    # ∂τ(U⃗) = - ∇η + G⃗
    @inbounds η[i, j, 1] -=  Δτ * div_xyᶜᶜᵃ(i, j, 1, grid, U, V)
    # time-averaging
    @inbounds U̅[i, j, 1] +=  velocity_weight * U[i, j, 1]
    @inbounds V̅[i, j, 1] +=  velocity_weight * V[i, j, 1]
    @inbounds η̅[i, j, 1] +=  free_surface_weight * η[i, j, 1]
end

function free_surface_substep!(arch, grid, Δτ, free_surface::SplitExplicitFreeSurface, substep_index)
    sefs = free_surface #split explicit free surface
    U, V, η̅, U̅, V̅, Gᵁ, Gⱽ  = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
    Hᶠᶜ, Hᶜᶠ = sefs.Hᶠᶜ, sefs.Hᶜᶠ
    g = sefs.parameters.g
    velocity_weight = sefs.velocity_weights[substep_index]
    free_surface_weight = sefs.free_surface_weights[substep_index]

    fill_halo_regions!(η, arch)

    event = launch!(arch, grid, :xy, free_surface_substep_kernel_1!, 
            grid, Δτ, η, U, V, Gᵁ, Gⱽ, g, Hᶠᶜ, Hᶜᶠ,
            dependencies=Event(device(arch)))

    wait(device(arch), event)

    # U, V has been updated thus need to refill halo
    fill_halo_regions!(U, arch)
    fill_halo_regions!(V, arch)

    event = launch!(arch, grid, :xy, free_surface_substep_kernel_2!, 
            grid, Δτ, η, U, V, η̅, U̅, V̅, velocity_weight, free_surface_weight,
            dependencies=Event(device(arch)))

    wait(device(arch), event)
            
end

# Barotropic Model Kernels
# u_Δz = u * Δz

@kernel function barotropic_mode_kernel!(U, V, u, v, grid)
    i, j = @index(Global, NTuple)

    # hand unroll first loop 
    # FIXME: this algorithm will not work with bathymetry until
    # Oceananigans has vertical spacing operators with horizontal locations,
    # see https://github.com/CliMA/Oceananigans.jl/issues/2049
    @inbounds U[i, j, 1] = Δzᵃᵃᶜ(i, j, 1, grid) * u[i, j, 1]
    @inbounds V[i, j, 1] = Δzᵃᵃᶜ(i, j, 1, grid) * v[i, j, 1]

    @unroll for k in 2:grid.Nz
        @inbounds U[i, j, 1] += Δzᵃᵃᶜ(i, j, k, grid) * u[i, j, k] 
        @inbounds V[i, j, 1] += Δzᵃᵃᶜ(i, j, k, grid) * v[i, j, k] 
    end
end

# may need to do Val(Nk) since it may not be known at compile
function barotropic_mode!(U, V, arch, grid, u, v)
    event = launch!(arch, grid, :xy,
                    barotropic_mode_kernel!, 
                    U, V, u, v, grid,
                    dependencies = Event(device(arch)))

    wait(device(arch), event)        
end

@kernel function barotropic_split_explicit_corrector_kernel!(u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        u[i, j, k] = u[i, j, k] + (-U[i, j] + U̅[i, j]) / Hᶠᶜ[i, j]
        v[i, j, k] = v[i, j, k] + (-V[i, j] + V̅[i, j]) / Hᶜᶠ[i, j]
    end
end

# 
# may need to do Val(Nk) since it may not be known at compile. Also figure out where to put H
function barotropic_split_explicit_corrector!(u, v, free_surface, arch, grid)
    sefs = free_surface.state
    U, V, U̅, V̅ = sefs.U, sefs.V, sefs.U̅, sefs.V̅
    Hᶠᶜ, Hᶜᶠ = free_surface.auxiliary.Hᶠᶜ, free_surface.auxiliary.Hᶜᶠ

    # take out "bad" barotropic mode, 
    # !!!! reusing U and V for this storage since last timestep doesn't matter
    barotropic_mode!(U, V, arch, grid, u, v)
    # add in "good" barotropic mode

    event = launch!(arch, grid, :xyz, barotropic_split_explicit_corrector_kernel!,
        u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ,
        dependencies = Event(device(arch)))

    wait(device(arch), event)
end

@kernel function set_average_zero_kernel!(η̅, U̅, V̅)
    i, j = @index(Global, NTuple)
    @inbounds U̅[i, j, 1] = 0.0
    @inbounds V̅[i, j, 1] = 0.0
    @inbounds η̅[i, j, 1] = 0.0
end

function set_average_to_zero!(arch, grid, η̅, U̅, V̅)
    event = launch!(arch, grid, :xy, set_average_zero_kernel!, 
            η̅, U̅, V̅,
            dependencies=Event(device(arch)))
    wait(event)        
end