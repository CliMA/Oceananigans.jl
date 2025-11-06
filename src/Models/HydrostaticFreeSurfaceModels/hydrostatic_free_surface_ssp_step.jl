using Oceananigans.Fields: location, instantiated_location
using Oceananigans.TurbulenceClosures: implicit_step!
using Oceananigans.ImmersedBoundaries: get_active_cells_map, get_active_column_map
using Oceananigans.TimeSteppers: IMEXSSPTimeStepper

import Oceananigans.TimeSteppers: ssp_substep1!, ssp_substep2!, ssp_substep3!

ssp_substep1!(model::HydrostaticFreeSurfaceModel, args...) = ssp_substep!(Val(1), model, model.free_surface, model.grid, args...)
ssp_substep2!(model::HydrostaticFreeSurfaceModel, args...) = ssp_substep!(Val(2), model, model.free_surface, model.grid, args...)
ssp_substep3!(model::HydrostaticFreeSurfaceModel, args...) = ssp_substep!(Val(3), model, model.free_surface, model.grid, args...)

ssp_substep!(stage, model, free_surface, grid, Δt, callbacks) = throw(ErrorException("ssp_substep! not implemented for stage=$stage and model=$(typeof(model))"))

function store_tendencies!(G, Gⁿ)
    for key in propertynames(Gⁿ)
        if !isnothing(Gⁿ[key])
            parent(G[key]) .= parent(Gⁿ[key])
        end
    end
end

stage_tendency(Gⁱ, ::Val{1}) = Gⁱ.n₁ 
stage_tendency(Gⁱ, ::Val{2}) = Gⁱ.n₂ 
stage_tendency(Gⁱ, ::Val{3}) = Gⁱ.n₃

stage_free_surface_coefficient(::Val{1}) = d₁₂
stage_free_surface_coefficient(::Val{2}) = d₂₃
stage_free_surface_coefficient(::Val{3}) = nothing

stored_free_surface(::Val{1}, timestepper) = timestepper.Ψ⁻.η₁
stored_free_surface(::Val{2}, timestepper) = timestepper.Ψ⁻.η₂
stored_free_surface(::Val{3}, timestepper) = timestepper.Ψ⁻.η₃

# SET 1:

# b = sqrt(3) / 6 + 0.5
# c = - (sqrt(3) + 1) / 8

# a₁₁ = 1

# a₂₁ = 1/4
# a₂₂ = 1/4

# a₃₁ = 1/6
# a₃₂ = 1/6
# a₃₃ = 2/3

# d₁₁ = 4c + 2b
# d₁₂ = 1 - 4c - 2b

# d₂₁ = 1/2 - b - c
# d₂₂ = c
# d₂₃ = b

# d₃₁ = 1/6
# d₃₂ = 1/6
# d₃₃ = 2/3
# d₃₄ = 0

# SET 2:
 
a₁₁ = 1

a₂₁ = 1/4
a₂₂ = 1/4

a₃₁ = 1/6
a₃₂ = 1/6
a₃₃ = 2/3

d₁₁ = 0
d₁₂ = 1

d₂₁ =  1/6
d₂₂ = -1/3
d₂₃ =  2/3

d₃₁ = 1/6
d₃₂ = 1/6
d₃₃ = 2/3
d₃₄ = 0

# SET 3:

# a₁₁ = 0.711664700366941

# a₂₁ = 0.077338168947683
# a₂₂ = 0.917273367886007

# a₃₁ = 0.398930808264688
# a₃₂ = 0.345755244189623
# a₃₃ = 0.255313947545689

# d₁₁ = 0.353842865099275
# d₁₂ = 0.353842865099275

# d₂₁ = 0.398930808264689
# d₂₂ = 0.345755244189622
# d₂₃ = 0.255313947545689

# d₃₁ = 0.398930808264688
# d₃₂ = 0.345755244189623
# d₃₃ = 0.255313947545689
# d₃₄ = 0

# SET 4:

# a₁₁ = 1/3

# a₂₁ = 1/6
# a₂₂ = 1/2

# a₃₁ = 1/2
# a₃₂ = -1/2
# a₃₃ = 1

# d₁₁ = 1/6
# d₁₂ = 1/6

# d₂₁ = 1/3
# d₂₂ = 0
# d₂₃ = 1/3

# d₃₁ = 
# d₃₂ = 
# d₃₃ = 
# d₃₄ = 

@kernel function _compute_free_surface_rhs!(rhs, grid, g, Δt, U, η, ::Val{1}, D₁, D₂, D₃)
    i, j = @index(Global, NTuple)
    kᴺ⁺¹ = grid.Nz + 1
    δx_U = δxᶜᶜᶜ(i, j, kᴺ⁺¹, grid, Δy_qᶠᶜᶜ, barotropic_U, nothing, U.u)
    δy_V = δyᶜᶜᶜ(i, j, kᴺ⁺¹, grid, Δx_qᶜᶠᶜ, barotropic_V, nothing, U.v)
    
    D★ = (δx_U + δy_V) 
    Az = Azᶜᶜᶠ(i, j, kᴺ⁺¹, grid)

    @inbounds rhs[i, j, kᴺ⁺¹] = (d₁₁ * D₁[i, j, kᴺ⁺¹] + d₁₂ * D★ - Az * η[i, j, kᴺ⁺¹] / Δt) / (g * Δt)
end

@kernel function _compute_free_surface_rhs!(rhs, grid, g, Δt, U, η, ::Val{2}, D₁, D₂, D₃)
    i, j = @index(Global, NTuple)
    kᴺ⁺¹ = grid.Nz + 1
    δx_U = δxᶜᶜᶜ(i, j, kᴺ⁺¹, grid, Δy_qᶠᶜᶜ, barotropic_U, nothing, U.u)
    δy_V = δyᶜᶜᶜ(i, j, kᴺ⁺¹, grid, Δx_qᶜᶠᶜ, barotropic_V, nothing, U.v)
    
    D★ = (δx_U + δy_V)
    Az = Azᶜᶜᶠ(i, j, kᴺ⁺¹, grid)

    @inbounds rhs[i, j, kᴺ⁺¹] = (d₂₁ * D₁[i, j, kᴺ⁺¹] + d₂₂ * D₂[i, j, kᴺ⁺¹] + d₂₃ * D★ - Az * η[i, j, kᴺ⁺¹] / Δt) / (g * Δt)
end

@kernel function _compute_free_surface_rhs!(rhs, grid, g, Δt, U, η, ::Val{3}, D₁, D₂, D₃)
    i, j = @index(Global, NTuple)
    kᴺ⁺¹ = grid.Nz + 1
    # δx_U = δxᶜᶜᶜ(i, j, kᴺ⁺¹, grid, Δy_qᶠᶜᶜ, barotropic_U, nothing, U.u)
    # δy_V = δyᶜᶜᶜ(i, j, kᴺ⁺¹, grid, Δx_qᶜᶠᶜ, barotropic_V, nothing, U.v)
    
    # D★ = (δx_U + δy_V)
    # Az = Azᶜᶜᶠ(i, j, kᴺ⁺¹, grid)

    @inbounds rhs[i, j, kᴺ⁺¹] = (d₃₁ * D₁[i, j, kᴺ⁺¹] + d₃₂ * D₂[i, j, kᴺ⁺¹] + d₃₃ * D₃[i, j, kᴺ⁺¹]) * Az⁻¹ᶜᶜᶠ(i, j, kᴺ⁺¹, grid)
end

function step_free_surface!(val_stage, free_surface::ImplicitFreeSurface, model, timestepper::IMEXSSPTimeStepper, Δt)
    η      = free_surface.η
    g      = free_surface.gravitational_acceleration
    rhs    = free_surface.implicit_step_solver.right_hand_side
    solver = free_surface.implicit_step_solver
    arch   = model.architecture
    grid   = model.grid

    parent(free_surface.η) .= parent(timestepper.Ψ⁻.η)
    mask_immersed_field!(model.velocities.u)
    mask_immersed_field!(model.velocities.v)
    fill_halo_regions!(model.velocities, model.clock, fields(model))

    D₁ = timestepper.Dⁱ.n₁.η
    D₂ = timestepper.Dⁱ.n₂.η
    D₃ = timestepper.Dⁱ.n₃.η
    launch!(arch, grid, :xy, _compute_free_surface_rhs!, rhs, grid, g, Δt, model.velocities, η, val_stage, D₁, D₂, D₃)

    C = stage_free_surface_coefficient(val_stage)
    solve!(val_stage, η, solver, rhs, g, Δt, C)
    fill_halo_regions!(η)

    return nothing
end

solve!(::Val{1}, η, solver, rhs, g, Δt, C) = solve!(η, solver, rhs, g, Δt, C)
solve!(::Val{2}, η, solver, rhs, g, Δt, C) = solve!(η, solver, rhs, g, Δt, C)
solve!(::Val{3}, η, solver, rhs, g, Δt, C) = parent(η) .-= Δt .* parent(rhs)

@kernel function _store_free_surface_rhs!(D, grid, U)
    i, j = @index(Global, NTuple)
    kᴺ⁺¹ = grid.Nz + 1
    δx_U = δxᶜᶜᶜ(i, j, kᴺ⁺¹, grid, Δy_qᶠᶜᶜ, barotropic_U, nothing, U.u)
    δy_V = δyᶜᶜᶜ(i, j, kᴺ⁺¹, grid, Δx_qᶜᶠᶜ, barotropic_V, nothing, U.v)
    @inbounds D.η[i, j, kᴺ⁺¹] = δx_U + δy_V
end

@inline function ssp_substep!(val_stage, model, ::ImplicitFreeSurface, grid, Δt, callbacks)

    # Compute tendencies....
    compute_momentum_tendencies!(model, callbacks)
    compute_tracer_tendencies!(model)

    # Cache diagnostics that need to be reused later....
    store_tendencies!(stage_tendency(model.timestepper.Gⁱ, val_stage), model.timestepper.Gⁿ)
    
    stored_η = stored_free_surface(val_stage, model.timestepper)  
    parent(stored_η) .= parent(model.free_surface.η)

    launch!(architecture(grid), grid, :xy, _store_free_surface_rhs!, 
            stage_tendency(model.timestepper.Dⁱ, val_stage), grid, model.velocities)

    # Finally Substep! Advance grid and (predictor) momentum 
    ssp_substep_grid!(grid, model, model.vertical_coordinate, Δt)
    ssp_substep_velocities!(model.velocities, model, Δt, val_stage, model.free_surface.gravitational_acceleration)

    # Advancing free surface in preparation for the correction step
    step_free_surface!(val_stage, model.free_surface, model, model.timestepper, Δt)
    
    # Correct for the updated barotropic mode
    C = stage_free_surface_coefficient(val_stage)
    correct_barotropic_mode!(val_stage, model, model.free_surface, Δt, C)

    # TODO: fill halo regions for horizontal velocities should be here before the tracer update.   
    ssp_substep_tracers!(model.tracers, model, Δt, val_stage)

    return nothing
end

correct_barotropic_mode!(val_stage, model, free_surface, Δt, C) = correct_barotropic_mode!(model, free_surface, Δt, C)
correct_barotropic_mode!(::Val{3},  model, free_surface, Δt, C) = nothing

# A Fallback to be extended for specific ztypes and grid types
ssp_substep_grid!(grid, model, ztype::ZCoordinate, Δt) = nothing

#####
##### Step Velocities
#####

function ssp_substep_velocities!(velocities, model, Δt, val_stage, g)
    grid = model.grid
    FT = eltype(grid)

    u, v, w = velocities

    Gu¹ = model.timestepper.Gⁱ.n₁.u
    Gu² = model.timestepper.Gⁱ.n₂.u
    Gu³ = model.timestepper.Gⁱ.n₃.u

    Gv¹ = model.timestepper.Gⁱ.n₁.v
    Gv² = model.timestepper.Gⁱ.n₂.v
    Gv³ = model.timestepper.Gⁱ.n₃.v

    u⁻ = model.timestepper.Ψ⁻.u
    v⁻ = model.timestepper.Ψ⁻.v
    η₁ = model.timestepper.Ψ⁻.η₁ 
    η₂ = model.timestepper.Ψ⁻.η₂
    η₃ = model.timestepper.Ψ⁻.η₃ 

    launch!(architecture(grid), grid, :xyz, _ssp_evolve_u_velocity!, 
            u, val_stage, grid, u⁻, g, convert(FT, Δt), 
            Gu¹, Gu², Gu³, η₁, η₂, η₃)

    launch!(architecture(grid), grid, :xyz, _ssp_evolve_v_velocity!, 
            v, val_stage, grid, v⁻, g, convert(FT, Δt), 
            Gv¹, Gv², Gv³, η₁, η₂, η₃)

    return nothing
end

#####
##### Step velocities Kernels
#####

@kernel function _ssp_evolve_u_velocity!(u, ::Val{1}, grid, u⁻, g, Δt, Gu¹, Gu², Gu³, η₁, η₂, η₃)
    i, j, k = @index(Global, NTuple)
    kᴺ⁺¹ = grid.Nz + 1
    ηx₁  = g * ∂xᶠᶜᶜ(i, j, kᴺ⁺¹, grid, η₁) 
    @inbounds u[i, j, k] = u⁻[i, j, k] + Δt * (a₁₁ * Gu¹[i, j, k] -
                                               d₁₁ * ηx₁)
end

@kernel function _ssp_evolve_v_velocity!(v, ::Val{1}, grid, v⁻, g, Δt, Gv¹, Gv², Gv³, η₁, η₂, η₃)
    i, j, k = @index(Global, NTuple)
    kᴺ⁺¹ = grid.Nz + 1
    ηy₁  = g * ∂yᶜᶠᶜ(i, j, kᴺ⁺¹, grid, η₁) 
    @inbounds v[i, j, k] = v⁻[i, j, k] + Δt * (a₁₁ * Gv¹[i, j, k] -
                                               d₁₁ * ηy₁)
end

@kernel function _ssp_evolve_u_velocity!(u, ::Val{2}, grid, u⁻, g, Δt, Gu¹, Gu², Gu³, η₁, η₂, η₃)
    i, j, k = @index(Global, NTuple)
    kᴺ⁺¹ = grid.Nz + 1
    ηx₁  = g * ∂xᶠᶜᶜ(i, j, kᴺ⁺¹, grid, η₁) 
    ηx₂  = g * ∂xᶠᶜᶜ(i, j, kᴺ⁺¹, grid, η₂) 
    @inbounds u[i, j, k] = u⁻[i, j, k] + Δt * (a₂₁ * Gu¹[i, j, k] +
                                               a₂₂ * Gu²[i, j, k] -
                                               d₂₁ * ηx₁ -
                                               d₂₂ * ηx₂)
end

@kernel function _ssp_evolve_v_velocity!(v, ::Val{2}, grid, v⁻, g, Δt, Gv¹, Gv², Gv³, η₁, η₂, η₃)
    i, j, k = @index(Global, NTuple)
    kᴺ⁺¹ = grid.Nz + 1
    ηy₁  = g * ∂yᶜᶠᶜ(i, j, kᴺ⁺¹, grid, η₁) 
    ηy₂  = g * ∂yᶜᶠᶜ(i, j, kᴺ⁺¹, grid, η₂) 
    @inbounds v[i, j, k] = v⁻[i, j, k] + Δt * (a₂₁ * Gv¹[i, j, k] +
                                               a₂₂ * Gv²[i, j, k] -
                                               d₂₁ * ηy₁ - 
                                               d₂₂ * ηy₂)
end

@kernel function _ssp_evolve_u_velocity!(u, ::Val{3}, grid, u⁻, g, Δt, Gu¹, Gu², Gu³, η₁, η₂, η₃)
    i, j, k = @index(Global, NTuple)
    kᴺ⁺¹ = grid.Nz + 1
    ηx₁  = g * ∂xᶠᶜᶜ(i, j, kᴺ⁺¹, grid, η₁) 
    ηx₂  = g * ∂xᶠᶜᶜ(i, j, kᴺ⁺¹, grid, η₂) 
    ηx₃  = g * ∂xᶠᶜᶜ(i, j, kᴺ⁺¹, grid, η₃) 
    @inbounds u[i, j, k] = u⁻[i, j, k] + Δt * (a₃₁ * Gu¹[i, j, k] +
                                               a₃₂ * Gu²[i, j, k] +
                                               a₃₃ * Gu³[i, j, k] -
                                               d₃₁ * ηx₁ -
                                               d₃₂ * ηx₂ - 
                                               d₃₃ * ηx₃)
end

@kernel function _ssp_evolve_v_velocity!(v, ::Val{3}, grid, v⁻, g, Δt, Gv¹, Gv², Gv³, η₁, η₂, η₃)
    i, j, k = @index(Global, NTuple)
    kᴺ⁺¹ = grid.Nz + 1
    ηy₁  = g * ∂yᶜᶠᶜ(i, j, kᴺ⁺¹, grid, η₁) 
    ηy₂  = g * ∂yᶜᶠᶜ(i, j, kᴺ⁺¹, grid, η₂)  
    ηy₃  = g * ∂yᶜᶠᶜ(i, j, kᴺ⁺¹, grid, η₃)
    @inbounds v[i, j, k] = v⁻[i, j, k] + Δt * (a₃₁ * Gv¹[i, j, k] +
                                               a₃₂ * Gv²[i, j, k] +
                                               a₃₃ * Gv³[i, j, k] -
                                               d₃₁ * ηy₁ - 
                                               d₃₂ * ηy₂ - 
                                               d₃₃ * ηy₃)
end

#####
##### Step Tracers
#####

ssp_substep_tracers!(::EmptyNamedTuple, model, Δt, val_stage) = nothing

function ssp_substep_tracers!(tracers, model, Δt, val_stage)

    closure = model.closure
    grid = model.grid
    FT = eltype(grid)

    # Tracer update kernels
    for (tracer_index, tracer_name) in enumerate(propertynames(tracers))
        G¹ = model.timestepper.Gⁱ.n₁[tracer_name]
        G² = model.timestepper.Gⁱ.n₂[tracer_name]
        G³ = model.timestepper.Gⁱ.n₃[tracer_name]
        c⁻ = model.timestepper.Ψ⁻[tracer_name]
        Ψ⁻ = model.timestepper.Ψ⁻[tracer_name]
        c  = tracers[tracer_name]
        closure = model.closure

        launch!(architecture(grid), grid, :xyz,
                _ssp_evolve_tracer!, c, val_stage, grid, Ψ⁻, convert(FT, Δt), G¹, G², G³)
    end

    return nothing
end

#####
##### Tracer update in mutable vertical coordinates
#####

@kernel function _ssp_evolve_tracer!(c, ::Val{1}, grid, σc⁻, Δt, Gc¹, Gc², Gc³)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = (σc⁻[i, j, k] + Δt * a₁₁ * Gc¹[i, j, k]) / σᶜᶜⁿ
end

@kernel function _ssp_evolve_tracer!(c, ::Val{2}, grid, σc⁻, Δt, Gc¹, Gc², Gc³)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = (σc⁻[i, j, k] + Δt * (a₂₁ * Gc¹[i, j, k] + 
                                                 a₂₂ * Gc²[i, j, k])) / σᶜᶜⁿ
end

@kernel function _ssp_evolve_tracer!(c, ::Val{3}, grid, σc⁻, Δt, Gc¹, Gc², Gc³)
    i, j, k = @index(Global, NTuple)
    σᶜᶜⁿ = σⁿ(i, j, k, grid, Center(), Center(), Center())
    @inbounds c[i, j, k] = (σc⁻[i, j, k] + Δt * (a₃₁ * Gc¹[i, j, k] + 
                                                 a₃₂ * Gc²[i, j, k] + 
                                                 a₃₃ * Gc³[i, j, k])) / σᶜᶜⁿ
end
