using Oceananigans.Fields: VelocityFields, ZeroField
using Oceananigans.Grids: inactive_node, peripheral_node
using Oceananigans.BuoyancyFormulations: ∂x_b, ∂y_b, ∂z_b
using Oceananigans.TimeSteppers: implicit_step!
using Oceananigans.Units

# Fallback
compute_eddy_velocities!(diffusivities, closure, model; parameters = :xyz) = nothing
compute_eddy_velocities!(diffusivities, ::NoSkewAdvectionISSD, model; parameters = :xyz) = nothing

import Oceananigans.BoundaryConditions: fill_halo_regions!

@inline fill_halo_regions!(::Oceananigans.Solvers.BatchedTridiagonalSolver, args...; kwargs...) = nothing

function compute_eddy_velocities!(diffusivities, closure::SkewAdvectionISSD, model; parameters = :xyz)
    uₑ = diffusivities.u
    vₑ = diffusivities.v
    wₑ = diffusivities.w
    Ψx = diffusivities.Ψx
    Ψy = diffusivities.Ψy
    Is = diffusivities.implicit_solver

    buoyancy = model.buoyancy
    grid     = model.grid
    clock    = model.clock
    model_fields = fields(model)

    launch!(architecture(grid), grid, parameters, _advance_eddy_streamfunctions!,
            Ψx, Ψy, grid, clock, closure, buoyancy, model_fields)

    for Ψ in (Ψx, Ψy)    
        implicit_step!(Ψ,
                       Is,
                       model.closure,
                       model.diffusivity_fields,
                       nothing,
                       model.clock,
                       25minutes)
    end

    launch!(architecture(grid), grid, parameters, _compute_eddy_velocities!,
            uₑ, vₑ, wₑ, grid, Ψx, Ψy)

    return nothing
end

"""
    tapering_factor(Sx, Sy, slope_limiter)

Return the tapering factor `min(1, Sₘₐₓ² / S²)`, where `S² = Sx² + Sy²`
that multiplies all components of the isopycnal slope tensor. 

References
==========
R. Gerdes, C. Koberle, and J. Willebrand. (1991), "The influence of numerical advection schemes
    on the results of ocean general circulation models", Clim. Dynamics, 5 (4), 211–226.
"""
@inline function tapering_factor(Sx, Sy, slope_limiter::FluxTapering)
    S²  = Sx^2 + Sy^2
    Sₘ² = slope_limiter.max_slope^2 
    return min(one(S²), Sₘ² / S²)
end

# Slope in x-direction at F, C, F locations, zeroed out on peripheries
@inline function Sxᶠᶜᶠ(i, j, k, grid, b, C) 
    bx = ℑzᵃᵃᶠ(i, j, k, grid, ∂x_b, b, C) 
    bz = ℑxᶠᵃᵃ(i, j, k, grid, ∂z_b, b, C)
    Sx = ifelse((bz == 0) & (bx == 0), zero(grid), - bx / bz)
    
    # Impose a boundary condition on immersed peripheries
    inactive = peripheral_node(i, j, k, grid, Face(), Center(), Face())

    return ifelse(inactive, zero(grid), Sx)
end

# Slope in y-direction at F, C, F locations, zeroed out on peripheries
@inline function Syᶜᶠᶠ(i, j, k, grid, b, C) 
    by = ℑzᵃᵃᶠ(i, j, k, grid, ∂y_b, b, C) 
    bz = ℑyᵃᶠᵃ(i, j, k, grid, ∂z_b, b, C)
    Sy = ifelse((bz == 0) & (by == 0), zero(grid), - by / bz)

    # Impose a boundary condition on immersed peripheries
    inactive = peripheral_node(i, j, k, grid, Center(), Face(), Face())
    
    return ifelse(inactive, zero(grid), Sy)
end

# tapered slope in x-direction at F, C, F locations
@inline function ϵSxᶠᶜᶠ(i, j, k, grid, slope_limiter, b, C)
    Sx = Sxᶠᶜᶠ(i, j, k, grid, b, C) 
    ϵ  = tapering_factor(Sx, zero(grid), slope_limiter)
    return Sx
end

# tapered slope in y-direction at F, C, F locations
@inline function ϵSyᶜᶠᶠ(i, j, k, grid, slope_limiter, b, C)
    Sy = Syᶜᶠᶠ(i, j, k, grid, b, C) 
    ϵ  = tapering_factor(zero(grid), Sy, slope_limiter)
    return Sy
end

@inline function κ_ϵSxᶠᶜᶠ(i, j, k, grid, clk, sl, κ, b, fields) 
    κ = κᶠᶜᶠ(i, j, k, grid, issd_coefficient_loc, κ, clk.time, fields) 
    ϵS = ϵSxᶠᶜᶠ(i, j, k, grid, sl, b, fields)
    return κ * ϵS
end

@inline function κ_ϵSyᶜᶠᶠ(i, j, k, grid, clk, sl, κ, b, fields) 
    κ = κᶜᶠᶠ(i, j, k, grid, issd_coefficient_loc, κ, clk.time, fields) 
    ϵS = ϵSyᶜᶠᶠ(i, j, k, grid, sl, b, fields)
    return κ * ϵS
end

@kernel function _advance_eddy_streamfunctions!(Ψx, Ψy, grid, clock, closure, buoyancy, fields)
    i, j, k = @index(Global, NTuple)

    closure = getclosure(i, j, closure)
    κ = closure.κ_skew
    slope_limiter = closure.slope_limiter
    Δt = clock.last_Δt

    @inbounds begin
        GΨx = (κ_ϵSxᶠᶜᶠ(i, j, k, grid, clock, slope_limiter, κ, buoyancy, fields) - Ψx[i, j, k]) / 100days
        GΨy = (κ_ϵSyᶜᶠᶠ(i, j, k, grid, clock, slope_limiter, κ, buoyancy, fields) - Ψy[i, j, k]) / 100days

        # Advance streamfunctions
        Ψx[i, j, k] += 25minutes * GΨx 
        Ψy[i, j, k] += 25minutes * GΨy 
    end
end

@kernel function _compute_eddy_velocities!(uₑ, vₑ, wₑ, grid, Ψx, Ψy)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        uₑ[i, j, k] = - ∂zᶠᶜᶜ(i, j, k, grid, Ψx) 
        vₑ[i, j, k] = - ∂zᶜᶠᶜ(i, j, k, grid, Ψy) 

        wˣ = δxᶜᶜᶠ(i, j, k, grid, Δy_qᶠᶜᶠ, Ψx) 
        wʸ = δyᶜᶜᶠ(i, j, k, grid, Δx_qᶜᶠᶠ, Ψy) 
        
        wₑ[i, j, k] = (wˣ + wʸ) * Az⁻¹ᶜᶜᶠ(i, j, k, grid)
    end
end

# Single closure version
@inline closure_auxiliary_velocity(clo, K, val_tracer_name) = nothing
@inline closure_auxiliary_velocity(::NoSkewAdvectionISSD, K, val_tracer_name) = nothing
@inline closure_auxiliary_velocity(::SkewAdvectionISSD, K, val_tracer_name) = (u = K.u, v = K.v, w = K.w)

# 2-tuple closure
@inline select_velocities(::Nothing, U) = U
@inline select_velocities(U, ::Nothing) = U
@inline select_velocities(::Nothing, ::Nothing) = nothing

# 3-tuple closure
@inline select_velocities(U, ::Nothing, ::Nothing) = U
@inline select_velocities(::Nothing, U, ::Nothing) = U
@inline select_velocities(::Nothing, ::Nothing, U) = U
@inline select_velocities(::Nothing, ::Nothing, ::Nothing) = nothing

# 4-tuple closure
@inline select_velocities(U, ::Nothing, ::Nothing, ::Nothing) = U
@inline select_velocities(::Nothing, U, ::Nothing, ::Nothing) = U
@inline select_velocities(::Nothing, ::Nothing, U, ::Nothing) = U
@inline select_velocities(::Nothing, ::Nothing, ::Nothing, U) = U
@inline select_velocities(::Nothing, ::Nothing, ::Nothing, ::Nothing) = nothing

# Handle tuple of closures.
# Assumption: there is only one ISSD closure in the tuple of closures.
@inline closure_auxiliary_velocity(closures::Tuple{<:Any}, Ks, val_tracer_name) =
            closure_auxiliary_velocity(closures[1], Ks[1], val_tracer_name)

@inline closure_auxiliary_velocity(closures::Tuple{<:Any, <:Any}, Ks, val_tracer_name) =
    select_velocities(closure_auxiliary_velocity(closures[1], Ks[1], val_tracer_name),
                      closure_auxiliary_velocity(closures[2], Ks[2], val_tracer_name))

@inline closure_auxiliary_velocity(closures::Tuple{<:Any, <:Any, <:Any}, Ks, val_tracer_name) =
    select_velocities(closure_auxiliary_velocity(closures[1], Ks[1], val_tracer_name),
                      closure_auxiliary_velocity(closures[2], Ks[2], val_tracer_name),
                      closure_auxiliary_velocity(closures[3], Ks[3], val_tracer_name))

@inline closure_auxiliary_velocity(closures::Tuple{<:Any, <:Any, <:Any, <:Any}, Ks, val_tracer_name) =
    select_velocities(closure_auxiliary_velocity(closures[1], Ks[1], val_tracer_name),
                      closure_auxiliary_velocity(closures[2], Ks[2], val_tracer_name),
                      closure_auxiliary_velocity(closures[3], Ks[3], val_tracer_name),
                      closure_auxiliary_velocity(closures[4], Ks[4], val_tracer_name))

@inline closure_auxiliary_velocity(closures::Tuple, Ks, val_tracer_name) =
    select_velocities(closure_auxiliary_velocity(closures[1], Ks[1], val_tracer_name),
                      closure_auxiliary_velocity(closures[2], Ks[2], val_tracer_name),
                      closure_auxiliary_velocity(closures[3], Ks[3], val_tracer_name),
                      closure_auxiliary_velocity(closures[4:end], Ks[4:end], val_tracer_name))
