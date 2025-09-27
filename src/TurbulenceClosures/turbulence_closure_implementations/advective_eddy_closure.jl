using Oceananigans.Fields: VelocityFields, ZeroField
using Oceananigans.Grids: inactive_node, peripheral_node
using Oceananigans.BuoyancyFormulations: ∂x_b, ∂y_b, ∂z_b
using Oceananigans.TimeSteppers: implicit_step!
using Oceananigans.Units
using Adapt

import Oceananigans.BoundaryConditions: fill_halo_regions!

struct EddyAdvectiveClosure{K, L, N} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, N}
      κ_skew :: K
    tapering :: L
    EddyAdvectiveClosure{N}(κ::K, tapering::L) where {K, L, N} = new{K, L, N}(κ, tapering)
end

function EddyAdvectiveClosure(; κ_skew = 1000.0, 
                                tapering = EddyEvolvingStreamfunction(500days),
                                required_halo_size::Int = 1) 

    return EddyAdvectiveClosure{required_halo_size}(κ_skew, tapering) 
end

const EAC = EddyAdvectiveClosure

# A particular approach that does not require tapering
struct EddyEvolvingStreamfunction{FT}
    time_scale :: FT
end

νzᶠᶜᶜ(i, j, k, grid, closure::FlavorOfISSD,  K, args...) = zero(grid)
νzᶜᶠᶜ(i, j, k, grid, closure::FlavorOfISSD,  K, args...) = zero(grid)
νzᶠᶜᶜ(i, j, k, grid, closure::EAC, K, args...) = zero(grid)
νzᶜᶠᶜ(i, j, k, grid, closure::EAC, K, args...) = zero(grid)

@inline diffusive_flux_x(i, j, k, grid, closure::EAC, K, ::Val{id}, args...) where id = zero(grid)
@inline diffusive_flux_y(i, j, k, grid, closure::EAC, K, ::Val{id}, args...) where id = zero(grid)
@inline diffusive_flux_z(i, j, k, grid, closure::EAC, K, ::Val{id}, args...) where id = zero(grid)

@inline viscous_flux_ux(i, j, k, grid, closure::EAC, args...) = zero(grid)
@inline viscous_flux_uy(i, j, k, grid, closure::EAC, args...) = zero(grid)
@inline viscous_flux_uz(i, j, k, grid, closure::EAC, args...) = zero(grid)
@inline viscous_flux_vx(i, j, k, grid, closure::EAC, args...) = zero(grid)
@inline viscous_flux_vy(i, j, k, grid, closure::EAC, args...) = zero(grid)
@inline viscous_flux_vz(i, j, k, grid, closure::EAC, args...) = zero(grid)
@inline viscous_flux_wx(i, j, k, grid, closure::EAC, args...) = zero(grid)
@inline viscous_flux_wy(i, j, k, grid, closure::EAC, args...) = zero(grid)
@inline viscous_flux_wz(i, j, k, grid, closure::EAC, args...) = zero(grid)

with_tracers(tracers, closure::EAC) = closure

struct EddyClosureDiffusivities{U, V, W, PX, PY, IS}
    u  :: U
    v  :: V
    w  :: W
    Ψx :: PX
    Ψy :: PY
    implicit_solver :: IS
end

Adapt.adapt_structure(to, K::EddyClosureDiffusivities) = 
    EddyClosureDiffusivities(adapt(to, K.u),
                             adapt(to, K.v),
                             adapt(to, K.w),
                             adapt(to, K.Ψx),
                             adapt(to, K.Ψy),
                             nothing)

function build_diffusivity_fields(grid, clock, tracer_names, bcs, closure::EAC) 

    U  = VelocityFields(grid)
    Ψx = Field{Face, Center, Face}(grid)
    Ψy = Field{Center, Face, Face}(grid)

    if closure.tapering isa EddyEvolvingStreamfunction
        implicit_solver = implicit_diffusion_solver(VerticallyImplicitTimeDiscretization(), grid)
    else
        implicit_solver = nothing
    end

    return EddyClosureDiffusivities(U.u, U.v, U.w, Ψx, Ψy, implicit_solver)
end

@inline function fill_halo_regions!(diffusivities::EddyClosureDiffusivities, args...; kwargs...) 
    fill_halo_regions!(diffusivities.u,  args...; kwargs...)
    fill_halo_regions!(diffusivities.v,  args...; kwargs...)
    fill_halo_regions!(diffusivities.w,  args...; kwargs...)
    fill_halo_regions!(diffusivities.Ψx, args...; kwargs...)
    fill_halo_regions!(diffusivities.Ψy, args...; kwargs...)
    return nothing
end

function compute_diffusivities!(diffusivities, closure::EddyAdvectiveClosure, model; kwargs...)
    uₑ = diffusivities.u
    vₑ = diffusivities.v
    wₑ = diffusivities.w
    Ψx = diffusivities.Ψx
    Ψy = diffusivities.Ψy
    grid  = model.grid
    clock = model.clock

    if isfinite(clock.last_Δt)
        Δt = clock.last_Δt
    else
        Δt = 0
    end

    # Include the periphery
    parameters = KernelParameters(0:size(grid, 1)+1, 0:size(grid, 2)+1, 0:size(grid, 3)+1)

    launch!(architecture(grid), grid, parameters, _advance_eddy_streamfunctions!,
            Ψx, Ψy, grid, Δt, clock, closure, model.buoyancy, fields(model), closure.tapering)

    diffuse_streamfunctions!(closure.tapering, 
                             diffusivities,
                             model.closure, # Note that these can be different than the closure passed to this function
                             model.diffusivity_fields, 
                             clock, Δt)

    mask_immersed_field!(Ψx)
    mask_immersed_field!(Ψy)

    launch!(architecture(grid), grid, parameters, _compute_eddy_velocities!, uₑ, vₑ, wₑ, grid, Ψx, Ψy)

    return nothing
end

diffuse_streamfunctions!(tapering, args...) = nothing

function diffuse_streamfunctions!(::EddyEvolvingStreamfunction, K, closure, diffusivities, clock, Δt)
    Ψx = K.Ψx
    Ψy = K.Ψy 

    implicit_step!(Ψx, K.implicit_solver, closure, diffusivities, nothing, clock, Δt)
    implicit_step!(Ψy, K.implicit_solver, closure, diffusivities, nothing, clock, Δt)

    return nothing
end

@inline function tapering_factor(Sx, Sy, tapering::FluxTapering)
    S²  = Sx^2 + Sy^2
    Sₘ² = tapering.max_slope^2 
    return min(one(S²), Sₘ² / S²)
end

# In any other case, we do not taper
@inline tapering_factor(Sx, Sy, tapering) = 1

# Slope in x-direction at F, C, F locations, zeroed out on peripheries
@inline function Sxᶠᶜᶠ(i, j, k, grid, b, C) 
    bx = ℑzᵃᵃᶠ(i, j, k, grid, ∂x_b, b, C) 
    bz = ℑxᶠᵃᵃ(i, j, k, grid, ∂z_b, b, C)
    Sx = ifelse(bz == 0, zero(grid), - bx / bz)
    
    # Impose a boundary condition on immersed peripheries
    inactive = peripheral_node(i, j, k, grid, Face(), Center(), Face())

    return ifelse(inactive, zero(grid), Sx)
end

# Slope in y-direction at F, C, F locations, zeroed out on peripheries
@inline function Syᶜᶠᶠ(i, j, k, grid, b, C) 
    by = ℑzᵃᵃᶠ(i, j, k, grid, ∂y_b, b, C) 
    bz = ℑyᵃᶠᵃ(i, j, k, grid, ∂z_b, b, C)
    Sy = ifelse(bz == 0, zero(grid), - by / bz)

    # Impose a boundary condition on immersed peripheries
    inactive = peripheral_node(i, j, k, grid, Center(), Face(), Face())
    
    return ifelse(inactive, zero(grid), Sy)
end

@inline function κ_ϵSxᶠᶜᶠ(i, j, k, grid, clk, sl, κ, b, fields) 
    κ  = κᶠᶜᶠ(i, j, k, grid, issd_coefficient_loc, κ, clk.time, fields) 
    Sx = Sxᶠᶜᶠ(i, j, k, grid, b, fields) 
    ϵ  = tapering_factor(Sx, zero(grid), sl)
    return κ * ϵ * Sx
end

@inline function κ_ϵSyᶜᶠᶠ(i, j, k, grid, clk, sl, κ, b, fields) 
    κ  = κᶜᶠᶠ(i, j, k, grid, issd_coefficient_loc, κ, clk.time, fields) 
    Sy = Syᶜᶠᶠ(i, j, k, grid, b, fields) 
    ϵ  = tapering_factor(zero(grid), Sy, sl)
    return κ * ϵ * Sy
end

@kernel function _advance_eddy_streamfunctions!(Ψx, Ψy, grid, Δt, clock, closure, buoyancy, fields, sl::EddyEvolvingStreamfunction)
    i, j, k = @index(Global, NTuple)
    κ = closure.κ_skew

    @inbounds begin        
        GΨx = (κ_ϵSxᶠᶜᶠ(i, j, k, grid, clock, sl, κ, buoyancy, fields) - Ψx[i, j, k]) / sl.time_scale
        GΨy = (κ_ϵSyᶜᶠᶠ(i, j, k, grid, clock, sl, κ, buoyancy, fields) - Ψy[i, j, k]) / sl.time_scale

        # Advance streamfunctions
        Ψx[i, j, k] += Δt * GΨx 
        Ψy[i, j, k] += Δt * GΨy 
    end
end

@kernel function _advance_eddy_streamfunctions!(Ψx, Ψy, grid, Δt, clock, closure, buoyancy, fields, sl)
    i, j, k = @index(Global, NTuple)
    κ = closure.κ_skew

    @inbounds Ψx[i, j, k] = κ_ϵSxᶠᶜᶠ(i, j, k, grid, clock, sl, κ, buoyancy, fields)
    @inbounds Ψy[i, j, k] = κ_ϵSyᶜᶠᶠ(i, j, k, grid, clock, sl, κ, buoyancy, fields)
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
@inline closure_auxiliary_velocity(::EddyAdvectiveClosure, K, val_tracer_name) = (u = K.u, v = K.v, w = K.w)

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
# Assumption: there is only one EAC closure in the tuple of closures.
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
