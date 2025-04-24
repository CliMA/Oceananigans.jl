using Oceananigans.Fields: VelocityFields, ZeroField
using Oceananigans.Grids: inactive_node, peripheral_node
using Oceananigans.BuoyancyFormulations: ∂x_b, ∂y_b, ∂z_b

# Fallback
compute_eddy_velocities!(diffusivities, closure, model; parameters = :xyz) = nothing
compute_eddy_velocities!(diffusivities, ::NoSkewAdvectionISSD, model; parameters = :xyz) = nothing

function compute_eddy_velocities!(diffusivities, closure::SkewAdvectionISSD, model; parameters = :xyz)
    uₑ = diffusivities.u
    vₑ = diffusivities.v
    wₑ = diffusivities.w

    buoyancy = model.buoyancy
    grid     = model.grid
    clock    = model.clock

    model_fields = fields(model)

    launch!(architecture(grid), grid, parameters, _compute_eddy_velocities!,
            uₑ, vₑ, wₑ, grid, clock, closure, buoyancy, model_fields)

    return nothing
end

# Slope in x-direction at F, C, F locations
@inline function Sxᶠᶜᶠ(i, j, k, grid, clo, b, C)
    bx = ℑzᵃᵃᶠ(i, j, k, grid, ∂x_b, b, C)
    bz = ℑxᶠᵃᵃ(i, j, k, grid, ∂z_b, b, C)

    Sₘ = clo.slope_limiter.max_slope

    # Impose a boundary condition on immersed peripheries
    inactive = peripheral_node(i, j, k, grid, Face(), Center(), Face())
    Sx = ifelse(bz == 0,  zero(grid), - bx / bz)
    ϵ  = min(one(grid), Sₘ^2 / Sx^2)
    Sx = ifelse(inactive, zero(grid), Sx)

    return ϵ * Sx
end

# Slope in y-direction at F, C, F locations
@inline function Syᶜᶠᶠ(i, j, k, grid, clo, b, C)
    by = ℑzᵃᵃᶠ(i, j, k, grid, ∂y_b, b, C)
    bz = ℑyᵃᶠᵃ(i, j, k, grid, ∂z_b, b, C)

    Sₘ = clo.slope_limiter.max_slope

    # Impose a boundary condition on immersed peripheries
    inactive = peripheral_node(i, j, k, grid, Center(), Face(), Face())
    Sy = ifelse(bz == 0,  zero(grid), - by / bz)
    ϵ  = min(one(grid), Sₘ^2 / Sy^2)
    Sy = ifelse(inactive, zero(grid), Sy)

    return ϵ * Sy
end

@inline κ_Sxᶠᶜᶠ(i, j, k, grid, clk, clo, κ, b, fields) = κᶠᶜᶠ(i, j, k, grid, issd_coefficient_loc, κ, clk.time, fields) * Sxᶠᶜᶠ(i, j, k, grid, clo, b, fields)
@inline κ_Syᶜᶠᶠ(i, j, k, grid, clk, clo, κ, b, fields) = κᶜᶠᶠ(i, j, k, grid, issd_coefficient_loc, κ, clk.time, fields) * Syᶜᶠᶠ(i, j, k, grid, clo, b, fields)

@kernel function _compute_eddy_velocities!(uₑ, vₑ, wₑ, grid, clock, closure, buoyancy, fields)
    i, j, k = @index(Global, NTuple)

    closure = getclosure(i, j, closure)
    κ = closure.κ_skew

    @inbounds begin
        uₑ[i, j, k] = - ∂zᶠᶜᶜ(i, j, k, grid, κ_Sxᶠᶜᶠ, clock, closure, κ, buoyancy, fields)
        vₑ[i, j, k] = - ∂zᶜᶠᶜ(i, j, k, grid, κ_Syᶜᶠᶠ, clock, closure, κ, buoyancy, fields)

        wˣ = ∂xᶜᶜᶠ(i, j, k, grid, κ_Sxᶠᶜᶠ, clock, closure, κ, buoyancy, fields)
        wʸ = ∂yᶜᶜᶠ(i, j, k, grid, κ_Syᶜᶠᶠ, clock, closure, κ, buoyancy, fields)

        wₑ[i, j, k] =  wˣ + wʸ
    end
end

# Single closure version
@inline closure_turbulent_velocity(clo, K, val_tracer_name) = nothing
@inline closure_turbulent_velocity(::NoSkewAdvectionISSD, K, val_tracer_name) = nothing
@inline closure_turbulent_velocity(::SkewAdvectionISSD, K, val_tracer_name) = (u = K.u, v = K.v, w = K.w)

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
@inline closure_turbulent_velocity(closures::Tuple{<:Any}, Ks, val_tracer_name) =
            closure_turbulent_velocity(closures[1], Ks[1], val_tracer_name)

@inline closure_turbulent_velocity(closures::Tuple{<:Any, <:Any}, Ks, val_tracer_name) =
    select_velocities(closure_turbulent_velocity(closures[1], Ks[1], val_tracer_name),
                      closure_turbulent_velocity(closures[2], Ks[2], val_tracer_name))

@inline closure_turbulent_velocity(closures::Tuple{<:Any, <:Any, <:Any}, Ks, val_tracer_name) =
    select_velocities(closure_turbulent_velocity(closures[1], Ks[1], val_tracer_name),
                      closure_turbulent_velocity(closures[2], Ks[2], val_tracer_name),
                      closure_turbulent_velocity(closures[3], Ks[3], val_tracer_name))

@inline closure_turbulent_velocity(closures::Tuple{<:Any, <:Any, <:Any, <:Any}, Ks, val_tracer_name) =
    select_velocities(closure_turbulent_velocity(closures[1], Ks[1], val_tracer_name),
                      closure_turbulent_velocity(closures[2], Ks[2], val_tracer_name),
                      closure_turbulent_velocity(closures[3], Ks[3], val_tracer_name),
                      closure_turbulent_velocity(closures[4], Ks[4], val_tracer_name))

@inline closure_turbulent_velocity(closures::Tuple, Ks, val_tracer_name) =
    select_velocities(closure_turbulent_velocity(closures[1], Ks[1], val_tracer_name),
                      closure_turbulent_velocity(closures[2], Ks[2], val_tracer_name),
                      closure_turbulent_velocity(closures[3], Ks[3], val_tracer_name),
                      closure_turbulent_velocity(closures[4:end], Ks[4:end], val_tracer_name))
