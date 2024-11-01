function materialize_timestepper(name::Symbol, args...) 
    fullname = Symbol(name, :Scheme)
    TS = getglobal(@__MODULE__, fullname)
    return materialize_timestepper(TS, args...)
end

struct ForwardBackwardScheme end

materialize_timestepper(::ForwardBackwardScheme, grid, args...) = ForwardBackwardScheme()

struct AdamsBashforth3Scheme{CC, FC, CF, FT}
    ηᵐ   :: CC
    ηᵐ⁻¹ :: CC
    ηᵐ⁻² :: CC
    Uᵐ⁻¹ :: FC
    Uᵐ⁻² :: FC
    Vᵐ⁻¹ :: CF
    Vᵐ⁻² :: CF
       β :: FT
       α :: FT 
       θ :: FT 
       γ :: FT 
       δ :: FT 
       ϵ :: FT 
       μ :: FT 
end

AdamsBashforth3Scheme(; β = 0.281105, α = 1.5 + β, θ = - 0.5 - 2β, γ = 0.088, δ = 0.614, ϵ = 0.013, μ = 1 - δ - γ - ϵ) = 
        AdamsBashforth3Scheme(nothing, nothing, nothing, nothing, nothing, nothing, nothing, β, α, θ, γ, δ, ϵ, μ)

function materialize_timestepper(t::AdamsBashforth3Scheme, grid, free_surface, velocities, u_bc, v_bc)
    ηᵐ   = free_surface_displacement_field(velocities, free_surface, grid)
    ηᵐ⁻¹ = free_surface_displacement_field(velocities, free_surface, grid)
    ηᵐ⁻² = free_surface_displacement_field(velocities, free_surface, grid)

    Uᵐ⁻¹ = Field{Face, Center, Nothing}(grid; boundary_conditions = u_bc)
    Uᵐ⁻² = Field{Face, Center, Nothing}(grid; boundary_conditions = u_bc)
    Vᵐ⁻¹ = Field{Center, Face, Nothing}(grid; boundary_conditions = v_bc)
    Vᵐ⁻² = Field{Center, Face, Nothing}(grid; boundary_conditions = v_bc)

    FT = eltype(grid)

    β = convert(FT, t.β)
    α = convert(FT, t.α)
    θ = convert(FT, t.θ)
    γ = convert(FT, t.γ)
    δ = convert(FT, t.δ)
    ϵ = convert(FT, t.ϵ)
    μ = convert(FT, t.μ)
    return AdamsBashforth3Scheme(ηᵐ, ηᵐ⁻¹, ηᵐ⁻², Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², β, α, θ, γ, δ, ϵ, μ)
end

# The functions `η★` `U★` and `V★` represent the value of free surface, barotropic zonal and meridional velocity at time step m+1/2
@inline U★(i, j, k, grid, t::ForwardBackwardScheme, Uᵐ) = @inbounds Uᵐ[i, j, k]
@inline U★(i, j, k, grid, t::AdamsBashforth3Scheme, Uᵐ) = @inbounds t.α * Uᵐ[i, j, k] + t.θ * t.Uᵐ⁻¹[i, j, k] + t.β * t.Uᵐ⁻²[i, j, k]

@inline η★(i, j, k, grid, t::ForwardBackwardScheme, ηᵐ⁺¹) = @inbounds ηᵐ⁺¹[i, j, k]
@inline η★(i, j, k, grid, t::AdamsBashforth3Scheme, ηᵐ⁺¹) = @inbounds t.δ * ηᵐ⁺¹[i, j, k] + t.μ * t.ηᵐ[i, j, k] + t.γ * t.ηᵐ⁻¹[i, j, k] + t.ϵ * t.ηᵐ⁻²[i, j, k]

@inline advance_previous_velocities!(::ForwardBackwardScheme, i, j, k, U) = nothing

@inline function advance_previous_velocities!(t::AdamsBashforth3Scheme, i, j, k, U)
    @inbounds t.Uᵐ⁻²[i, j, k] = t.Uᵐ⁻¹[i, j, k]
    @inbounds t.Uᵐ⁻¹[i, j, k] =      U[i, j, k]

    return nothing
end

@inline advance_previous_free_surface!(::ForwardBackwardScheme, i, j, k, η) = nothing

@inline function advance_previous_free_surface!(t::AdamsBashforth3Scheme, i, j, k, η)
    @inbounds t.ηᵐ⁻²[i, j, k] = t.ηᵐ⁻¹[i, j, k]
    @inbounds t.ηᵐ⁻¹[i, j, k] =   t.ηᵐ[i, j, k]
    @inbounds   t.ηᵐ[i, j, k] =      η[i, j, k]

    return nothing
end
