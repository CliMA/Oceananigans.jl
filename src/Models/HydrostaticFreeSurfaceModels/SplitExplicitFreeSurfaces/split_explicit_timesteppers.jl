"""
    struct ForwardBackwardScheme

A timestepping scheme used for substepping in the split-explicit free surface solver.

The equations are evolved as follows:
```math
\\begin{gather}
η^{m+1} = η^m - Δτ (∂_x U^m + ∂_y V^m), \\\\
U^{m+1} = U^m - Δτ (∂_x η^{m+1} - G^U), \\\\
V^{m+1} = V^m - Δτ (∂_y η^{m+1} - G^V).
\\end{gather}
```
"""
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

"""
    AdamsBashforth3Scheme(; β = 0.281105,
                            α = 1.5 + β,
                            θ = -0.5 - 2β,
                            γ = 0.088,
                            δ = 0.614,
                            ϵ = 0.013,
                            μ = 1 - δ - γ - ϵ)

Create an instance of `AdamsBashforth3Scheme` with the specified parameters.
This scheme is used for substepping in the split-explicit free surface solver,
where an AB3 extrapolation is used to evaluate barotropic velocities and
free surface at time-step `m + 1/2`:

The equations are evolved as follows:

```math
\\begin{gather}
η^{m+1} = η^m - Δτ g H (∂_x Ũ + ∂y Ṽ), \\\\
U^{m+1} = U^m - Δτ (∂_x η̃ - G^U), \\\\
V^{m+1} = V^m - Δτ (∂_y η̃ - G^V),
\\end{gather}
```

where `η̃`, `Ũ` and `Ṽ` are the AB3 time-extrapolated values of free surface,
barotropic zonal and meridional velocities, respectively:

```math
\\begin{gather}
Ũ = α U^m + θ U^{m-1} + β U^{m-2}, \\\\
Ṽ = α V^m + θ V^{m-1} + β V^{m-2}, \\\\
η̃ = δ η^{m+1} + μ η^m + γ η^{m-1} + ϵ η^{m-2}.
\\end{gather}
```

The default values for the time-extrapolation coefficients, described by [Shchepetkin2005](@citet),
correspond to the best stability range for the AB3 algorithm.
"""
AdamsBashforth3Scheme(; β = 0.281105, α = 1.5 + β, θ = - 0.5 - 2β, γ = 0.088, δ = 0.614, ϵ = 0.013, μ = 1 - δ - γ - ϵ) =
        AdamsBashforth3Scheme(nothing, nothing, nothing, nothing, nothing, nothing, nothing, β, α, θ, γ, δ, ϵ, μ)

Adapt.adapt_structure(to, t::AdamsBashforth3Scheme) =
    AdamsBashforth3Scheme(
        Adapt.adapt(to, t.ηᵐ  ),
        Adapt.adapt(to, t.ηᵐ⁻¹),
        Adapt.adapt(to, t.ηᵐ⁻²),
        Adapt.adapt(to, t.Uᵐ⁻¹),
        Adapt.adapt(to, t.Uᵐ⁻²),
        Adapt.adapt(to, t.Vᵐ⁻¹),
        Adapt.adapt(to, t.Vᵐ⁻²),
        t.β, t.α, t.θ, t.γ, t.δ, t.ϵ, t.μ)

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

#####
##### Timestepper extrapolations and utils
#####

function materialize_timestepper(name::Symbol, args...)
    fullname = Symbol(name, :Scheme)
    TS = getglobal(@__MODULE__, fullname)
    return materialize_timestepper(TS, args...)
end

initialize_free_surface_timestepper!(::ForwardBackwardScheme, args...) = nothing

function initialize_free_surface_timestepper!(timestepper::AdamsBashforth3Scheme, η, U, V)
    parent(timestepper.Uᵐ⁻¹) .= parent(U)
    parent(timestepper.Vᵐ⁻¹) .= parent(V)

    parent(timestepper.Uᵐ⁻²) .= parent(U)
    parent(timestepper.Vᵐ⁻²) .= parent(V)

    parent(timestepper.ηᵐ)   .= parent(η)
    parent(timestepper.ηᵐ⁻¹) .= parent(η)
    parent(timestepper.ηᵐ⁻²) .= parent(η)

    return nothing
end

# The functions `η★` `U★` and `V★` represent the value of free surface, barotropic zonal and meridional velocity at time step m+1/2
@inline U★(i, j, k, grid,  ::ForwardBackwardScheme, Uᵐ) = @inbounds Uᵐ[i, j, k]
@inline U★(i, j, k, grid, t::AdamsBashforth3Scheme, Uᵐ) = @inbounds t.α * Uᵐ[i, j, k] + t.θ * t.Uᵐ⁻¹[i, j, k] + t.β * t.Uᵐ⁻²[i, j, k]

@inline η★(i, j, k, grid,  ::ForwardBackwardScheme, ηᵐ⁺¹) = @inbounds ηᵐ⁺¹[i, j, k]
@inline η★(i, j, k, grid, t::AdamsBashforth3Scheme, ηᵐ⁺¹) = @inbounds t.δ * ηᵐ⁺¹[i, j, k] + t.μ * t.ηᵐ[i, j, k] + t.γ * t.ηᵐ⁻¹[i, j, k] + t.ϵ * t.ηᵐ⁻²[i, j, k]

@inline cache_previous_velocities!(::ForwardBackwardScheme,   i, j, k, U) = nothing
@inline cache_previous_free_surface!(::ForwardBackwardScheme, i, j, k, η) = nothing

@inline function cache_previous_velocities!(t::AdamsBashforth3Scheme, i, j, k, U)
    @inbounds t.Uᵐ⁻²[i, j, k] = t.Uᵐ⁻¹[i, j, k]
    @inbounds t.Uᵐ⁻¹[i, j, k] =      U[i, j, k]

    return nothing
end

@inline function cache_previous_free_surface!(t::AdamsBashforth3Scheme, i, j, k, η)
    @inbounds t.ηᵐ⁻²[i, j, k] = t.ηᵐ⁻¹[i, j, k]
    @inbounds t.ηᵐ⁻¹[i, j, k] =   t.ηᵐ[i, j, k]
    @inbounds   t.ηᵐ[i, j, k] =      η[i, j, k]

    return nothing
end
