using Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶠᶜᶠ, Δzᶜᶠᶠ, Az_qᶜᶜᶠ, Azᶜᶜᶠ

"""
    AdaptiveImplicitVerticalAdvection(; explicit_scheme = Centered(),
                                       cfl = 0.9)

An adaptively implicit vertical advection scheme following Shchepetkin (2015) / CROCO.

Splits vertical advection into explicit and implicit parts based on the local
vertical Courant number `α = |w| Δt / Δz`. When `α ≤ cfl`, advection is fully
explicit using `explicit_scheme`. When `α > cfl`, the vertical velocity is decomposed
as `w = wᵉ + wⁱ` where `wᵉ` is CFL-limited and `wⁱ` is treated with implicit
first-order upwind in the existing tridiagonal solver.

The splitting function is:

    f(α, cfl) = max(1, α / cfl)
    wᵉ = w / f     (explicit, CFL-limited)
    wⁱ = w - wᵉ    (implicit, first-order upwind)

Keyword Arguments
=================

- `explicit_scheme`: The advection scheme for the explicit vertical fluxes (default: `Centered()`).
- `cfl`: Maximum vertical CFL for the explicit part (default: `0.9`).
"""
struct AdaptiveImplicitVerticalAdvection{S, FT, R} <: AbstractAdvectionScheme{1, FT}
    explicit_scheme :: S
    cfl :: FT
    Δt  :: R # Ref{FT} storing the current time step, updated before each tendency computation
end

function AdaptiveImplicitVerticalAdvection(FT::DataType = Float64;
                                           explicit_scheme = Centered(FT),
                                           cfl = 0.1)
    cfl = convert(FT, cfl)
    Δt  = Ref(zero(FT))
    return AdaptiveImplicitVerticalAdvection(explicit_scheme, cfl, Δt)
end

@inline required_halo_size_x(scheme::AdaptiveImplicitVerticalAdvection) = required_halo_size_x(scheme.explicit_scheme)
@inline required_halo_size_y(scheme::AdaptiveImplicitVerticalAdvection) = required_halo_size_y(scheme.explicit_scheme)
@inline required_halo_size_z(scheme::AdaptiveImplicitVerticalAdvection) = required_halo_size_z(scheme.explicit_scheme)

Adapt.adapt_structure(to, a::AdaptiveImplicitVerticalAdvection{S, FT}) where {S, FT} =
    AdaptiveImplicitVerticalAdvection(Adapt.adapt(to, a.explicit_scheme), a.cfl, a.Δt)

Base.summary(a::AdaptiveImplicitVerticalAdvection) =
    string("AdaptiveImplicitVerticalAdvection(cfl=$(a.cfl), explicit_scheme=$(summary(a.explicit_scheme)))")

Base.show(io::IO, a::AdaptiveImplicitVerticalAdvection) =
    print(io, "AdaptiveImplicitVerticalAdvection:", "\n",
              "├── explicit_scheme: ", summary(a.explicit_scheme), "\n",
              "└── cfl: ", a.cfl)

#####
##### Explicit velocity scaling
#####
##### The explicit vertical velocity is wᵉ = w / f(α, cfl) where
##### f = max(1, α / cfl) and α = |w| * Δt / Δz.
##### This ensures the explicit CFL is always ≤ cfl.
#####

# Scale factor: min(1, cfl * Δz / (|w| * Δt))
# When |w| * Δt / Δz ≤ cfl: scale = 1 (fully explicit)
# When |w| * Δt / Δz > cfl: scale = cfl * Δz / (|w| * Δt) < 1
@inline function explicit_velocity_scale(w, Δz, Δt, cfl)
    α = abs(w) * Δt / Δz
    return ifelse(α > cfl, cfl / α, one(α))
end

#####
##### Flux dispatch
#####
##### Horizontal fluxes pass through to the explicit_scheme unchanged.
##### Vertical tracer flux uses the CFL-scaled velocity wᵉ.
##### Vertical momentum fluxes also pass through to the explicit scheme
##### (implicit treatment is only for tracers and horizontal velocities).
#####

# Horizontal fluxes: pass through
@inline _advective_tracer_flux_x(i, j, k, grid, a::AdaptiveImplicitVerticalAdvection, U, c) = _advective_tracer_flux_x(i, j, k, grid, a.explicit_scheme, U, c)
@inline _advective_tracer_flux_y(i, j, k, grid, a::AdaptiveImplicitVerticalAdvection, V, c) = _advective_tracer_flux_y(i, j, k, grid, a.explicit_scheme, V, c)

# Vertical tracer flux: scale w → wᵉ, then use explicit scheme
# We compute the flux as: Az * wᵉ * interpolated(c)
# The scaling factor is applied to the velocity before the explicit scheme acts.
# For centered schemes: flux = Az * w * interp(c), so we scale: flux_explicit = scale * flux_full
# For upwind schemes: flux = Az * w * interp(c, bias(w)), bias doesn't change since sign(wᵉ) = sign(w)
# So in both cases, scaling the flux by the explicit_velocity_scale is correct.
@inline function _advective_tracer_flux_z(i, j, k, grid, a::AdaptiveImplicitVerticalAdvection, W, c)
    Δt = a.Δt[]
    @inbounds w = W[i, j, k]
    Δz = Δzᶜᶜᶠ(i, j, k, grid)
    s = explicit_velocity_scale(w, Δz, Δt, a.cfl)
    return s * _advective_tracer_flux_z(i, j, k, grid, a.explicit_scheme, W, c)
end

# Horizontal momentum fluxes: pass through to explicit_scheme unchanged
@inline _advective_momentum_flux_Uu(i, j, k, grid, a::AdaptiveImplicitVerticalAdvection, U, u) = _advective_momentum_flux_Uu(i, j, k, grid, a.explicit_scheme, U, u)
@inline _advective_momentum_flux_Vu(i, j, k, grid, a::AdaptiveImplicitVerticalAdvection, V, u) = _advective_momentum_flux_Vu(i, j, k, grid, a.explicit_scheme, V, u)
@inline _advective_momentum_flux_Uv(i, j, k, grid, a::AdaptiveImplicitVerticalAdvection, U, v) = _advective_momentum_flux_Uv(i, j, k, grid, a.explicit_scheme, U, v)
@inline _advective_momentum_flux_Vv(i, j, k, grid, a::AdaptiveImplicitVerticalAdvection, V, v) = _advective_momentum_flux_Vv(i, j, k, grid, a.explicit_scheme, V, v)

# Vertical advection of w: pass through (w is at Face in z, not treated implicitly here)
@inline _advective_momentum_flux_Uw(i, j, k, grid, a::AdaptiveImplicitVerticalAdvection, U, w) = _advective_momentum_flux_Uw(i, j, k, grid, a.explicit_scheme, U, w)
@inline _advective_momentum_flux_Vw(i, j, k, grid, a::AdaptiveImplicitVerticalAdvection, V, w) = _advective_momentum_flux_Vw(i, j, k, grid, a.explicit_scheme, V, w)
@inline _advective_momentum_flux_Ww(i, j, k, grid, a::AdaptiveImplicitVerticalAdvection, W, w) = _advective_momentum_flux_Ww(i, j, k, grid, a.explicit_scheme, W, w)

# Vertical advection of horizontal momentum: scale by explicit_velocity_scale
# Wu flux is at (Face, Center, Face); use Δzᶠᶜᶠ for the local CFL
@inline function _advective_momentum_flux_Wu(i, j, k, grid, a::AdaptiveImplicitVerticalAdvection, W, u)
    Δt = a.Δt[]
    @inbounds w = W[i, j, k]
    Δz = Δzᶠᶜᶠ(i, j, k, grid)
    s = explicit_velocity_scale(w, Δz, Δt, a.cfl)
    return s * _advective_momentum_flux_Wu(i, j, k, grid, a.explicit_scheme, W, u)
end

# Wv flux is at (Center, Face, Face); use Δzᶜᶠᶠ for the local CFL
@inline function _advective_momentum_flux_Wv(i, j, k, grid, a::AdaptiveImplicitVerticalAdvection, W, v)
    Δt = a.Δt[]
    @inbounds w = W[i, j, k]
    Δz = Δzᶜᶠᶠ(i, j, k, grid)
    s = explicit_velocity_scale(w, Δz, Δt, a.cfl)
    return s * _advective_momentum_flux_Wv(i, j, k, grid, a.explicit_scheme, W, v)
end

#####
##### Utility functions
#####

needs_implicit_solver(advection) = false
needs_implicit_solver(::AdaptiveImplicitVerticalAdvection) = true
needs_implicit_solver(a::FluxFormAdvection) = needs_implicit_solver(a.z)
needs_implicit_solver(a::NamedTuple) = any(needs_implicit_solver, values(a))

"""
    update_advection_timestep!(advection, Δt)

Update the time step stored in the advection scheme. Called before tendency computation.
"""
update_advection_timestep!(advection, Δt) = nothing
update_advection_timestep!(a::AdaptiveImplicitVerticalAdvection, Δt) = (a.Δt[] = Δt; nothing)
update_advection_timestep!(a::FluxFormAdvection, Δt) = update_advection_timestep!(a.z, Δt)

function update_advection_timestep!(a::NamedTuple, Δt)
    for scheme in values(a)
        update_advection_timestep!(scheme, Δt)
    end
    return nothing
end
