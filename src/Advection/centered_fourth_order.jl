using Oceananigans.Grids

#####
##### Centered fourth-order advection scheme
#####

struct CenteredFourthOrder <: AbstractAdvectionScheme end

const C4 = CenteredFourthOrder

const centered_second_order = CenteredSecondOrder()

@inline ℑ³xᶠᵃᵃ(i, j, k, grid, u) = @inbounds u[i, j, k] - δxᶠᵃᵃ(i, j, k, grid, δxᶜᵃᵃ, u) / 6
@inline ℑ³xᶜᵃᵃ(i, j, k, grid, c) = @inbounds c[i, j, k] - δxᶜᵃᵃ(i, j, k, grid, δxᶠᵃᵃ, c) / 6

@inline ℑ³yᵃᶠᵃ(i, j, k, grid, v) = @inbounds v[i, j, k] - δyᵃᶠᵃ(i, j, k, grid, δyᵃᶜᵃ, v) / 6
@inline ℑ³yᵃᶜᵃ(i, j, k, grid, c) = @inbounds c[i, j, k] - δyᵃᶜᵃ(i, j, k, grid, δyᵃᶠᵃ, c) / 6

@inline ℑ³zᵃᵃᶠ(i, j, k, grid, w) = @inbounds w[i, j, k] - δzᵃᵃᶠ(i, j, k, grid, δzᵃᵃᶜ, w) / 6
@inline ℑ³zᵃᵃᶜ(i, j, k, grid, c) = @inbounds c[i, j, k] - δzᵃᵃᶜ(i, j, k, grid, δzᵃᵃᶠ, c) / 6

# Momentum

@inline fourth_order_momentum_flux_uu(i, j, k, grid, u)    = ℑxᶜᵃᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * ℑxᶜᵃᵃ(i, j, k, grid, ℑ³xᶠᵃᵃ, u)
@inline fourth_order_momentum_flux_uv(i, j, k, grid, u, v) = ℑxᶠᵃᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * ℑyᵃᶠᵃ(i, j, k, grid, ℑ³yᵃᶜᵃ, u)
@inline fourth_order_momentum_flux_uw(i, j, k, grid, u, w) = ℑxᶠᵃᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * ℑzᵃᵃᶠ(i, j, k, grid, ℑ³zᵃᵃᶜ, u)

@inline fourth_order_momentum_flux_vu(i, j, k, grid, u, v) = ℑyᵃᶠᵃ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * ℑxᶠᵃᵃ(i, j, k, grid, ℑ³xᶜᵃᵃ, v)
@inline fourth_order_momentum_flux_vv(i, j, k, grid, v)    = ℑyᵃᶜᵃ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * ℑyᵃᶜᵃ(i, j, k, grid, ℑ³yᵃᶠᵃ, v)
@inline fourth_order_momentum_flux_vw(i, j, k, grid, v, w) = ℑyᵃᶠᵃ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * ℑzᵃᵃᶠ(i, j, k, grid, ℑ³zᵃᵃᶜ, v)

@inline fourth_order_momentum_flux_wu(i, j, k, grid, u, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ax_ψᵃᵃᶠ, u) * ℑxᶠᵃᵃ(i, j, k, grid, ℑ³xᶜᵃᵃ, w)
@inline fourth_order_momentum_flux_wv(i, j, k, grid, v, w) = ℑzᵃᵃᶠ(i, j, k, grid, Ay_ψᵃᵃᶠ, v) * ℑyᵃᶠᵃ(i, j, k, grid, ℑ³yᵃᶜᵃ, w)
@inline fourth_order_momentum_flux_ww(i, j, k, grid, w)    = ℑzᵃᵃᶜ(i, j, k, grid, Az_ψᵃᵃᵃ, w) * ℑzᵃᵃᶜ(i, j, k, grid, ℑ³zᵃᵃᶠ, w)

# Periodic directions!

@inline momentum_flux_uu(i, j, k, grid, ::C4, u)    = fourth_order_momentum_flux_uu(i, j, k, grid, u)
@inline momentum_flux_uv(i, j, k, grid, ::C4, u, v) = fourth_order_momentum_flux_uv(i, j, k, grid, u, v)
@inline momentum_flux_uw(i, j, k, grid, ::C4, u, w) = fourth_order_momentum_flux_uw(i, j, k, grid, u, w)

@inline momentum_flux_vu(i, j, k, grid, ::C4, u, v) = fourth_order_momentum_flux_vu(i, j, k, grid, u, v)
@inline momentum_flux_vv(i, j, k, grid, ::C4, v)    = fourth_order_momentum_flux_vv(i, j, k, grid, v)
@inline momentum_flux_vw(i, j, k, grid, ::C4, v, w) = fourth_order_momentum_flux_vw(i, j, k, grid, v, w)

@inline momentum_flux_wu(i, j, k, grid, ::C4, u, w) = fourth_order_momentum_flux_wu(i, j, k, grid, u, w)
@inline momentum_flux_wv(i, j, k, grid, ::C4, v, w) = fourth_order_momentum_flux_wv(i, j, k, grid, v, w)
@inline momentum_flux_ww(i, j, k, grid, ::C4, w)    = fourth_order_momentum_flux_ww(i, j, k, grid, w)

# Bounded directions

@inline function momentum_flux_uu(i, j, k, grid::AbstractGrid{FT, <:Bounded}, ::C4, u) where FT
    if i > 1 && i < grid.Nx
        return fourth_order_momentum_flux_uu(i, j, k, grid, u)
    else
        return momentum_flux_uu(i, j, k, grid, centered_second_order, u)
    end
end

@inline function momentum_flux_uv(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, ::C4, u, v) where {FT, TX}
    if j > 1 && j < grid.Ny
        return fourth_order_momentum_flux_uv(i, j, k, grid, u, v)
    else
        return momentum_flux_uv(i, j, k, grid, centered_second_order, u, v)
    end
end

@inline function momentum_flux_uw(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, ::C4, u, w) where {FT, TX, TY}
    if k > 1 && k < grid.Nz
        return fourth_order_momentum_flux_uw(i, j, k, grid, u, w)
    else
        return momentum_flux_uw(i, j, k, grid, centered_second_order, u, w)
    end
end

@inline function momentum_flux_vu(i, j, k, grid::AbstractGrid{FT, <:Bounded}, ::C4, u, v) where {FT}
    if i > 1 && i < grid.Nx
        return fourth_order_momentum_flux_vu(i, j, k, grid, u, v)
    else
        return momentum_flux_vu(i, j, k, grid, centered_second_order, u, v)
    end
end
    

@inline function momentum_flux_vv(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, ::C4, v) where {FT, TX}
    if j > 1 && j < grid.Ny
        return fourth_order_momentum_flux_vv(i, j, k, grid, v)
    else
        return momentum_flux_vv(i, j, k, grid, centered_second_order, v)
    end
end

@inline function momentum_flux_vw(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, ::C4, v, w) where {FT, TX, TY}
    if k > 1 && k < grid.Nz
        return fourth_order_momentum_flux_vw(i, j, k, grid, v, w)
    else
        return momentum_flux_vw(i, j, k, grid, centered_second_order, v, w)
    end
end

@inline function momentum_flux_wu(i, j, k, grid::AbstractGrid{FT, <:Bounded}, ::C4, u, w) where FT
    if i > 1 && i < grid.Nx
        return fourth_order_momentum_flux_wu(i, j, k, grid, u, w)
    else
        return momentum_flux_wu(i, j, k, grid, centered_second_order, u, w)
    end
end

@inline function momentum_flux_wv(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, ::C4, v, w) where {FT, TX}
    if j > 1 && j < grid.Ny
        return fourth_order_momentum_flux_wv(i, j, k, grid, v, w)
    else
        return momentum_flux_wv(i, j, k, grid, centered_second_order, v, w)
    end
end

@inline function momentum_flux_ww(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, ::C4, w) where {FT, TX, TY}
    if k > 1 && i < grid.Nz
        return fourth_order_momentum_flux_ww(i, j, k, grid, w)
    else
        return momentum_flux_ww(i, j, k, grid, centered_second_order, w)
    end
end


# Tracers

@inline fourth_order_advective_tracer_flux_x(i, j, k, grid, u, c) = Ax_ψᵃᵃᶠ(i, j, k, grid, u) * ℑxᶠᵃᵃ(i, j, k, grid, ℑ³xᶜᵃᵃ, c)
@inline fourth_order_advective_tracer_flux_y(i, j, k, grid, v, c) = Ay_ψᵃᵃᶠ(i, j, k, grid, v) * ℑyᵃᶠᵃ(i, j, k, grid, ℑ³yᵃᶜᵃ, c)
@inline fourth_order_advective_tracer_flux_z(i, j, k, grid, w, c) = Az_ψᵃᵃᵃ(i, j, k, grid, w) * ℑzᵃᵃᶠ(i, j, k, grid, ℑ³zᵃᵃᶜ, c)

@inline advective_tracer_flux_x(i, j, k, grid, ::C4, u, c) = fourth_order_advective_tracer_flux_x(i, j, k, grid, u, c)
@inline advective_tracer_flux_y(i, j, k, grid, ::C4, v, c) = fourth_order_advective_tracer_flux_y(i, j, k, grid, v, c)
@inline advective_tracer_flux_z(i, j, k, grid, ::C4, w, c) = fourth_order_advective_tracer_flux_z(i, j, k, grid, w, c)

@inline function advective_tracer_flux_x(i, j, k, grid::AbstractGrid{FT, <:Bounded}, ::C4, u, c) where FT
    if i > 1 && i < grid.Nx
        return fourth_order_advective_tracer_flux_x(i, j, k, grid, u, c)
    else
        return advective_tracer_flux_x(i, j, k, grid, centered_second_order, u, c)
    end
end

@inline function advective_tracer_flux_y(i, j, k, grid::AbstractGrid{FT, TX, <:Bounded}, ::C4, v, c) where {FT, TX}
    if j > 1 && i < grid.Ny
        return fourth_order_advective_tracer_flux_y(i, j, k, grid, v, c)
    else
        return advective_tracer_flux_y(i, j, k, grid, centered_second_order, v, c)
    end
end

@inline function advective_tracer_flux_z(i, j, k, grid::AbstractGrid{FT, TX, TY, <:Bounded}, ::C4, w, c) where {FT, TX, TY}
    if k > 1 && i < grid.Nz
        return fourth_order_advective_tracer_flux_z(i, j, k, grid, w, c)
    else
        return advective_tracer_flux_z(i, j, k, grid, centered_second_order, w, c)
    end
end
