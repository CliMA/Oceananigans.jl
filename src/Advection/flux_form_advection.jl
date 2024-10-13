using Oceananigans.Operators: Vᶜᶜᶜ
using Oceananigans.Fields: ZeroField

struct FluxFormAdvection{N, FT, A, B, C} <: AbstractAdvectionScheme{N, FT}
    x :: A
    y :: B
    z :: C

    FluxFormAdvection{N, FT}(x::A, y::B, z::C) where {N, FT, A, B, C} = new{N, FT, A, B, C}(x, y, z)
end

"""
    function FluxFormAdvection(x, y, z)

Builds a `FluxFormAdvection` type with reconstructions schemes `x`, `y`, and `z` to be applied in
the x, y, and z direction, respectively.
"""
function FluxFormAdvection(x_advection, y_advection, z_advection)
    Hx = required_halo_size_x(x_advection)
    Hy = required_halo_size_y(y_advection)
    Hz = required_halo_size_z(z_advection)

    FT = eltype(x_advection)
    H = max(Hx, Hy, Hz)

    return FluxFormAdvection{H, FT}(x_advection, y_advection, z_advection)
end

Base.show(io::IO, scheme::FluxFormAdvection) = 
    print(io, "FluxFormAdvection with reconstructions: ", " \n",
          "    ├── x: ", summary(scheme.x), "\n",
          "    ├── y: ", summary(scheme.y), "\n",
          "    └── z: ", summary(scheme.z))

@inline required_halo_size_x(scheme::FluxFormAdvection) = required_halo_size_x(scheme.x)
@inline required_halo_size_y(scheme::FluxFormAdvection) = required_halo_size_y(scheme.y)
@inline required_halo_size_z(scheme::FluxFormAdvection) = required_halo_size_z(scheme.z)

Adapt.adapt_structure(to, scheme::FluxFormAdvection{N, FT}) where {N, FT} = 
    FluxFormAdvection{N, FT}(Adapt.adapt(to, scheme.x),
                             Adapt.adapt(to, scheme.y),
                             Adapt.adapt(to, scheme.z))

@inline _advective_tracer_flux_x(i, j, k, grid, advection::FluxFormAdvection, args...) = _advective_tracer_flux_x(i, j, k, grid, advection.x, args...)
@inline _advective_tracer_flux_y(i, j, k, grid, advection::FluxFormAdvection, args...) = _advective_tracer_flux_y(i, j, k, grid, advection.y, args...)
@inline _advective_tracer_flux_z(i, j, k, grid, advection::FluxFormAdvection, args...) = _advective_tracer_flux_z(i, j, k, grid, advection.z, args...)

@inline _advective_momentum_flux_Uu(i, j, k, grid, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Uu(i, j, k, grid, advection.x, args...)
@inline _advective_momentum_flux_Vu(i, j, k, grid, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Vu(i, j, k, grid, advection.y, args...)
@inline _advective_momentum_flux_Wu(i, j, k, grid, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Wu(i, j, k, grid, advection.z, args...)

@inline _advective_momentum_flux_Uv(i, j, k, grid, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Uv(i, j, k, grid, advection.x, args...)
@inline _advective_momentum_flux_Vv(i, j, k, grid, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Vv(i, j, k, grid, advection.y, args...)
@inline _advective_momentum_flux_Wv(i, j, k, grid, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Wv(i, j, k, grid, advection.z, args...)

@inline _advective_momentum_flux_Uw(i, j, k, grid, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Uw(i, j, k, grid, advection.x, args...)
@inline _advective_momentum_flux_Vw(i, j, k, grid, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Vw(i, j, k, grid, advection.y, args...)
@inline _advective_momentum_flux_Ww(i, j, k, grid, advection::FluxFormAdvection, args...) = _advective_momentum_flux_Ww(i, j, k, grid, advection.z, args...)
