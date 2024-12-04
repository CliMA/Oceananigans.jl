

function ∂x_b end
function ∂y_b end
function ∂z_b end

struct ResidualTracerAdvection{B, FT, S, U, V, W, K, M} <: AbstractAdvectionScheme{B, FT} 
    scheme :: S
    u_eddy :: U
    v_eddy :: V
    w_eddy :: W
    diffusivity :: K
    maximum_slope :: M

    ResidualTracerAdvection{B, FT}(s::S, u::U, v::V, w::W, κ::K, m::M) where {B, FT, S, U, V, W, K, M} = 
        new{B, FT, S, U, V, W, K, M}(s, u, v, w, κ, m)
end

Adapt.adapt_structure(to, ra::ResidualTracerAdvection{B, FT}) where {B, FT} = 
    ResidualTracerAdvection{B, FT}(Adapt.adapt(to, ra.scheme), 
                                   Adapt.adapt(to, ra.u_eddy), 
                                   Adapt.adapt(to, ra.v_eddy), 
                                   Adapt.adapt(to, ra.w_eddy), 
                                   Adapt.adapt(to, ra.diffusivity),
                                   Adapt.adapt(to, ra.maximum_slope))

function ResidualTracerAdvection(grid, scheme::AbstractAdvectionScheme; 
                                 diffusivity = 1000.0,
                                 maximum_slope = 10000.0)

    U = VelocityFields(grid)
    u, v, w = U

    B = max(required_halo_size_x(scheme),
            required_halo_size_y(scheme),
            required_halo_size_z(scheme))

    FT = eltype(grid)

    return ResidualTracerAdvection{B, FT}(scheme, u, v, w, diffusivity, maximum_slope)
end

@inline function div_Uc(i, j, k, grid, advection::ResidualTracerAdvection, U, c)
    u_residual = SumOfArrays{2}(U.u, advection.u_eddy)
    v_residual = SumOfArrays{2}(U.v, advection.v_eddy)
    w_residual = SumOfArrays{2}(U.w, advection.w_eddy)

    scheme = advection.scheme

    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, scheme, u_residual, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, scheme, v_residual, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, scheme, w_residual, c))
end

# Fallbacks
compute_eddy_velocities!(advection, grid, buoyancy, tracers; parameters = :xy) = nothing
compute_eddy_velocities!(::NamedTuple, grid, ::Nothing, tracers; parameters = :xy) = nothing
compute_eddy_velocities!(::ResidualTracerAdvection, grid, ::Nothing, tracers; parameters = :xy) = nothing

function compute_eddy_velocities!(advection::NamedTuple, grid, buoyancy, tracers; parameters = :xy) 
    for adv in advection
        compute_eddy_velocities!(adv, grid, buoyancy, tracers; parameters)
    end

    return nothing
end
