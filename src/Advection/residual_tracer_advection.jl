using Oceananigans.Utils: SumOfArrays

struct ResidualTracerAdvection{B, FT, S, U, V, W, K} <: AbstractAdvectionScheme{B, FT} 
    scheme :: S
    u_eddy :: U
    v_eddy :: V
    w_eddy :: W
    diffusivity :: K

    ResidualTracerAdvection{B, FT}(s::S, u::U, v::V, w::W, κ::K) where {B, FT, S, U, V, W, K} = 
        new{B, FT, S, U, V, W, K}(s, u, v, w, κ)
end

Adapt.adapt_structure(to, ra::ResidualTracerAdvection{B, FT}) where {B, FT} = 
    ResidualTracerAdvection{B, FT}(Adapt.adapt(to, ra.scheme), 
                                   Adapt.adapt(to, ra.u_eddy), 
                                   Adapt.adapt(to, ra.v_eddy), 
                                   Adapt.adapt(to, ra.w_eddy), 
                                   Adapt.adapt(to, ra.diffusivity))

function ResidualTracerAdvection(grid, scheme::AbstractAdvectionScheme; diffusivity = 1000.0)

    U = VelocityFields(grid)
    u, v, w = U

    B = max(required_halo_size_x(scheme),
            required_halo_size_y(scheme),
            required_halo_size_z(scheme))

    FT = eltype(grid)

    return ResidualTracerAdvection{B, FT}(scheme, u, v, w, diffusivity)
end

@inline function div_Uc(i, j, k, grid, advection::ResidualTracerAdvection, U, c)
    u_residual = SumOfArrays{2}(U.u, advection.u_eddy)
    v_residual = SumOfArrays{2}(U.v, advection.v_eddy)
    w_residual = SumOfArrays{2}(U.w, advection.w_eddy)

    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, advection, u_residual, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, advection, v_residual, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, advection, w_residual, c))
end

compute_eddy_velocities!(advection, buoyancy, tracers) = nothing

function compute_eddy_velocities!(advection::ResidualTracerAdvection, grid, buoyancy, tracers)
    return nothing
end

@kernel function _compute_eddy_velocities!(advection::ResidualTracerAdvection, grid, buoyancy, tracers)
    i, j, k = @index(Global, NTuple)
    
end