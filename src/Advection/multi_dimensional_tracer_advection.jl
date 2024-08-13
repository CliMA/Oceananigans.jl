
struct MultiDimensionalTracerAdvection{N, FT, A} <: AbstractAdvectionScheme{N, FT}
    scheme :: A
    MultiDimensionalTracerAdvection{N, FT}(scheme::A) where {N, FT, A} = new{N, FT, A}(scheme)
end

MultiDimensionalTracerAdvection(; scheme::AbstractAdvectionScheme{N, FT}) where {N, FT} = 
    MultiDimensionalTracerAdvection{N, FT}(scheme)

# Extend interpolate functions for VectorInvariant to allow MultiDimensional reconstruction
for bias in (:_biased, :_symmetric)
    for (dir1, dir2) in zip((:xᶠᵃᵃ, :xᶜᵃᵃ, :yᵃᶠᵃ, :yᵃᶜᵃ), (:y, :y, :x, :x))
        interp_func = Symbol(bias, :_interpolate_, dir1)
        multidim_interp = Symbol(:_multi_dimensional_reconstruction_, dir2)

        @eval begin
            @inline $interp_func(i, j, k, grid, advection::MultiDimensionalTracerAdvection, args...) = 
                        $multidim_interp(i, j, k, grid, advection.scheme, $interp_func, args...)
        end
    end
end

const MTA{A} = MultiDimensionalTracerAdvection{<:Any, <:Any, A} where A

@inline _biased_interpolate_zᵃᵃᶜ(i, j, k, grid, advection::MTA, args...) = _biased_interpolate_zᵃᵃᶜ(i, j, k, grid, advection.scheme, args...)
@inline _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, advection::MTA, args...) = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, advection.scheme, args...)

@inline _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, advection::MTA, args...) = _symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, advection.scheme, args...)
@inline _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, advection::MTA, args...) = _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, advection.scheme, args...)

@inline advective_tracer_flux_x(i, j, k, grid, scheme::MTA{<:CenteredScheme}, U, c) = @inbounds Ax_qᶠᶜᶜ(i, j, k, grid, U) * _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_y(i, j, k, grid, scheme::MTA{<:CenteredScheme}, V, c) = @inbounds Ay_qᶜᶠᶜ(i, j, k, grid, V) * _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, c)
@inline advective_tracer_flux_z(i, j, k, grid, scheme::MTA{<:CenteredScheme}, W, c) = @inbounds Az_qᶜᶜᶠ(i, j, k, grid, W) * _symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, c)

    
@inline function advective_tracer_flux_x(i, j, k, grid, scheme::MTA{<:UpwindScheme}, U, c) 

    @inbounds ũ = U[i, j, k]
    cᴿ = _biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, bias(ũ), c)

    return Axᶠᶜᶜ(i, j, k, grid) * ũ * cᴿ
end

@inline function advective_tracer_flux_y(i, j, k, grid, scheme::MTA{<:UpwindScheme}, V, c)

    @inbounds ṽ = V[i, j, k]
    cᴿ = _biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias(ṽ), c)

    return Ayᶜᶠᶜ(i, j, k, grid) * ṽ * cᴿ
end

@inline function advective_tracer_flux_z(i, j, k, grid, scheme::MTA{<:UpwindScheme}, W, c)

    @inbounds w̃ = W[i, j, k]
    cᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, bias(w̃), c)

    return Azᶜᶜᶠ(i, j, k, grid) * w̃ * cᴿ
end
