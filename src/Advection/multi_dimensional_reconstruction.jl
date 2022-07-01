#####
##### Multi dimensional advection reconstruction
##### following the implementation in "High–Order WENO Finite Volume Methods on Cartesian Grids with Adaptive Mesh Refinement", P. Buchmueller
##### 

struct MultiDimensionalScheme{N, FT, A1} <: AbstractMultiDimensionalAdvectionScheme{N, FT, A1}

    "1D reconstruction scheme"
    scheme_1d :: A1

    function MultiDimensionalScheme{N, FT}(scheme_1d::A1) where {N, FT, A1}
            return new{N, FT, A1}(scheme_1d)
    end
end

function MultiDimensionalScheme(scheme_1d::AbstractAdvectionScheme{N1, FT}; order = 4) where {N1, FT}
    N = Int(order ÷ 2)
    return MultiDimensionalScheme{N, FT}(scheme_1d)
end

Base.summary(a::MultiDimensionalScheme{N}) where N = string("N-dimensional reconstruction scheme order ", N*2)

Base.show(io::IO, a::MultiDimensionalScheme{N, FT}) where {N, FT} =
    print(io, summary(a), " \n",
              " One dimensional scheme: ", "\n",
              "    └── ", summary(a.scheme_1d))

Adapt.adapt_structure(to, scheme::MultiDimensionalScheme{N, FT}) where {N, FT} =
            MultiDimensionalScheme{N, FT}(Adapt.adapt(to, scheme.scheme_1d))

# Coefficients for average to pointwise reconstruction
const coeff4_multi = (-1/24,  2/24, -1/24)
const coeff6_multi = ( 3/640, -29/480, 107/960, -29/480, 3/640)

const MDS{N, FT} = MultiDimensionalScheme{N, FT} where {N, FT}

# Defining the reconstruction operators
for side in (:symmetric, :left_biased, :right_biased), loc in (:ᶠ, :ᶜ)
    interpolate_x = Symbol(:_, side, :_interpolate_x, loc, :ᵃᵃ)
    interpolate_y = Symbol(:_, side, :_interpolate_yᵃ, loc, :ᵃ)
    interpolate_z = Symbol(:_, side, :_interpolate_zᵃᵃ, loc)
    for buffer in (2, 3)
        coeff = Symbol(:coeff, buffer*2, :_multi)
        @eval begin 
            @inline $interpolate_x(i, j, k, grid, scheme::MDS{$buffer}, ψ, args...) = 
                multi_dimensional_interpolate_yz(i, j, k, grid, scheme, $coeff, $interpolate_x, scheme.scheme_1d, ψ, args...)
            @inline $interpolate_y(i, j, k, grid, scheme::MDS{$buffer}, ψ, args...) = 
                multi_dimensional_interpolate_xz(i, j, k, grid, scheme, $coeff, $interpolate_y, scheme.scheme_1d, ψ, args...)
            @inline $interpolate_z(i, j, k, grid, scheme::MDS{$buffer}, ψ, args...) = 
                multi_dimensional_interpolate_xy(i, j, k, grid, scheme, $coeff, $interpolate_z, scheme.scheme_1d, ψ, args...)
        end
    end
end

@inline mds_stencil_x(i, j, k, grid, scheme::MDS{2}, f, args...) = (f(i-1, j, k, grid, args...), f(i, j, k, grid, args...),   f(i+1, j, k, grid, args...))
@inline mds_stencil_y(i, j, k, grid, scheme::MDS{2}, f, args...) = (f(i, j-1, k, grid, args...), f(i, j, k, grid, args...),   f(i, j+1, k, grid, args...))
@inline mds_stencil_z(i, j, k, grid, scheme::MDS{2}, f, args...) = (f(i, j, k-1, grid, args...), f(i, j, k, grid, args...),   f(i, j, k+1, grid, args...))
@inline mds_stencil_x(i, j, k, grid, scheme::MDS{3}, f, args...) = (f(i-2, j, k, grid, args...), f(i-1, j, k, grid, args...), f(i, j, k, grid, args...), f(i+1, j, k, grid, args...), f(i+2, j, k, grid, args...))
@inline mds_stencil_y(i, j, k, grid, scheme::MDS{3}, f, args...) = (f(i, j-2, k, grid, args...), f(i, j-1, k, grid, args...), f(i, j, k, grid, args...), f(i, j+1, k, grid, args...), f(i, j+2, k, grid, args...))
@inline mds_stencil_z(i, j, k, grid, scheme::MDS{3}, f, args...) = (f(i, j, k-2, grid, args...), f(i, j, k-1, grid, args...), f(i, j, k, grid, args...), f(i, j, k+1, grid, args...), f(i, j, k+2, grid, args...))

const NotMDSSchemes{N} = Union{MultiDimensionalScheme{N, <:Any, <:UpwindBiased{1}},
                               MultiDimensionalScheme{N, <:Any, <:Centered{2}}} where N

for (dir, ξ) in enumerate((:x, :y, :z))
    md_interpolate = Symbol(:multi_dimensional_interpolate_, ξ)
    mds_stencil    = Symbol(:mds_stencil_, ξ)

    for buffer in (2, 3)
        @eval begin
            # If 1D scheme is second order fallback to 1D scheme
            @inline $md_interpolate(i, j, k, grid, ::NotMDSSchemes{$buffer}, args...) = zero(grid)
            
            # Otherwise calculate the correction
            @inline function $md_interpolate(i, j, k, grid, scheme::MDS{$buffer, FT}, coeff, func, args...) where FT               
                # Compute ψ(i, j, k, grid, args...) at -(buffer-1):(buffer-1)
                ψₜ = $mds_stencil(i, j, k, grid, scheme, func, args...)
                flux_diff = sum(FT.(coeff) .* ψₜ)

                # Limit the correction
                if abs(flux_diff) > 0.5*abs(ψₜ[$buffer])
                    flux_diff = zero(grid)
                end

                return flux_diff
            end
        end
    end
end

@inline multi_dimensional_interpolate_yz(i, j, k, grid, scheme, coeff, func, args...) = (
                    func(i, j, k, grid, args...) + 
                    _multi_dimensional_interpolate_y(i, j, k, grid, scheme, coeff, func, args...) +
                    _multi_dimensional_interpolate_z(i, j, k, grid, scheme, coeff, func, args...))

@inline multi_dimensional_interpolate_xz(i, j, k, grid, scheme, coeff, func, args...) = (
                    func(i, j, k, grid, args...) +  
                    _multi_dimensional_interpolate_x(i, j, k, grid, scheme, coeff, func, args...) +
                    _multi_dimensional_interpolate_z(i, j, k, grid, scheme, coeff, func, args...))

@inline multi_dimensional_interpolate_xy(i, j, k, grid, scheme, coeff, func, args...) = (
                     func(i, j, k, grid, args...) + 
                     _multi_dimensional_interpolate_x(i, j, k, grid, scheme, coeff, func, args...) +
                     _multi_dimensional_interpolate_y(i, j, k, grid, scheme, coeff, func, args...))     
    
