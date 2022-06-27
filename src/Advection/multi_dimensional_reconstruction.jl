#####
##### Multi dimensional advection reconstruction
##### following the implementation in "High–Order WENO Finite Volume Methods on Cartesian Grids with Adaptive Mesh Refinement", P. Buchmueller
##### 

struct MultiDimensionalScheme{N, FT, A1} <: AbstractMultiDimensionalAdvectionScheme{N, FT, A1}

    "1D reconstruction"
    scheme_1d :: A1

    function MultiDimensionalScheme{N, FT}(scheme_1d::A1) where {N, FT, A1}
            return new{N, FT, A1}(scheme_1d)
    end
end

function MultiDimensionalScheme(scheme_1d::AbstractAdvectionScheme{N, FT}; order = 4) where {N, FT}
    NT = Int(order ÷ 2)
    return MultiDimensionalScheme{NT, FT}(scheme_1d)
end

Base.summary(a::MultiDimensionalScheme{N}) where N = string("N-dimensional reconstruction scheme order ", N*2)

Base.show(io::IO, a::MultiDimensionalScheme{N, FT}) where {N, FT} =
    print(io, summary(a), " \n",
              " One dimensional scheme: ", "\n",
              "    └── ", summary(a.scheme_1d))

Adapt.adapt_structure(to, scheme::MultiDimensionalScheme{N, FT}) where {N, FT} =
            MultiDimensionalScheme{N, FT}(Adapt.adapt(to, scheme.scheme_1d))

# Coefficients for center to center reconstruction
const coeff4_multi_Q = (-1/24,  2/24, -1/24)
const coeff4_multi_F = ( 1/24, -2/24,  1/24)

const coeff6_multi_Q = ( 3/640,  -29/480,   107/960, -29/480,   3/640)
const coeff6_multi_F = (-17/5760, 77/1440, -97/960,   77/1440, -17/5760)

const MDS{N, FT} = MultiDimensionalScheme{N, FT} where {N, FT}

# Defining the reconstruction operators
for side in (:symmetric, :left_biased, :right_biased), loc in (:ᶠ, :ᶜ)
    interpolate_x = Symbol(:_, side, :_interpolate_x, loc, :ᵃᵃ)
    interpolate_y = Symbol(:_, side, :_interpolate_yᵃ, loc, :ᵃ)
    interpolate_z = Symbol(:_, side, :_interpolate_zᵃᵃ, loc)
    for buffer in (2, 3)
        coeff = Symbol(:coeff, buffer*2, :_multi_Q)
        @eval begin 
            @inline $interpolate_x(i, j, k, grid, scheme::MDS{$buffer}, ψ, args...) = 
                _multi_dimensional_interpolate_y(i, j, k, grid, scheme, $coeff, $interpolate_x, scheme.scheme_1d, ψ, args...)
            @inline $interpolate_y(i, j, k, grid, scheme::MDS{$buffer}, ψ, args...) = 
                _multi_dimensional_interpolate_x(i, j, k, grid, scheme, $coeff, $interpolate_y, scheme.scheme_1d, ψ, args...)
            @inline $interpolate_z(i, j, k, grid, scheme::MDS{$buffer}, ψ, args...) = 
                    $interpolate_z(i, j, k, grid, scheme.scheme_1d, ψ, args...)
        end
    end
end

@inline mds_stencil_x(i, j, k, grid, scheme::MDS{2}, f, args...) = (f(i-1, j, k, grid, args...), f(i, j, k, grid, args...),   f(i+1, j, k, grid, args...))
@inline mds_stencil_y(i, j, k, grid, scheme::MDS{2}, f, args...) = (f(i, j-1, k, grid, args...), f(i, j, k, grid, args...),   f(i, j+1, k, grid, args...))
@inline mds_stencil_x(i, j, k, grid, scheme::MDS{3}, f, args...) = (f(i-2, j, k, grid, args...), f(i-1, j, k, grid, args...), f(i, j, k, grid, args...), f(i+1, j, k, grid, args...), f(i+2, j, k, grid, args...))
@inline mds_stencil_y(i, j, k, grid, scheme::MDS{3}, f, args...) = (f(i, j-2, k, grid, args...), f(i, j-1, k, grid, args...), f(i, j, k, grid, args...), f(i, j+1, k, grid, args...), f(i, j+2, k, grid, args...))

const NotMultiDimensionalScheme = MultiDimensionalScheme{<:Any, <:Any, <:AbstractAdvectionScheme{1}}

for (dir, ξ) in enumerate((:x, :y))
    md_interpolate = Symbol(:multi_dimensional_interpolate_, ξ)
    mds_stencil    = Symbol(:mds_stencil_, ξ)

    # Fallback if the 1D scheme is second order
    @eval @inline $md_interpolate(i, j, k, grid, scheme::NotMultiDimensionalScheme, coeff, func, scheme_1d, args...) = func(i, j, k, grid, scheme.scheme_1d, args...)

    for buffer in (2, 3)
        @eval begin
            @inline function $md_interpolate(i, j, k, grid, scheme::MDS{$buffer}, coeff, func, scheme_1d, args...)
                # compute ψ(i, j, k, grid, scheme_1d, args...) at -(buffer-1):(buffer-1)
                ψₜ = $mds_stencil(i, j, k, grid, scheme, func, scheme_1d, args...)
                flux_diff = sum(coeff .* ψₜ)
                return ψₜ[$buffer] + flux_diff
            end
        end
    end
end
