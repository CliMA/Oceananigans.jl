#####
##### Multi dimensional advection reconstruction
##### following the implementation in "High–Order WENO Finite Volume Methods on Cartesian Grids with Adaptive Mesh Refinement", P. Buchmueller
##### 

struct MultiDimensionalScheme{N, FT, A1} <: AbstractAdvectionScheme{N, FT}

    "1D reconstruction"
    one_dimensional_scheme :: A1

    function MultiDimensionalScheme{N, FT}(one_dimensional_scheme::A1) where {N, FT, A1}
            return new{N, FT, A1}(one_dimensional_scheme)
    end
end

function MultiDimensionalScheme(one_dimensional_scheme::AbstractAdvectionScheme{N, FT}; order = 4) where {N, FT}
    NT = Int(order ÷ 2)
    return MultiDimensionalScheme{NT, FT}(one_dimensional_scheme)
end

Base.summary(a::MultiDimensionalScheme{N}) where N = string("N-dimensional reconstruction scheme order ", N*2)

Base.show(io::IO, a::MultiDimensionalScheme{N, FT}) where {N, FT} =
    print(io, summary(a), " \n",
              " One dimensional scheme: ", "\n",
              "    └── ", summary(a.one_dimensional_scheme))

Adapt.adapt_structure(to, scheme::MultiDimensionalScheme{N, FT}) where {N, FT} =
            MultiDimensionalScheme{N, FT}(Adapt.adapt(to, scheme.one_dimensional_scheme))

coeff4_multi_Q = (-1/24,  26/24, -1/24)
coeff4_multi_F = ( 1/24,  22/24,  1/24)

coeff6_multi_Q = (3/640,    -29/480,  1067/960,  -29/480,   3/640)
coeff6_multi_F = (-17/5760, 77/1440,   863/960, 77/1440, -17/5760)

const MDS{N, FT} = MultiDimensionalScheme{N, FT} where {N, FT}

# Defining the reconstruction operators
for side in (:symmetric, :left, :right), loc in (:ᶠ, :ᶜ)
    interpolate_x = Symbol(:_, side, :_biased_interpolate_x, loc, :ᵃᵃ)
    interpolate_y = Symbol(:_, side, :_biased_interpolate_yᵃ, loc, :ᵃ)
    interpolate_z = Symbol(:_, side, :_biased_interpolate_zᵃᵃ, loc)
    for buffer in (2, 3)
        coeff = Symbol(:coeff, buffer*2, :_multi_Q)
        @eval begin 
            @inline $interpolate_x(i, j, k, grid, scheme::MDS{$buffer}, ψ, args...) = 
                _multi_dimensional_interpolate_y(i, j, k, grid, scheme, $coeff, $interpolate_x, ψ, args...)
            @inline $interpolate_y(i, j, k, grid, scheme::MDS{$buffer}, ψ, args...) = 
                _multi_dimensional_interpolate_x(i, j, k, grid, scheme, $coeff, $interpolate_y, ψ, args...)
            @inline $interpolate_z(i, j, k, grid, scheme::MDS{$buffer}, ψ, args...) = 
                    $interpolate_z(i, j, k, grid, scheme.one_dimensional_scheme, ψ, args...)
        end
    end
end

for buffer in (2, 3)
    @eval @inline stencil_x(i, j, k, scheme::MDS{$buffer}, ψ, args...) = $(reconstruction_stencil(buffer, :right, :x, true)) 
    @eval @inline stencil_y(i, j, k, scheme::MDS{$buffer}, ψ, args...) = $(reconstruction_stencil(buffer, :right, :y, true)) 
end

for (dir, ξ) in enumerate((:x, :y))
    md_interpolate = Symbol(:multi_dimensional_interpolate_, ξ)
    stencil        = Symbol(:stencil_, ξ)

    @eval begin
        @inline function $md_interpolate(i, j, k, grid, coeff, scheme::MDS, func, ψ, args...)
            ψₜ = stencil(i, j, k, grid, func, scheme.one_dimensional_scheme, ψ, args...)
            return coeff .* ψₜ
        end
    end
end
