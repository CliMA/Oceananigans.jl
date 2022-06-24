#####
##### Centered advection scheme
#####

"""
    struct Centered <: AbstractCenteredAdvectionScheme{2}

Centered reconstruction scheme.
"""
struct Centered{N, FT, XT, YT, ZT, CA} <: AbstractCenteredAdvectionScheme{N, FT} 
    "coefficient for Centered reconstruction on stretched x-faces" 
    coeff_xᶠᵃᵃ::XT
    "coefficient for Centered reconstruction on stretched x-centers"
    coeff_xᶜᵃᵃ::XT
    "coefficient for Centered reconstruction on stretched y-faces"
    coeff_yᵃᶠᵃ::YT
    "coefficient for Centered reconstruction on stretched y-centers"
    coeff_yᵃᶜᵃ::YT
    "coefficient for Centered reconstruction on stretched z-faces"
    coeff_zᵃᵃᶠ::ZT
    "coefficient for Centered reconstruction on stretched z-centers"
    coeff_zᵃᵃᶜ::ZT

    "advection scheme used near boundaries"
    boundary_scheme :: CA

    function Centered{N, FT}(coeff_xᶠᵃᵃ::XT, coeff_xᶜᵃᵃ::XT,
                             coeff_yᵃᶠᵃ::YT, coeff_yᵃᶜᵃ::YT, 
                             coeff_zᵃᵃᶠ::ZT, coeff_zᵃᵃᶜ::ZT,
                             boundary_scheme::CA) where {N, FT, XT, YT, ZT, CA}

        return new{N, FT, XT, YT, ZT, CA}(coeff_xᶠᵃᵃ, coeff_xᶜᵃᵃ, 
                                          coeff_yᵃᶠᵃ, coeff_yᵃᶜᵃ, 
                                          coeff_zᵃᵃᶠ, coeff_zᵃᵃᶜ,
                                          boundary_scheme)
    end
end

function Centered(FT::DataType = Float64; grid = nothing, order = 2) 

    if !(grid isa Nothing) 
        FT = eltype(grid)
    end

    mod(order, 2) != 0 && throw(ArgumentError("Centered reconstruction scheme is defined only for even orders"))

    N  = Int(order ÷ 2)
    if N > 1 
        coefficients = compute_reconstruction_coefficients(grid, FT, :Centered; order)
        boundary_scheme = Centered(FT; grid, order = order - 2)
    else
        coefficients    = Tuple(nothing for i in 1:6)
        boundary_scheme = nothing
    end
    return Centered{N, FT}(coefficients..., boundary_scheme)
end

Base.summary(a::Centered{N}) where N = string("Centered reconstruction order ", N*2)

Base.show(io::IO, a::Centered{N, FT, XT, YT, ZT}) where {N, FT, XT, YT, ZT} =
    print(io, summary(a), " \n",
              " Boundary scheme: ", "\n",
              "    └── ", summary(a.boundary_scheme), "\n",
              " Directions:", "\n",
              "    ├── X $(XT == Nothing ? "regular" : "stretched") \n",
              "    ├── Y $(YT == Nothing ? "regular" : "stretched") \n",
              "    └── Z $(ZT == Nothing ? "regular" : "stretched")" )


Adapt.adapt_structure(to, scheme::Centered{N, FT}) where {N, FT} =
    Centered{N, FT}(Adapt.adapt(to, scheme.coeff_xᶠᵃᵃ), Adapt.adapt(to, scheme.coeff_xᶜᵃᵃ),
                    Adapt.adapt(to, scheme.coeff_yᵃᶠᵃ), Adapt.adapt(to, scheme.coeff_yᵃᶜᵃ),
                    Adapt.adapt(to, scheme.coeff_zᵃᵃᶠ), Adapt.adapt(to, scheme.coeff_zᵃᵃᶜ),
                    Adapt.adapt(to, scheme.boundary_scheme))
                    
# Useful aliases
Centered(grid, FT::DataType=Float64; kwargs...) = Centered(FT; grid, kwargs...)

CenteredSecondOrder(grid=nothing, FT::DataType=Float64) = Centered(grid, FT; order=2)
CenteredFourthOrder(grid=nothing, FT::DataType=Float64) = Centered(grid, FT; order=4)

# uniform centered reconstruction
for buffer in [1, 2, 3, 4, 5, 6]
    @eval begin
        @inline stretched_symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::Centered{$buffer, FT, <:Nothing}, ψ, idx, loc, args...)           where FT= @inbounds $(calc_reconstruction_stencil(buffer, :symm, :x, false))
        @inline stretched_symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::Centered{$buffer, FT, <:Nothing}, ψ::Function, idx, loc, args...) where FT= @inbounds $(calc_reconstruction_stencil(buffer, :symm, :x,  true))
    
        @inline stretched_symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::Centered{$buffer, FT, XT, <:Nothing}, ψ, idx, loc, args...)           where {FT, XT} = @inbounds $(calc_reconstruction_stencil(buffer, :symm, :y, false))
        @inline stretched_symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::Centered{$buffer, FT, XT, <:Nothing}, ψ::Function, idx, loc, args...) where {FT, XT} = @inbounds $(calc_reconstruction_stencil(buffer, :symm, :y,  true))
    
        @inline stretched_symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::Centered{$buffer, FT, XT, YT, <:Nothing}, ψ, idx, loc, args...)           where {FT, XT, YT} = @inbounds $(calc_reconstruction_stencil(buffer, :symm, :z, false))
        @inline stretched_symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::Centered{$buffer, FT, XT, YT, <:Nothing}, ψ::Function, idx, loc, args...) where {FT, XT, YT} = @inbounds $(calc_reconstruction_stencil(buffer, :symm, :z,  true))
    end
end

# stretched centered reconstruction
for (dir, ξ, val) in zip((:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ), (:x, :y, :z), (1, 2, 3))
    stencil = Symbol(:stretched_symmetric_interpolate_, dir)

    for buffer in [1, 2, 3, 4, 5, 6]
        @eval begin
            @inline $stencil(i, j, k, grid, scheme::Centered{$buffer}, ψ, idx, loc, args...)           = @inbounds sum($(reconstruction_stencil(buffer, :symm, ξ, false)) .* retrieve_coeff(scheme, Val($val), idx, loc))
            @inline $stencil(i, j, k, grid, scheme::Centered{$buffer}, ψ::Function, idx, loc, args...) = @inbounds sum($(reconstruction_stencil(buffer, :symm, ξ,  true)) .* retrieve_coeff(scheme, Val($val), idx, loc))
        end
    end
end

# Retrieve precomputed coefficients 
@inline retrieve_coeff(scheme::Centered, ::Val{1}, i, ::Type{Face})   = @inbounds scheme.coeff_xᶠᵃᵃ[i] 
@inline retrieve_coeff(scheme::Centered, ::Val{1}, i, ::Type{Center}) = @inbounds scheme.coeff_xᶜᵃᵃ[i] 
@inline retrieve_coeff(scheme::Centered, ::Val{2}, i, ::Type{Face})   = @inbounds scheme.coeff_yᵃᶠᵃ[i] 
@inline retrieve_coeff(scheme::Centered, ::Val{2}, i, ::Type{Center}) = @inbounds scheme.coeff_yᵃᶜᵃ[i] 
@inline retrieve_coeff(scheme::Centered, ::Val{3}, i, ::Type{Face})   = @inbounds scheme.coeff_zᵃᵃᶠ[i] 
@inline retrieve_coeff(scheme::Centered, ::Val{3}, i, ::Type{Center}) = @inbounds scheme.coeff_zᵃᵃᶜ[i] 

