#####
##### Upwind-biased 3rd-order advection scheme
#####

"""
    struct UpwindBiasedFifthOrder <: AbstractUpwindBiasedAdvectionScheme{3}

Upwind-biased fifth-order advection scheme.
"""
struct UpwindBiased{N, FT, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT} 
    "Reconstruction scheme used near boundaries"
    near_boundary_scheme :: CA

    "Reconstruction scheme used for symmetric interpolation"
    advecting_velocity_scheme :: SI

    function UpwindBiased{N, FT}(near_boundary_scheme::CA, advecting_velocity_scheme::SI) where {N, FT, CA, SI}
        return new{N, FT, CA, SI}(near_boundary_scheme, advecting_velocity_scheme)
    end
end

"""
    UpwindBiased([FT=Float64], grid; order=3)

"""
function UpwindBiased(FT=Float64, grid=nothing; order=3)

    mod(order, 2) == 0 && throw(ArgumentError("`order` must be `odd` for UpwindBiased reconstruction!"))
    order > 0 && throw(ArgumentError("`order` must be positive for UpwindBiased reconstruction!"))

    !isnothing(grid) && (FT = eltype(grid))

    # N is...
    N = Int((order + 1) / 2)
    advecting_velocity_scheme = Centered(FT; grid, order=max(2, order-1))

    if order == 1
        near_boundary_scheme = UpwindBiased(FT; grid, order=order-2)
    else
        near_boundary_scheme = nothing
    end

    return UpwindBiased{N, FT}(near_boundary_scheme, advecting_velocity_scheme)
end

# Default to eltype(grid)
UpwindBiased(grid::AbstractGrid; kw...) = UpwindBiased(eltype(grid), grid; kw...)

# Aliases
UpwindBiasedFirstOrder(args...) = UpwindBiased(args...; order = 1)
UpwindBiasedThirdOrder(args...) = UpwindBiased(args...; order = 3)
UpwindBiasedFifthOrder(args...) = UpwindBiased(args...; order = 5)

Base.summary(a::UpwindBiased{N}) where N = string("UpwindBiased reconstruction with order ", N*2-1)

Base.show(io::IO, a::UpwindBiased{N, FT, XT, YT, ZT}) where {N, FT, XT, YT, ZT} =
    print(io, summary(a), " \n",
              " Boundary scheme: ", "\n",
              "    └── ", summary(a.near_boundary_scheme) , "\n",
              " Symmetric scheme: ", "\n",
              "    └── ", summary(a.advecting_velocity_scheme), "\n",
              " Directions:", "\n",
              "    ├── X $(XT == Nothing ? "regular" : "stretched") \n",
              "    ├── Y $(YT == Nothing ? "regular" : "stretched") \n",
              "    └── Z $(ZT == Nothing ? "regular" : "stretched")" )

Adapt.adapt_structure(to, scheme::UpwindBiased{N, FT}) where {N, FT} =
    UpwindBiased{N, FT}(Adapt.adapt(to, scheme.coeff_xᶠᵃᵃ), Adapt.adapt(to, scheme.coeff_xᶜᵃᵃ),
                        Adapt.adapt(to, scheme.coeff_yᵃᶠᵃ), Adapt.adapt(to, scheme.coeff_yᵃᶜᵃ),
                        Adapt.adapt(to, scheme.coeff_zᵃᵃᶠ), Adapt.adapt(to, scheme.coeff_zᵃᵃᶜ),
                        Adapt.adapt(to, scheme.near_boundary_scheme),
                        Adapt.adapt(to, scheme.advecting_velocity_scheme))

const AUBAS = AbstractUpwindBiasedAdvectionScheme

# Symmetric interpolation of advection velocity fields
@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::AUBAS, c) = @inbounds symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, c)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::AUBAS, c) = @inbounds symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, c)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::AUBAS, c) = @inbounds symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.advecting_velocity_scheme, c)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::AUBAS, u) = @inbounds symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, u)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::AUBAS, v) = @inbounds symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, v)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::AUBAS, w) = @inbounds symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme.advecting_velocity_scheme, w)

# uniform upwind biased reconstruction
for side in (:left, :right)
    stencil_x = Symbol(:inner_, side, :_biased_interpolate_xᶠᵃᵃ)
    stencil_y = Symbol(:inner_, side, :_biased_interpolate_yᵃᶠᵃ)
    stencil_z = Symbol(:inner_, side, :_biased_interpolate_zᵃᵃᶠ)

    for buffer in [1, 2, 3, 4, 5, 6]
        @eval begin
            @inline $stencil_x(i, j, k, grid, scheme::UpwindBiased{$buffer, FT, <:Nothing}, ψ, idx, loc, args...)           where FT = @inbounds $(calc_reconstruction_stencil(buffer, side, :x, false))
            @inline $stencil_x(i, j, k, grid, scheme::UpwindBiased{$buffer, FT, <:Nothing}, ψ::Function, idx, loc, args...) where FT = @inbounds $(calc_reconstruction_stencil(buffer, side, :x,  true))
        
            @inline $stencil_y(i, j, k, grid, scheme::UpwindBiased{$buffer, FT, XT, <:Nothing}, ψ, idx, loc, args...)           where {FT, XT} = @inbounds $(calc_reconstruction_stencil(buffer, side, :y, false))
            @inline $stencil_y(i, j, k, grid, scheme::UpwindBiased{$buffer, FT, XT, <:Nothing}, ψ::Function, idx, loc, args...) where {FT, XT} = @inbounds $(calc_reconstruction_stencil(buffer, side, :y,  true))
        
            @inline $stencil_z(i, j, k, grid, scheme::UpwindBiased{$buffer, FT, XT, YT, <:Nothing}, ψ, idx, loc, args...)           where {FT, XT, YT} = @inbounds $(calc_reconstruction_stencil(buffer, side, :z, false))
            @inline $stencil_z(i, j, k, grid, scheme::UpwindBiased{$buffer, FT, XT, YT, <:Nothing}, ψ::Function, idx, loc, args...) where {FT, XT, YT} = @inbounds $(calc_reconstruction_stencil(buffer, side, :z,  true))
    
        end
    end
end

# stretched upwind biased reconstruction
for (sd, side) in enumerate((:left, :right)), (dir, ξ, val) in zip((:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ), (:x, :y, :z), (1, 2, 3))
    stencil = Symbol(:inner_, side, :_biased_interpolate_, dir)

    for buffer in [1, 2, 3, 4, 5, 6]
        @eval begin
            @inline $stencil(i, j, k, grid, scheme::UpwindBiased{$buffer}, ψ, idx, loc, args...)           = @inbounds sum($(reconstruction_stencil(buffer, side, ξ, false)) .* retrieve_coeff(scheme, Val($sd), Val($val), idx, loc))
            @inline $stencil(i, j, k, grid, scheme::UpwindBiased{$buffer}, ψ::Function, idx, loc, args...) = @inbounds sum($(reconstruction_stencil(buffer, side, ξ,  true)) .* retrieve_coeff(scheme, Val($sd), Val($val), idx, loc))
        end
    end
end

# Retrieve precomputed coefficients 
@inline retrieve_coeff(scheme::UpwindBiased, ::Val{1}, ::Val{1}, i, ::Type{Face})   = @inbounds scheme.coeff_xᶠᵃᵃ[1][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::Val{1}, ::Val{1}, i, ::Type{Center}) = @inbounds scheme.coeff_xᶜᵃᵃ[1][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::Val{1}, ::Val{2}, i, ::Type{Face})   = @inbounds scheme.coeff_yᵃᶠᵃ[1][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::Val{1}, ::Val{2}, i, ::Type{Center}) = @inbounds scheme.coeff_yᵃᶜᵃ[1][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::Val{1}, ::Val{3}, i, ::Type{Face})   = @inbounds scheme.coeff_zᵃᵃᶠ[1][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::Val{1}, ::Val{3}, i, ::Type{Center}) = @inbounds scheme.coeff_zᵃᵃᶜ[1][i] 

@inline retrieve_coeff(scheme::UpwindBiased, ::Val{2}, ::Val{1}, i, ::Type{Face})   = @inbounds scheme.coeff_xᶠᵃᵃ[2][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::Val{2}, ::Val{1}, i, ::Type{Center}) = @inbounds scheme.coeff_xᶜᵃᵃ[2][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::Val{2}, ::Val{2}, i, ::Type{Face})   = @inbounds scheme.coeff_yᵃᶠᵃ[2][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::Val{2}, ::Val{2}, i, ::Type{Center}) = @inbounds scheme.coeff_yᵃᶜᵃ[2][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::Val{2}, ::Val{3}, i, ::Type{Face})   = @inbounds scheme.coeff_zᵃᵃᶠ[2][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::Val{2}, ::Val{3}, i, ::Type{Center}) = @inbounds scheme.coeff_zᵃᵃᶜ[2][i] 
