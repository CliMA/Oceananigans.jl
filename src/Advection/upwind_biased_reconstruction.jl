#####
##### Upwind-biased 3rd-order advection scheme
#####

"""
    struct UpwindBiased <: AbstractUpwindBiasedAdvectionScheme{3}

Upwind-biased reconstruction scheme.
"""
struct UpwindBiased{N, FT, XT, YT, ZT, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT} 
    "Coefficient for Upwind reconstruction on stretched ``x``-faces" 
    coeff_xᶠᵃᵃ::XT
    "Coefficient for Upwind reconstruction on stretched ``x``-centers"
    coeff_xᶜᵃᵃ::XT
    "Coefficient for Upwind reconstruction on stretched ``y``-faces"
    coeff_yᵃᶠᵃ::YT
    "Coefficient for Upwind reconstruction on stretched ``y``-centers"
    coeff_yᵃᶜᵃ::YT
    "Coefficient for Upwind reconstruction on stretched ``z``-faces"
    coeff_zᵃᵃᶠ::ZT
    "Coefficient for Upwind reconstruction on stretched ``z``-centers"
    coeff_zᵃᵃᶜ::ZT
    
    "Reconstruction scheme used near boundaries"
    buffer_scheme :: CA
    "Reconstruction scheme used for symmetric interpolation"
    advecting_velocity_scheme :: SI

    function UpwindBiased{N, FT}(coeff_xᶠᵃᵃ::XT, coeff_xᶜᵃᵃ::XT,
                                 coeff_yᵃᶠᵃ::YT, coeff_yᵃᶜᵃ::YT, 
                                 coeff_zᵃᵃᶠ::ZT, coeff_zᵃᵃᶜ::ZT,
                                 buffer_scheme::CA, advecting_velocity_scheme::SI) where {N, FT, XT, YT, ZT, CA, SI}

        return new{N, FT, XT, YT, ZT, CA, SI}(coeff_xᶠᵃᵃ, coeff_xᶜᵃᵃ, 
                                              coeff_yᵃᶠᵃ, coeff_yᵃᶜᵃ, 
                                              coeff_zᵃᵃᶠ, coeff_zᵃᵃᶜ,
                                              buffer_scheme, advecting_velocity_scheme)
    end
end

function UpwindBiased(FT::DataType = Float64; grid = nothing, order = 3) 

    if !(grid isa Nothing) 
        FT = eltype(grid)
    end

    mod(order, 2) == 0 && throw(ArgumentError("UpwindBiased reconstruction scheme is defined only for odd orders"))

    N  = Int((order + 1) ÷ 2)

    if N > 1
        coefficients = Tuple(nothing for i in 1:6)
        # Stretched coefficient seem to be more unstable that constant spacing ones for 
        # linear (non-WENO) upwind reconstruction. We keep constant coefficients for the moment
        # Some tests are needed to verify why this is the case (and if it is expected)
        # coefficients = compute_reconstruction_coefficients(grid, FT, :Upwind; order)
        advecting_velocity_scheme = Centered(FT; grid, order = order - 1)
        buffer_scheme  = UpwindBiased(FT; grid, order = order - 2)
    else
        coefficients     = Tuple(nothing for i in 1:6)
        advecting_velocity_scheme = Centered(FT; grid, order = 2)
        buffer_scheme  = nothing
    end

    return UpwindBiased{N, FT}(coefficients..., buffer_scheme, advecting_velocity_scheme)
end

Base.summary(a::UpwindBiased{N}) where N = string("UpwindBiased(order=", 2N-1, ")")

Base.show(io::IO, a::UpwindBiased{N, FT, XT, YT, ZT}) where {N, FT, XT, YT, ZT} =
    print(io, summary(a), " \n",
              " Boundary scheme: ", "\n",
              "    └── ", summary(a.buffer_scheme) , "\n",
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
                        Adapt.adapt(to, scheme.buffer_scheme),
                        Adapt.adapt(to, scheme.advecting_velocity_scheme))

on_architecture(to, scheme::UpwindBiased{N, FT}) where {N, FT} =
    UpwindBiased{N, FT}(on_architecture(to, scheme.coeff_xᶠᵃᵃ), on_architecture(to, scheme.coeff_xᶜᵃᵃ),
                        on_architecture(to, scheme.coeff_yᵃᶠᵃ), on_architecture(to, scheme.coeff_yᵃᶜᵃ),
                        on_architecture(to, scheme.coeff_zᵃᵃᶠ), on_architecture(to, scheme.coeff_zᵃᵃᶜ),
                        on_architecture(to, scheme.buffer_scheme),
                        on_architecture(to, scheme.advecting_velocity_scheme))

# Useful aliases
UpwindBiased(grid, FT::DataType=Float64; kwargs...) = UpwindBiased(FT; grid, kwargs...)

const AUAS = AbstractUpwindBiasedAdvectionScheme

# symmetric interpolation for UpwindBiased and WENO
@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::AUAS, c, args...) = @inbounds symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, c, args...)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::AUAS, c, args...) = @inbounds symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, c, args...)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::AUAS, c, args...) = @inbounds symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.advecting_velocity_scheme, c, args...)

@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::AUAS, u, args...) = @inbounds symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, u, args...)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::AUAS, v, args...) = @inbounds symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, v, args...)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::AUAS, w, args...) = @inbounds symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme.advecting_velocity_scheme, w, args...)

const UX{N, FT} = UpwindBiased{N, FT, <:Nothing} where {N, FT}
const UY{N, FT} = UpwindBiased{N, FT, <:Any, <:Nothing} where {N, FT}
const UZ{N, FT} = UpwindBiased{N, FT, <:Any, <:Any, <:Nothing} where {N, FT}

# Uniform upwind biased reconstruction
for buffer in advection_buffers, FT in fully_supported_float_types
    @eval begin
        @inline inner_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::UX{$buffer, $FT}, left_bias, ψ, idx, loc, args...) = 
            @inbounds ifelse(left_bias, $(calc_reconstruction_stencil(FT, buffer, :left,  :x, false)), 
                                        $(calc_reconstruction_stencil(FT, buffer, :right, :x, false)))

        @inline inner_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::UX{$buffer, $FT}, left_bias, ψ::Function, idx, loc, args...) = 
            @inbounds ifelse(left_bias, $(calc_reconstruction_stencil(FT, buffer, :left,  :x, true)), 
                                        $(calc_reconstruction_stencil(FT, buffer, :right, :x, true)))
    
        @inline inner_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::UY{$buffer, $FT}, left_bias, ψ, idx, loc, args...) = 
            @inbounds ifelse(left_bias, $(calc_reconstruction_stencil(FT, buffer, :left,  :y, false)), 
                                        $(calc_reconstruction_stencil(FT, buffer, :right, :y, false)))
                                                 
        @inline inner_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::UY{$buffer, $FT}, left_bias, ψ::Function, idx, loc, args...) = 
            @inbounds ifelse(left_bias, $(calc_reconstruction_stencil(FT, buffer, :left,  :y, true)), 
                                        $(calc_reconstruction_stencil(FT, buffer, :right, :y, true)))
    
        @inline inner_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::UZ{$buffer, $FT}, left_bias, ψ, idx, loc, args...) = 
            @inbounds ifelse(left_bias, $(calc_reconstruction_stencil(FT, buffer, :left,  :z, false)), 
                                        $(calc_reconstruction_stencil(FT, buffer, :right, :z, false)))

        @inline inner_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::UZ{$buffer, $FT}, left_bias, ψ::Function, idx, loc, args...) = 
            @inbounds ifelse(left_bias, $(calc_reconstruction_stencil(FT, buffer, :left,  :z, true)), 
                                        $(calc_reconstruction_stencil(FT, buffer, :right, :z, true)))                                          
    end
end

# Stretched upwind reconstruction
for (dir, ξ, val) in zip((:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ), (:x, :y, :z), (1, 2, 3))
    stencil = Symbol(:inner_biased_interpolate_, dir)

    for buffer in advection_buffers
        @eval begin
            @inline $stencil(i, j, k, grid, scheme::UpwindBiased{$buffer, FT}, left_bias, ψ, idx, loc, args...) where FT = 
                @inbounds ifelse(left_bias, sum($(reconstruction_stencil(buffer, :left,  ξ, false)) .* retrieve_coeff(scheme, Val(1), Val($val), idx, loc)),
                                            sum($(reconstruction_stencil(buffer, :right, ξ, false)) .* retrieve_coeff(scheme, Val(2), Val($val), idx, loc)))

            @inline $stencil(i, j, k, grid, scheme::UpwindBiased{$buffer, FT}, left_bias, ψ::Function, idx, loc, args...) where FT = 
                @inbounds ifelse(left_bias, sum($(reconstruction_stencil(buffer, :left,  ξ, true)) .* retrieve_coeff(scheme, Val(1), Val($val), idx, loc)),
                                            sum($(reconstruction_stencil(buffer, :right, ξ, true)) .* retrieve_coeff(scheme, Val(2), Val($val), idx, loc)))
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
