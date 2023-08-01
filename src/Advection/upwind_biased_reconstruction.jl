#####
##### Upwind-biased 3rd-order advection scheme
#####

"""
    struct UpwindBiasedFifthOrder <: AbstractUpwindBiasedAdvectionScheme{3}

Upwind-biased fifth-order advection scheme.
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

Base.summary(a::UpwindBiased{N}) where N = string("Upwind Biased reconstruction order ", N*2-1)

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

# Useful aliases
UpwindBiased(grid, FT::DataType=Float64; kwargs...) = UpwindBiased(FT; grid, kwargs...)

UpwindBiasedFirstOrder(grid=nothing, FT::DataType=Float64) = UpwindBiased(grid, FT; order = 1)
UpwindBiasedThirdOrder(grid=nothing, FT::DataType=Float64) = UpwindBiased(grid, FT; order = 3)
UpwindBiasedFifthOrder(grid=nothing, FT::DataType=Float64) = UpwindBiased(grid, FT; order = 5)

const AUAS = AbstractUpwindBiasedAdvectionScheme

# symmetric interpolation for UpwindBiased and WENO
@inline symmetric_interpolate_x(i, j, k, grid, scheme::AUAS, args...) = symmetric_interpolate_x(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline symmetric_interpolate_y(i, j, k, grid, scheme::AUAS, args...) = symmetric_interpolate_y(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline symmetric_interpolate_z(i, j, k, grid, scheme::AUAS, args...) = symmetric_interpolate_z(i, j, k, grid, scheme.advecting_velocity_scheme, args...)

# uniform upwind biased reconstruction
for (Dir, side) in zip((LeftBiasedStencil, RightBiasedStencil), (:left, :right))
    for buffer in advection_buffers
        @eval begin
            @inline function upwind_biased_interpolate_x(i, j, k, grid, dir::Dir, parent_scheme::UpwindBiased{$buffer, FT, <:Nothing}, ψ, idx, loc, args...) where FT 
                scheme = _topologically_conditional_scheme_x(i, j, k, grid, dir, loc, parent_scheme)
                @inbounds $(calc_reconstruction_stencil(buffer, side, :x, false))
            end

            @inline function upwind_biased_interpolate_x(i, j, k, grid, dir::Dir, parent_scheme::UpwindBiased{$buffer, FT, <:Nothing}, ψ::Function, idx, loc, args...) where FT
                scheme = _topologically_conditional_scheme_x(i, j, k, grid, dir, loc, parent_scheme)
                return @inbounds $(calc_reconstruction_stencil(buffer, side, :x,  true))
            end

            @inline function upwind_biased_interpolate_y(i, j, k, grid, dir::Dir, parent_scheme::UpwindBiased{$buffer, FT, XT, <:Nothing}, ψ, idx, loc, args...) where {FT, XT} 
                scheme = _topologically_conditional_scheme_y(i, j, k, grid, dir, loc, parent_scheme)
                return @inbounds $(calc_reconstruction_stencil(buffer, side, :y, false))
            end
            @inline function upwind_biased_interpolate_y(i, j, k, grid, dir::Dir, parent_scheme::UpwindBiased{$buffer, FT, XT, <:Nothing}, ψ::Function, idx, loc, args...) where {FT, XT} 
                scheme = _topologically_conditional_scheme_y(i, j, k, grid, dir, loc, parent_scheme)
                return @inbounds $(calc_reconstruction_stencil(buffer, side, :y,  true))
            end
        
            @inline function upwind_biased_interpolate_z(i, j, k, grid, dir::Dir, parent_scheme::UpwindBiased{$buffer, FT, XT, YT, <:Nothing}, ψ, idx, loc, args...) where {FT, XT, YT}
                scheme = _topologically_conditional_scheme_z(i, j, k, grid, dir, loc, parent_scheme)
                return @inbounds $(calc_reconstruction_stencil(buffer, side, :z, false))
            end
            @inline function upwind_biased_interpolate_z(i, j, k, grid, dir::Dir, parent_scheme::UpwindBiased{$buffer, FT, XT, YT, <:Nothing}, ψ::Function, idx, loc, args...) where {FT, XT, YT}
                scheme = _topologically_conditional_scheme_z(i, j, k, grid, dir, loc, parent_scheme)
                return @inbounds $(calc_reconstruction_stencil(buffer, side, :z,  true))
            end
        end
    end
end

for (Dir, side) in zip((LeftBiasedStencil, RightBiasedStencil), (:left, :right)), (ξ, side_index) in zip((:x, :y, :z), (1, 2, 3))
    interpolate = Symbol(:upwind_biased_interpolate_, ξ)
    conditional_scheme = Symbol(:_topologically_conditional_scheme_, ξ)
    for buffer in advection_buffers
        @eval begin
            @inline function $interpolate(i, j, k, grid, dir::Dir, parent_scheme::UpwindBiased{$buffer, FT}, ψ, idx, loc, args...)           where FT
                scheme = $conditional_scheme(i, j, k, grid, dir, loc, parent_scheme)
                return @inbounds sum($(reconstruction_stencil(buffer, side, ξ, false)) .* retrieve_coeff(scheme, Dir(), Val($side_index), idx, loc))
            end
            @inline function $interpolate(i, j, k, grid, dir::Dir, parent_scheme::UpwindBiased{$buffer, FT}, ψ::Function, idx, loc, args...) where FT
                scheme = $conditional_scheme(i, j, k, grid, dir, loc, parent_scheme)
                return @inbounds sum($(reconstruction_stencil(buffer, side, ξ,  true)) .* retrieve_coeff(scheme, Dir(), Val($side_index), idx, loc))
            end
        end
    end
end

# Retrieve precomputed coefficients 
@inline retrieve_coeff(scheme::UpwindBiased, ::LeftBiasedStencil, ::Val{1}, i, ::Type{Face})   = @inbounds scheme.coeff_xᶠᵃᵃ[1][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::LeftBiasedStencil, ::Val{1}, i, ::Type{Center}) = @inbounds scheme.coeff_xᶜᵃᵃ[1][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::LeftBiasedStencil, ::Val{2}, i, ::Type{Face})   = @inbounds scheme.coeff_yᵃᶠᵃ[1][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::LeftBiasedStencil, ::Val{2}, i, ::Type{Center}) = @inbounds scheme.coeff_yᵃᶜᵃ[1][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::LeftBiasedStencil, ::Val{3}, i, ::Type{Face})   = @inbounds scheme.coeff_zᵃᵃᶠ[1][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::LeftBiasedStencil, ::Val{3}, i, ::Type{Center}) = @inbounds scheme.coeff_zᵃᵃᶜ[1][i] 

@inline retrieve_coeff(scheme::UpwindBiased, ::RightBiasedStencil, ::Val{1}, i, ::Type{Face})   = @inbounds scheme.coeff_xᶠᵃᵃ[2][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::RightBiasedStencil, ::Val{1}, i, ::Type{Center}) = @inbounds scheme.coeff_xᶜᵃᵃ[2][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::RightBiasedStencil, ::Val{2}, i, ::Type{Face})   = @inbounds scheme.coeff_yᵃᶠᵃ[2][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::RightBiasedStencil, ::Val{2}, i, ::Type{Center}) = @inbounds scheme.coeff_yᵃᶜᵃ[2][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::RightBiasedStencil, ::Val{3}, i, ::Type{Face})   = @inbounds scheme.coeff_zᵃᵃᶠ[2][i] 
@inline retrieve_coeff(scheme::UpwindBiased, ::RightBiasedStencil, ::Val{3}, i, ::Type{Center}) = @inbounds scheme.coeff_zᵃᵃᶜ[2][i] 
