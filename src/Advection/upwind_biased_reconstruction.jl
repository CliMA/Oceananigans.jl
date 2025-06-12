#####
##### Upwind-biased 3rd-order advection scheme
#####

"""
    struct UpwindBiased <: AbstractUpwindBiasedAdvectionScheme{3}

Upwind-biased reconstruction scheme.
"""
struct UpwindBiased{N, FT, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT}
    "Reconstruction scheme used near boundaries"
    buffer_scheme :: CA
    "Reconstruction scheme used for symmetric interpolation"
    advecting_velocity_scheme :: SI

    UpwindBiased{N, FT}(buffer_scheme::CA, advecting_velocity_scheme::SI) where {N, FT, CA, SI} =
        new{N, FT, CA, SI}(buffer_scheme, advecting_velocity_scheme)
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
        advecting_velocity_scheme = Centered(FT; grid, order = 2)
        buffer_scheme  = nothing
    end

    return UpwindBiased{N, FT}(buffer_scheme, advecting_velocity_scheme)
end

Base.summary(a::UpwindBiased{N}) where N = string("UpwindBiased(order=", 2N-1, ")")

Base.show(io::IO, a::UpwindBiased{N, FT}) where {N, FT} =
    print(io, summary(a), " \n",
              " Boundary scheme: ", "\n",
              "    └── ", summary(a.buffer_scheme) , "\n",
              " Symmetric scheme: ", "\n",
              "    └── ", summary(a.advecting_velocity_scheme))

Adapt.adapt_structure(to, scheme::UpwindBiased{N, FT}) where {N, FT} =
    UpwindBiased{N, FT}(Adapt.adapt(to, scheme.buffer_scheme),
                        Adapt.adapt(to, scheme.advecting_velocity_scheme))

on_architecture(to, scheme::UpwindBiased{N, FT}) where {N, FT} =
    UpwindBiased{N, FT}(on_architecture(to, scheme.buffer_scheme),
                        on_architecture(to, scheme.advecting_velocity_scheme))

# Useful aliases
UpwindBiased(grid, FT::DataType=Float64; kwargs...) = UpwindBiased(FT; grid, kwargs...)

const AUAS = AbstractUpwindBiasedAdvectionScheme

# symmetric interpolation for UpwindBiased and WENO
@inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::AUAS, args...) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::AUAS, args...) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::AUAS, args...) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::AUAS, args...) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::AUAS, args...) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::AUAS, args...) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme.advecting_velocity_scheme, args...)

# Uniform upwind biased reconstruction
for buffer in advection_buffers, FT in fully_supported_float_types
    @eval begin
        @inline biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::UpwindBiased{$buffer, $FT}, bias, ψ, args...) = 
            @inbounds @muladd ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, buffer, :left,  :x, false)), 
                                                        $(calc_reconstruction_stencil(FT, buffer, :right, :x, false)))

        @inline biased_interpolate_xᶠᵃᵃ(i, j, k, grid, ::UpwindBiased{$buffer, $FT}, bias, ψ::Callable, args...) = 
            @inbounds @muladd ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, buffer, :left,  :x, true)), 
                                                        $(calc_reconstruction_stencil(FT, buffer, :right, :x, true)))
    
        @inline biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::UpwindBiased{$buffer, $FT}, bias, ψ, args...) = 
            @inbounds @muladd ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, buffer, :left,  :y, false)), 
                                                        $(calc_reconstruction_stencil(FT, buffer, :right, :y, false)))
                                                 
        @inline biased_interpolate_yᵃᶠᵃ(i, j, k, grid, ::UpwindBiased{$buffer, $FT}, bias, ψ::Callable, args...) = 
            @inbounds @muladd ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, buffer, :left,  :y, true)), 
                                                        $(calc_reconstruction_stencil(FT, buffer, :right, :y, true)))
    
        @inline biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::UpwindBiased{$buffer, $FT}, bias, ψ, args...) = 
            @inbounds @muladd ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, buffer, :left,  :z, false)), 
                                                        $(calc_reconstruction_stencil(FT, buffer, :right, :z, false)))

        @inline biased_interpolate_zᵃᵃᶠ(i, j, k, grid, ::UpwindBiased{$buffer, $FT}, bias, ψ::Callable, args...) = 
            @inbounds @muladd ifelse(bias isa LeftBias, $(calc_reconstruction_stencil(FT, buffer, :left,  :z, true)), 
                                                        $(calc_reconstruction_stencil(FT, buffer, :right, :z, true)))         
    end
end