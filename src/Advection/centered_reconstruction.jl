#####
##### Centered advection scheme
#####

"""
    struct Centered{N, FT, XT, YT, ZT, CA} <: AbstractCenteredAdvectionScheme{N, FT}

Centered reconstruction scheme.
"""
struct Centered{N, FT, CA} <: AbstractCenteredAdvectionScheme{N, FT}
    "advection scheme used near boundaries"
    buffer_scheme :: CA

    Centered{N, FT}(buffer_scheme::CA) where {N, FT, CA} = new{N, FT, CA}(buffer_scheme)
end

function Centered(FT::DataType=Oceananigans.defaults.FloatType; grid = nothing, order = 2)

    if !(grid isa Nothing)
        FT = eltype(grid)
    end

    mod(order, 2) != 0 && throw(ArgumentError("Centered reconstruction scheme is defined only for even orders"))

    N  = Int(order ÷ 2)
    if N > 1
        buffer_scheme = Centered(FT; grid, order = order - 2)
    else
        buffer_scheme = nothing
    end
    return Centered{N, FT}(buffer_scheme)
end

Base.summary(a::Centered{N}) where N = string("Centered(order=", 2N, ")")

Base.show(io::IO, a::Centered{N, FT}) where {N, FT} =
    print(io, summary(a), " \n",
              " Boundary scheme: ", "\n",
              "    └── ", summary(a.buffer_scheme))


Adapt.adapt_structure(to, scheme::Centered{N, FT}) where {N, FT} = Centered{N, FT}(Adapt.adapt(to, scheme.buffer_scheme))

on_architecture(to, scheme::Centered{N, FT}) where {N, FT} = Centered{N, FT}(on_architecture(to, scheme.buffer_scheme))

# Useful aliases
Centered(grid, FT::DataType=Float64; kwargs...) = Centered(FT; grid, kwargs...)

const ACAS = AbstractCenteredAdvectionScheme

# left and right biased for Centered reconstruction are just symmetric!
@inline biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::ACAS, bias, args...) = symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, args...)
@inline biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::ACAS, bias, args...) = symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, args...)
@inline biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::ACAS, bias, args...) = symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, args...)
@inline biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme::ACAS, bias, args...) = symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, args...)
@inline biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme::ACAS, bias, args...) = symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, args...)
@inline biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme::ACAS, bias, args...) = symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, args...)

# uniform centered reconstruction
for buffer in advection_buffers, FT in fully_supported_float_types
    @eval begin
        @inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::Centered{$buffer, $FT}, ψ, args...) = @inbounds @muladd $(calc_reconstruction_stencil(FT, buffer, :symmetric, :x, false))
        @inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::Centered{$buffer, $FT}, ψ, args...) = @inbounds @muladd $(calc_reconstruction_stencil(FT, buffer, :symmetric, :y, false))
        @inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::Centered{$buffer, $FT}, ψ, args...) = @inbounds @muladd $(calc_reconstruction_stencil(FT, buffer, :symmetric, :z, false))
        
        @inline symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, ::Centered{$buffer, $FT}, ψ::Callable, args...) = @inbounds @muladd $(calc_reconstruction_stencil(FT, buffer, :symmetric, :x,  true))
        @inline symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, ::Centered{$buffer, $FT}, ψ::Callable, args...) = @inbounds @muladd $(calc_reconstruction_stencil(FT, buffer, :symmetric, :y,  true))
        @inline symmetric_interpolate_zᵃᵃᶠ(i, j, k, grid, ::Centered{$buffer, $FT}, ψ::Callable, args...) = @inbounds @muladd $(calc_reconstruction_stencil(FT, buffer, :symmetric, :z,  true))
    end
end