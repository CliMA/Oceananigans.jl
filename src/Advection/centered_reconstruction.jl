#####
##### Centered advection scheme
#####

struct Centered{N, FT, VD, CA} <: AbstractCenteredAdvectionScheme{N, FT, VD}
    buffer_scheme :: CA
    vertical_discretization :: VD
    Centered{N, FT}(buffer_scheme::CA, vertical_discretization::VD) where {N, FT, CA} = new{N, FT, VD, CA}(buffer_scheme, vertical_discretization)
end

function Centered(FT::DataType=Oceananigans.defaults.FloatType;
                  order = 2,
                  vertical_discretization = ExplicitTimeDiscretization(),
                  buffer_scheme = DecreasingOrderAdvectionScheme())

    mod(order, 2) != 0 && throw(ArgumentError("Centered reconstruction scheme is defined only for even orders"))

    N  = Int(order ÷ 2)
    if buffer_scheme isa DecreasingOrderAdvectionScheme
        if N > 1
            buffer_scheme = Centered(FT; order=order-2)
        else
            buffer_scheme = nothing
        end
    end

    return Centered{N, FT}(buffer_scheme, vertical_discretization)
end

Base.summary(a::Centered{N}) where N = string("Centered(order=", 2N, ")")

Base.show(io::IO, a::Centered{N, FT}) where {N, FT} =
    print(io, summary(a), " \n",
              "└── buffer_scheme: ", summary(a.buffer_scheme))


Adapt.adapt_structure(to, scheme::Centered{N, FT}) where {N, FT} = 
    Centered{N, FT}(Adapt.adapt(to, scheme.buffer_scheme), Adapt.adapt(to, scheme.vertical_discretization))

on_architecture(to, scheme::Centered{N, FT}) where {N, FT} = 
    Centered{N, FT}(on_architecture(to, scheme.buffer_scheme), on_architecture(to, scheme.vertical_discretization))

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
