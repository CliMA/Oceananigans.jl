#####
##### Centered advection scheme
#####

struct Centered{N, FT, TD, CA} <: AbstractCenteredAdvectionScheme{N, FT, TD}
    buffer_scheme :: CA
    time_discretization :: TD
    Centered{N, FT}(buffer_scheme::CA, time_discretization::TD) where {N, FT, CA, TD} = new{N, FT, TD, CA}(buffer_scheme, time_discretization)
end

function Centered(FT::DataType=Oceananigans.defaults.FloatType;
                  order = 2,
                  time_discretization = ExplicitTimeDiscretization(),
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

    return Centered{N, FT}(buffer_scheme, time_discretization)
end

Base.summary(a::Centered{N}) where N = string("Centered(order=", 2N, ")")

Base.show(io::IO, a::Centered{N, FT}) where {N, FT} =
    print(io, summary(a), " \n",
              "└── buffer_scheme: ", summary(a.buffer_scheme))


Adapt.adapt_structure(to, scheme::Centered{N, FT}) where {N, FT} =
    Centered{N, FT}(Adapt.adapt(to, scheme.buffer_scheme), Adapt.adapt(to, scheme.time_discretization))

Architectures.on_architecture(to, scheme::Centered{N, FT}) where {N, FT} =
    Centered{N, FT}(on_architecture(to, scheme.buffer_scheme), on_architecture(to, scheme.time_discretization))

const ACAS = AbstractCenteredAdvectionScheme

# left and right biased for Centered reconstruction are just symmetric!
@inline biased_interpolate(i, j, k, grid, dir, ℓ, scheme::ACAS, bias, args...) = symmetric_interpolate(i, j, k, grid, dir, ℓ, scheme, args...)
@inline biased_interpolate(scheme::ACAS, bias, S) = symmetric_interpolate(scheme, S)

# uniform centered reconstruction from the `2N` values `S` of `load_stencil`
for buffer in advection_buffers, FT in fully_supported_float_types
    @eval @inline symmetric_interpolate(::Centered{$buffer, $FT}, S) = @inbounds @muladd $(calc_reconstruction_stencil(FT, buffer, :symmetric))
end

@inline function symmetric_interpolate(i, j, k, grid, dir, ℓ, scheme::Centered{N}, ψ, args...) where N
    S = load_stencil(i, j, k, grid, dir, ℓ, Val(N), ψ, args...)
    return symmetric_interpolate(scheme, S)
end
