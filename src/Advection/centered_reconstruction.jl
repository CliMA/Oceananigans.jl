#####
##### Centered advection scheme
#####

struct Centered{N, FT, TD, EX} <: AbstractCenteredAdvectionScheme{N, FT, TD}
    extrapolation :: EX
    time_discretization :: TD
    Centered{N, FT}(extrapolation::EX, time_discretization::TD) where {N, FT, EX, TD} = new{N, FT, TD, EX}(extrapolation, time_discretization)
end

function Centered(FT::DataType=Oceananigans.defaults.FloatType;
                  order = 2,
                  time_discretization = ExplicitTimeDiscretization(),
                  extrapolation = MinmodExtrapolation())

    mod(order, 2) != 0 && throw(ArgumentError("Centered reconstruction scheme is defined only for even orders"))

    N = Int(order ÷ 2)

    return Centered{N, FT}(extrapolation, time_discretization)
end

Base.summary(a::Centered{N}) where N = string("Centered(order=", 2N, ")")

Base.show(io::IO, a::Centered) = print(io, summary(a), '\n', "└── extrapolation: ", summary(a.extrapolation))

Adapt.adapt_structure(to, scheme::Centered{N, FT}) where {N, FT} =
    Centered{N, FT}(Adapt.adapt(to, scheme.extrapolation), Adapt.adapt(to, scheme.time_discretization))

Architectures.on_architecture(to, scheme::Centered{N, FT}) where {N, FT} =
    Centered{N, FT}(on_architecture(to, scheme.extrapolation), on_architecture(to, scheme.time_discretization))

const ACAS = AbstractCenteredAdvectionScheme

# left and right biased for Centered reconstruction are just symmetric!
@inline biased_interpolate(i, j, k, grid, scheme::ACAS, bias, dir, ℓ, args...) = symmetric_interpolate(i, j, k, grid, scheme, dir, ℓ, args...)

@inline function symmetric_interpolate(i, j, k, grid, scheme::Centered, dir, ℓ, ψ, args...)
    S = stencil_values(i, j, k, grid, scheme, dir, ℓ, ψ, args...)
    return linear_reconstruction(S, typeof(zero(grid)), Val(:symmetric))
end
