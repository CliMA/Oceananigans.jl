#####
##### Centered advection scheme
#####

struct Centered{N, FT, TD, EX} <: AbstractCenteredAdvectionScheme{N, FT, TD}
    extrapolation :: EX
    time_discretization :: TD
    Centered{N, FT}(extrapolation::EX, time_discretization::TD) where {N, FT, EX, TD} = new{N, FT, TD, EX}(extrapolation, time_discretization)
end

"""
    Centered([FT=Oceananigans.defaults.FloatType;]
             order = 2,
             time_discretization = ExplicitTimeDiscretization(),
             extrapolation = TransportedWallValue())

Construct a centered reconstruction of even `order`. Next to boundaries the stencil nodes without data are replaced with
ghosts that `extrapolation`, `TransportedWallValue()` or `HierarchicalExtrapolation()`, builds; `order = 2` takes the node
with data next to a boundary.
"""
function Centered(FT::DataType=Oceananigans.defaults.FloatType;
                  order = 2,
                  time_discretization = ExplicitTimeDiscretization(),
                  extrapolation = TransportedWallValue())

    mod(order, 2) != 0 && throw(ArgumentError("Centered reconstruction scheme is defined only for even orders"))

    N = Int(order ÷ 2)

    return Centered{N, FT}(extrapolation, time_discretization)
end

Base.summary(a::Centered{N}) where N = string("Centered(order=", 2N, ")")

Base.show(io::IO, a::Centered{N, FT}) where {N, FT} =
    print(io, summary(a), " \n",
              "└── extrapolation: ", summary(a.extrapolation))

Adapt.adapt_structure(to, scheme::Centered{N, FT}) where {N, FT} =
    Centered{N, FT}(Adapt.adapt(to, scheme.extrapolation), Adapt.adapt(to, scheme.time_discretization))

Architectures.on_architecture(to, scheme::Centered{N, FT}) where {N, FT} =
    Centered{N, FT}(on_architecture(to, scheme.extrapolation), on_architecture(to, scheme.time_discretization))


# Coefficients of a linear reconstruction `∑ cₙ Sₙ` on the differences δₘ = Sₘ₊₁ - Sₘ of its stencil, relative to the node `S_N`
linear_difference_coefficients(c, N) = ntuple(m -> m < N ? -sum(c[1:m]) : sum(c[m+1:end]), length(c) - 1)

# The centered reconstruction of order 2N from the upwind node ψ₀ = S_N and the 2N - 1 differences δ of the stencil
for buffer in advection_buffers[2:end], FT in fully_supported_float_types
    c = uniform_reconstruction_coefficients(BigFloat, Val(:symmetric), buffer)
    d = FT.(linear_difference_coefficients(c, buffer))
    terms = [:($(d[m]) * δ[$m]) for m in eachindex(d)]
    @eval @inline centered_reconstruction(::Centered{$buffer, $FT}, ψ₀, δ) = @inbounds @muladd ψ₀ + $(Expr(:call, :+, terms...))
end
