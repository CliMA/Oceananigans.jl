#####
##### Upwind-biased 3rd-order advection scheme
#####

struct UpwindBiased{N, FT, TD, EX, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT, TD}
    extrapolation :: EX
    advecting_velocity_scheme :: SI
    time_discretization :: TD

    UpwindBiased{N, FT}(extrapolation::EX, advecting_velocity_scheme::SI, time_discretization::TD) where {N, FT, EX, SI, TD} =
        new{N, FT, TD, EX, SI}(extrapolation, advecting_velocity_scheme, time_discretization)
end

"""
    UpwindBiased([FT=Oceananigans.defaults.FloatType;]
                 order = 3,
                 time_discretization = ExplicitTimeDiscretization(),
                 extrapolation = TransportedWallValue())

Construct an upwind-biased reconstruction of odd `order`. Next to boundaries the stencil nodes without data are replaced
with ghosts that `extrapolation`, `TransportedWallValue()` or `HierarchicalExtrapolation()`, builds; `order = 1` takes the
upwind node, or the other node where the upwind one has no data.
"""
function UpwindBiased(FT::DataType = Oceananigans.defaults.FloatType;
                      order = 3,
                      time_discretization = ExplicitTimeDiscretization(),
                      extrapolation = TransportedWallValue())

    mod(order, 2) == 0 && throw(ArgumentError("UpwindBiased reconstruction scheme is defined only for odd orders"))

    N = Int((order + 1) ÷ 2)

    advecting_velocity_scheme = Centered(FT; order = 2)

    return UpwindBiased{N, FT}(extrapolation, advecting_velocity_scheme, time_discretization)
end

Base.summary(a::UpwindBiased{N}) where N = string("UpwindBiased(order=", 2N-1, ")")

function Base.show(io::IO, a::UpwindBiased{N, FT}) where {N, FT}
    print(io, summary(a), '\n')
    print(io, "├── extrapolation: ", summary(a.extrapolation), '\n')
    print(io, "└── advecting_velocity_scheme: ", summary(a.advecting_velocity_scheme))
end

Adapt.adapt_structure(to, scheme::UpwindBiased{N, FT}) where {N, FT} =
    UpwindBiased{N, FT}(Adapt.adapt(to, scheme.extrapolation),
                        Adapt.adapt(to, scheme.advecting_velocity_scheme),
                        Adapt.adapt(to, scheme.time_discretization))

Architectures.on_architecture(to, scheme::UpwindBiased{N, FT}) where {N, FT} =
    UpwindBiased{N, FT}(on_architecture(to, scheme.extrapolation),
                        on_architecture(to, scheme.advecting_velocity_scheme),
                        on_architecture(to, scheme.time_discretization))


# symmetric interpolation for UpwindBiased and WENO


# The upwind reconstruction of order 2N - 1 from the upwind node ψ₀ = Sᵘ_N and the 2N - 2 differences δ of the stencil Sᵘ
for buffer in advection_buffers[2:end], FT in fully_supported_float_types
    c = reverse(uniform_reconstruction_coefficients(BigFloat, Val(:left), buffer))
    d = FT.(linear_difference_coefficients(c, buffer))
    terms = [:($(d[m]) * δ[$m]) for m in eachindex(d)]
    @eval @inline upwind_reconstruction(::UpwindBiased{$buffer, $FT}, ψ₀, δ) = @inbounds @muladd ψ₀ + $(Expr(:call, :+, terms...))
end
