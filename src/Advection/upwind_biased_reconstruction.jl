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

function UpwindBiased(FT::DataType = Oceananigans.defaults.FloatType;
                      order = 3,
                      time_discretization = ExplicitTimeDiscretization(),
                      extrapolation = MinmodExtrapolation())

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

const AUAS = AbstractUpwindBiasedAdvectionScheme

# symmetric interpolation for UpwindBiased and WENO
@inline symmetric_interpolate(i, j, k, grid, scheme::AUAS, dir, ℓ, args...) = symmetric_interpolate(i, j, k, grid, scheme.advecting_velocity_scheme, dir, ℓ, args...)

@inline function biased_interpolate(i, j, k, grid, scheme::UpwindBiased{<:Any, FT}, bias, dir, ℓ, ψ, args...) where FT
    S = stencil_nodes(i, j, k, grid, scheme, dir, ℓ, ψ, args...)
    return ifelse(bias == LeftBias, linear_reconstruction(S, FT, Val(:left)), linear_reconstruction(S, FT, Val(:right)))
end
