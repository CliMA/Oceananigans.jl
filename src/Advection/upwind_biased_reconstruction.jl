#####
##### Upwind-biased 3rd-order advection scheme
#####

struct UpwindBiased{N, FT, TD, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT, TD}
    advecting_velocity_scheme :: SI
    time_discretization :: TD

    UpwindBiased{N, FT}(advecting_velocity_scheme::SI, time_discretization::TD) where {N, FT, SI, TD} =
        new{N, FT, TD, SI}(advecting_velocity_scheme, time_discretization)
end

function UpwindBiased(FT::DataType = Oceananigans.defaults.FloatType;
                      order = 3,
                      time_discretization = ExplicitTimeDiscretization())

    mod(order, 2) == 0 && throw(ArgumentError("UpwindBiased reconstruction scheme is defined only for odd orders"))

    N = Int((order + 1) ÷ 2)
    return UpwindBiased{N, FT}(Centered(FT; order = 2), time_discretization)
end

Base.summary(a::UpwindBiased{N}) where N = string("UpwindBiased(order=", 2N-1, ")")

Base.show(io::IO, a::UpwindBiased) =
    print(io, summary(a), '\n', "└── advecting_velocity_scheme: ", summary(a.advecting_velocity_scheme))

Adapt.adapt_structure(to, scheme::UpwindBiased{N, FT}) where {N, FT} =
    UpwindBiased{N, FT}(Adapt.adapt(to, scheme.advecting_velocity_scheme),
                        Adapt.adapt(to, scheme.time_discretization))

Architectures.on_architecture(to, scheme::UpwindBiased{N, FT}) where {N, FT} =
    UpwindBiased{N, FT}(on_architecture(to, scheme.advecting_velocity_scheme),
                        on_architecture(to, scheme.time_discretization))

const AUAS = AbstractUpwindBiasedAdvectionScheme

# symmetric interpolation for UpwindBiased and WENO
@inline symmetric_interpolate(i, j, k, grid, dir, ℓ, scheme::AUAS, args...) =
    symmetric_interpolate(i, j, k, grid, dir, ℓ, scheme.advecting_velocity_scheme, args...)

# Uniform upwind biased reconstruction from the `2N` values `S` of `load_stencil`
for buffer in advection_buffers, FT in fully_supported_float_types
    @eval @inline biased_interpolate(::UpwindBiased{$buffer, $FT}, bias, S) =
        @inbounds @muladd ifelse(bias == LeftBias, $(calc_reconstruction_stencil(FT, buffer, :left)),
                                                   $(calc_reconstruction_stencil(FT, buffer, :right)))
end

@inline function biased_interpolate(i, j, k, grid, dir, ℓ, scheme::UpwindBiased{N}, bias, ψ, args...) where N
    S = load_stencil(i, j, k, grid, dir, ℓ, Val(N), ψ, args...)
    return biased_interpolate(scheme, bias, S)
end
