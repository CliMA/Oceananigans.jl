#####
##### Upwind-biased 3rd-order advection scheme
#####

struct UpwindBiased{N, FT, TD, CA, SI} <: AbstractUpwindBiasedAdvectionScheme{N, FT, TD}
    buffer_scheme :: CA
    advecting_velocity_scheme :: SI
    time_discretization :: TD

    UpwindBiased{N, FT}(buffer_scheme::CA, advecting_velocity_scheme::SI, time_discretization::TD) where {N, FT, CA, SI, TD} =
        new{N, FT, TD, CA, SI}(buffer_scheme, advecting_velocity_scheme, time_discretization)
end

function UpwindBiased(FT::DataType = Oceananigans.defaults.FloatType;
                      order = 3,
                      time_discretization = ExplicitTimeDiscretization(),
                      buffer_scheme = DecreasingOrderAdvectionScheme())

    mod(order, 2) == 0 && throw(ArgumentError("UpwindBiased reconstruction scheme is defined only for odd orders"))

    N = Int((order + 1) ÷ 2)
    advecting_velocity_scheme = Centered(FT; order = 2)

    if buffer_scheme isa DecreasingOrderAdvectionScheme
        buffer_scheme = N > 1 ? UpwindBiased(FT; order=order-2, buffer_scheme) : buffer_scheme.boundary_scheme
    end

    return UpwindBiased{N, FT}(buffer_scheme, advecting_velocity_scheme, time_discretization)
end

Base.summary(a::UpwindBiased{N}) where N = string("UpwindBiased(order=", 2N-1, ")")

# Helper function to recursively print buffer scheme tree
function print_buffer_scheme_tree(io::IO, scheme, prefix::String, is_last::Bool)
    connector = is_last ? "└── " : "├── "
    print(io, prefix, connector, "buffer_scheme: ", summary(scheme))

    # Check if this scheme has a nested buffer_scheme to display
    if hasproperty(scheme, :buffer_scheme) && !isnothing(scheme.buffer_scheme)
        println(io)
        new_prefix = prefix * (is_last ? "    " : "│   ")
        print_buffer_scheme_tree(io, scheme.buffer_scheme, new_prefix, true)
    end
end

function Base.show(io::IO, a::UpwindBiased{N, FT}) where {N, FT}
    print(io, summary(a), '\n')

    # Print buffer scheme tree recursively
    if !isnothing(a.buffer_scheme)
        print_buffer_scheme_tree(io, a.buffer_scheme, "", false)
        println(io)
    else
        print(io, "├── buffer_scheme: ", summary(a.buffer_scheme), '\n')
    end

    print(io, "└── advecting_velocity_scheme: ", summary(a.advecting_velocity_scheme))
end

Adapt.adapt_structure(to, scheme::UpwindBiased{N, FT}) where {N, FT} =
    UpwindBiased{N, FT}(Adapt.adapt(to, scheme.buffer_scheme),
                        Adapt.adapt(to, scheme.advecting_velocity_scheme),
                        Adapt.adapt(to, scheme.time_discretization))

Architectures.on_architecture(to, scheme::UpwindBiased{N, FT}) where {N, FT} =
    UpwindBiased{N, FT}(on_architecture(to, scheme.buffer_scheme),
                        on_architecture(to, scheme.advecting_velocity_scheme),
                        on_architecture(to, scheme.time_discretization))

const AUAS = AbstractUpwindBiasedAdvectionScheme

# symmetric interpolation for UpwindBiased and WENO
@inline symmetric_interpolate(i, j, k, grid, scheme::AUAS, dir, ℓ, args...) = symmetric_interpolate(i, j, k, grid, scheme.advecting_velocity_scheme, dir, ℓ, args...)

@inline function biased_interpolate(i, j, k, grid, ::UpwindBiased{N, FT}, bias, dir, ℓ, ψ, args...) where {N, FT}
    S = stencil_nodes(i, j, k, grid, Val(N), dir, ℓ, ψ, args...)
    return ifelse(bias == LeftBias, linear_reconstruction(S, FT, Val(:left)), linear_reconstruction(S, FT, Val(:right)))
end
