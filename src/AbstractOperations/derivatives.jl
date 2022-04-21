using Oceananigans.Operators: interpolation_code

struct Derivative{LX, LY, LZ, D, A, I, AD, G, T} <: AbstractOperation{LX, LY, LZ, G, T}
               ∂ :: D
             arg :: A
               ▶ :: I
      abstract_∂ :: AD
            grid :: G

    @doc """
        Derivative{LX, LY, LZ}(∂, arg, ▶, grid)

    Returns an abstract representation of the derivative `∂` on `arg`,
    and subsequent interpolation by `▶` on `grid`.
    """
    function Derivative{LX, LY, LZ}(∂::D, arg::A, ▶::I, abstract_∂::AD,
                                 grid::G) where {LX, LY, LZ, D, A, I, AD, G}
        T = eltype(grid)
        return new{LX, LY, LZ, D, A, I, AD, G, T}(∂, arg, ▶, abstract_∂, grid)
    end
end

@inline Base.getindex(d::Derivative, i, j, k) = d.▶(i, j, k, d.grid, d.∂, d.arg)

#####
##### Derivative construction
#####

"""Create a derivative operator `∂` acting on `arg` at `L∂`, followed by
interpolation to `L` on `grid`."""
function _derivative(L, ∂, arg, L∂, abstract_∂, grid) where {LX, LY, LZ}
    ▶ = interpolation_operator(L∂, L)
    return Derivative{L[1], L[2], L[3]}(∂, arg, ▶, abstract_∂, grid)
end

# Recompute location of derivative
@inline at(loc, d::Derivative) = d.abstract_∂(loc, d.arg)

"""Return `Center` if given `Face` or `Face` if given `Center`."""
flip(::Type{Face}) = Center
flip(::Type{Center}) = Face

const LocationType = Union{Type{Face}, Type{Center}, Type{Nothing}}

"""Return the ``x``-derivative function acting at (`X`, `Y`, `Any`)."""
∂x(X::LocationType, Y::LocationType, Z::LocationType) = eval(Symbol(:∂x, interpolation_code(flip(X)), interpolation_code(Y), interpolation_code(Z)))

"""Return the ``y``-derivative function acting at (`X`, `Y`, `Any`)."""
∂y(X::LocationType, Y::LocationType, Z::LocationType) = eval(Symbol(:∂y, interpolation_code(X), interpolation_code(flip(Y)), interpolation_code(Z)))

"""Return the ``z``-derivative function acting at (`Any`, `Any`, `Z`)."""
∂z(X::LocationType, Y::LocationType, Z::LocationType) = eval(Symbol(:∂z, interpolation_code(X), interpolation_code(Y), interpolation_code(flip(Z))))

const derivative_operators = Set([:∂x, :∂y, :∂z])
push!(operators, derivative_operators...)

"""
    ∂x(L::Tuple, arg::AbstractField)

Return an abstract representation of an ``x``-derivative acting on field `a` followed
by interpolation to `L`, where `L` is a 3-tuple of `Face`s and `Center`s.
"""
∂x(L::Tuple, arg::AF{LX, LY, LZ}) where {LX, LY, LZ} =
    _derivative(L, ∂x(LX, LY, LZ), arg, (flip(LX), LY, LZ), ∂x, arg.grid)

"""
    ∂y(L::Tuple, arg::AbstractField)

Return an abstract representation of a ``y``-derivative acting on field `a` followed
by interpolation to `L`, where `L` is a 3-tuple of `Face`s and `Center`s.
"""
∂y(L::Tuple, arg::AF{LX, LY, LZ}) where {LX, LY, LZ} =
    _derivative(L, ∂y(LX, LY, LZ), arg, (LX, flip(LY), LZ), ∂y, arg.grid)

"""
    ∂z(L::Tuple, arg::AbstractField)

Return an abstract representation of a ``z``-derivative acting on field `a` followed
by  interpolation to `L`, where `L` is a 3-tuple of `Face`s and `Center`s.
"""
∂z(L::Tuple, arg::AF{LX, LY, LZ}) where {LX, LY, LZ} =
    _derivative(L, ∂z(LX, LY, LZ), arg, (LX, LY, flip(LZ)), ∂z, arg.grid)

# Defaults
"""
    ∂x(arg::AbstractField)

Return an abstract representation of a ``x``-derivative acting on field `a`.
"""
∂x(arg::AF{LX, LY, LZ}) where {LX, LY, LZ} = ∂x((flip(LX), LY, LZ), arg)

"""
    ∂y(arg::AbstractField)

Return an abstract representation of a ``y``-derivative acting on field `a`.
"""
∂y(arg::AF{LX, LY, LZ}) where {LX, LY, LZ} = ∂y((LX, flip(LY), LZ), arg)
"""
    ∂z(arg::AbstractField)

Return an abstract representation of a ``z``-derivative acting on field `a`.
"""
∂z(arg::AF{LX, LY, LZ}) where {LX, LY, LZ} = ∂z((LX, LY, flip(LZ)), arg)

#####
##### Nested computations
#####

compute_at!(∂::Derivative, time) = compute_at!(∂.arg, time)

#####

#####
##### GPU capabilities
#####

"Adapt `Derivative` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, deriv::Derivative{LX, LY, LZ}) where {LX, LY, LZ} =
    Derivative{LX, LY, LZ}(Adapt.adapt(to, deriv.∂),
                           Adapt.adapt(to, deriv.arg),
                           Adapt.adapt(to, deriv.▶),
                           nothing,
                           Adapt.adapt(to, deriv.grid))

