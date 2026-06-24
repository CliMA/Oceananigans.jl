using Oceananigans.Operators: Operators, interpolation_code

"""
    Derivative{LX, LY, LZ}(∂, arg, ▶, grid)

Return an abstract representation of the derivative `∂` on `arg`,
and subsequent interpolation by `▶` on `grid`.
"""
struct Derivative{LX, LY, LZ, D, A, IN, AD, G, T} <: AbstractOperation{LX, LY, LZ, G, T}
               ∂ :: D
             arg :: A
               ▶ :: IN
      abstract_∂ :: AD
            grid :: G

    function Derivative{LX, LY, LZ}(∂::D, arg::A, ▶::IN, abstract_∂::AD,
                                    grid::G) where {LX, LY, LZ, D, A, IN, AD, G}
        T = eltype(grid)
        return new{LX, LY, LZ, D, A, IN, AD, G, T}(∂, arg, ▶, abstract_∂, grid)
    end
end

@inline Base.getindex(d::Derivative, i, j, k) = d.▶(i, j, k, d.grid, d.∂, d.arg)

#####
##### Derivative construction
#####

"""Create a derivative operator `∂` acting on `arg` at `L∂`, followed by
interpolation to `L` on `grid`."""
function _derivative(L::Tuple{LX, LY, LZ}, ∂, arg, L∂, abstract_∂, grid) where {LX, LY, LZ}
    ▶ = interpolation_operator(L∂, L)
    return Derivative{LX, LY, LZ}(∂, arg, ▶, abstract_∂, grid)
end

indices(d::Derivative) = indices(d.arg)

# Recompute location of derivative
@inline at(loc, d::Derivative) = d.abstract_∂(loc, d.arg)

"""Return `Center` if given `Face` or `Face` if given `Center`."""
flip(::Type{Face}) = Center
flip(::Type{Center}) = Face
flip(::Face) = Center()
flip(::Center) = Face()

"""Return the ``x``-derivative function acting at (`X`, `Y`, `Any`)."""
∂x(X::Location, Y::Location, Z::Location) = Operators.eval(Symbol(:∂x, interpolation_code(flip(X)), interpolation_code(Y), interpolation_code(Z)))

"""Return the ``y``-derivative function acting at (`X`, `Y`, `Any`)."""
∂y(X::Location, Y::Location, Z::Location) = Operators.eval(Symbol(:∂y, interpolation_code(X), interpolation_code(flip(Y)), interpolation_code(Z)))

"""Return the ``z``-derivative function acting at (`Any`, `Any`, `Z`)."""
∂z(X::Location, Y::Location, Z::Location) = Operators.eval(Symbol(:∂z, interpolation_code(X), interpolation_code(Y), interpolation_code(flip(Z))))

const derivative_operators = Set([:∂x, :∂y, :∂z])
push!(operators, derivative_operators...)

"""
$(TYPEDSIGNATURES)

Return an abstract representation of an ``x``-derivative acting on field `arg` followed
by interpolation to `L`, where `L` is a 3-tuple of instantiated `Face`s and `Center`s.
"""
∂x(L::Tuple{<:Location, <:Location, <:Location}, arg::AF{LX, LY, LZ}) where {LX, LY, LZ} =
    _derivative(L, ∂x(LX(), LY(), LZ()), arg, (flip(LX()), LY(), LZ()), ∂x, arg.grid)

"""
$(TYPEDSIGNATURES)

Return an abstract representation of a ``y``-derivative acting on field `arg` followed
by interpolation to `L`, where `L` is a 3-tuple of instantiated `Face`s and `Center`s.
"""
∂y(L::Tuple{<:Location, <:Location, <:Location}, arg::AF{LX, LY, LZ}) where {LX, LY, LZ} =
    _derivative(L, ∂y(LX(), LY(), LZ()), arg, (LX(), flip(LY()), LZ()), ∂y, arg.grid)

"""
$(TYPEDSIGNATURES)

Return an abstract representation of a ``z``-derivative acting on field `arg` followed
by  interpolation to `L`, where `L` is a 3-tuple of instantiated `Face`s and `Center`s.
"""
∂z(L::Tuple{<:Location, <:Location, <:Location}, arg::AF{LX, LY, LZ}) where {LX, LY, LZ} =
    _derivative(L, ∂z(LX(), LY(), LZ()), arg, (LX(), LY(), flip(LZ())), ∂z, arg.grid)

# Instantiate location if types are passed
∂x(L::Tuple, arg::AF) = ∂x((L[1](), L[2](), L[3]()), arg)
∂y(L::Tuple, arg::AF) = ∂y((L[1](), L[2](), L[3]()), arg)
∂z(L::Tuple, arg::AF) = ∂z((L[1](), L[2](), L[3]()), arg)

# Defaults

"""
$(TYPEDSIGNATURES)

Return an abstract representation of a ``x``-derivative acting on field `arg`.
"""
∂x(arg::AF{LX, LY, LZ}) where {LX, LY, LZ} = ∂x((flip(LX()), LY(), LZ()), arg)

"""
$(TYPEDSIGNATURES)

Return an abstract representation of a ``y``-derivative acting on field `arg`.
"""
∂y(arg::AF{LX, LY, LZ}) where {LX, LY, LZ} = ∂y((LX(), flip(LY()), LZ()), arg)

"""
$(TYPEDSIGNATURES)

Return an abstract representation of a ``z``-derivative acting on field `arg`.
"""
∂z(arg::AF{LX, LY, LZ}) where {LX, LY, LZ} = ∂z((LX(), LY(), flip(LZ())), arg)

#####
##### Nested computations
#####

compute_at!(∂::Derivative, time) = compute_at!(∂.arg, time)

#####

#####
##### GPU capabilities
#####

"Adapt `Derivative` to work on the GPU."
Adapt.adapt_structure(to, deriv::Derivative{LX, LY, LZ}) where {LX, LY, LZ} =
    Derivative{LX, LY, LZ}(Adapt.adapt(to, deriv.∂),
                           Adapt.adapt(to, deriv.arg),
                           Adapt.adapt(to, deriv.▶),
                           nothing,
                           Adapt.adapt(to, deriv.grid))

Architectures.on_architecture(to, deriv::Derivative{LX, LY, LZ}) where {LX, LY, LZ} =
    Derivative{LX, LY, LZ}(on_architecture(to, deriv.∂),
                           on_architecture(to, deriv.arg),
                           on_architecture(to, deriv.▶),
                           deriv.abstract_∂,
                           on_architecture(to, deriv.grid))
