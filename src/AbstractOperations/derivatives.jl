using Oceananigans.Operators: interpolation_code

"""
    Derivative{X, Y, Z, D, A, I, G} <: AbstractOperation{X, Y, Z, G}

An abstract representation of a derivative of an `AbstractField`.
"""
struct Derivative{X, Y, Z, D, A, I, G} <: AbstractOperation{X, Y, Z, G}
       ∂ :: D
     arg :: A
       ▶ :: I
    grid :: G

    """
        Derivative{X, Y, Z}(∂, arg, ▶, grid)

    Returns an abstract representation of the derivative `∂` on `arg`,
    and subsequent interpolation by `▶` on `grid`.
    """
    function Derivative{X, Y, Z}(∂, arg, ▶, grid) where {X, Y, Z}
        return new{X, Y, Z, typeof(∂), typeof(arg), typeof(▶), typeof(grid)}(∂, arg, ▶, grid)
    end
end

@inline Base.getindex(d::Derivative, i, j, k) = d.▶(i, j, k, d.grid, d.∂, d.arg)

#####
##### Derivative construction
#####

"""Create a derivative operator `∂` acting on `arg` at `L∂`, followed by
interpolation to `L` on `grid`."""
function _derivative(L, ∂, arg, L∂, grid) where {X, Y, Z}
    ▶ = interpolation_operator(L∂, L)
    return Derivative{L[1], L[2], L[3]}(∂, arg, ▶, grid)
end

"""Return `Center` if given `Face` or `Face` if given `Center`."""
flip(::Type{Face}) = Center
flip(::Type{Center}) = Face

const LocationType = Union{Type{Face}, Type{Center}, Type{Nothing}}

"""Return the x-derivative function acting at (`X`, `Y`, `Any`)."""
∂x(X::LocationType, Y::LocationType, Z::LocationType) = eval(Symbol(:∂x, interpolation_code(flip(X)), interpolation_code(Y), :ᵃ))

"""Return the y-derivative function acting at (`X`, `Y`, `Any`)."""
∂y(X::LocationType, Y::LocationType, Z::LocationType) = eval(Symbol(:∂y, interpolation_code(X), interpolation_code(flip(Y)), :ᵃ))

"""Return the z-derivative function acting at (`Any`, `Any`, `Z`)."""
∂z(X::LocationType, Y::LocationType, Z::LocationType) = eval(Symbol(:∂zᵃᵃ, interpolation_code(flip(Z))))

const derivative_operators = Set([:∂x, :∂y, :∂z])
push!(operators, derivative_operators...)

"""
    ∂x(L::Tuple, a::AbstractField)

Return an abstract representation of an x-derivative acting on `a` followed by
interpolation to `L`, where `L` is a 3-tuple of `Face`s and `Center`s.
"""
∂x(L::Tuple, arg::AF{X, Y, Z}) where {X, Y, Z} =
    _derivative(L, ∂x(X, Y, Z), arg, (flip(X), Y, Z), arg.grid)

"""
    ∂y(L::Tuple, a::AbstractField)

Return an abstract representation of a y-derivative acting on `a` followed by
interpolation to `L`, where `L` is a 3-tuple of `Face`s and `Center`s.
"""
∂y(L::Tuple, arg::AF{X, Y, Z}) where {X, Y, Z} =
    _derivative(L, ∂y(X, Y, Z), arg, (X, flip(Y), Z), arg.grid)

"""
    ∂z(L::Tuple, a::AbstractField)

Return an abstract representation of a z-derivative acting on `a` followed by
interpolation to `L`, where `L` is a 3-tuple of `Face`s and `Center`s.
"""
∂z(L::Tuple, arg::AF{X, Y, Z}) where {X, Y, Z} =
    _derivative(L, ∂z(X, Y, Z), arg, (X, Y, flip(Z)), arg.grid)

# Defaults
"""
    ∂x(a::AbstractField)

Return an abstract representation of a x-derivative acting on `a`.
"""
∂x(arg::AF{X, Y, Z}) where {X, Y, Z} = ∂x((flip(X), Y, Z), arg)

"""
    ∂y(a::AbstractField)

Return an abstract representation of a y-derivative acting on `a`.
"""
∂y(arg::AF{X, Y, Z}) where {X, Y, Z} = ∂y((X, flip(Y), Z), arg)
"""
    ∂z(a::AbstractField)

Return an abstract representation of a z-derivative acting on `a`.
"""
∂z(arg::AF{X, Y, Z}) where {X, Y, Z} = ∂z((X, Y, flip(Z)), arg)

#####
##### Architecture inference for derivatives
#####

architecture(∂::Derivative) = architecture(∂.arg)

#####
##### Nested computations
#####

compute_at!(∂::Derivative, time) = compute_at!(∂.arg, time)

#####

#####
##### GPU capabilities
#####

"Adapt `Derivative` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, deriv::Derivative{X, Y, Z}) where {X, Y, Z} =
    Derivative{X, Y, Z}(Adapt.adapt(to, deriv.∂), Adapt.adapt(to, deriv.arg),
                        Adapt.adapt(to, deriv.▶), deriv.grid)
