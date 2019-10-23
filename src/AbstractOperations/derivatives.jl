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

"""Create a derivative operator `∂` acting on `arg` at `L∂`, followed by
interpolation to `L` on `grid`."""
function _derivative(L, ∂, arg, L∂, grid) where {X, Y, Z}
    ▶ = interpolation_operator(L∂, L)
    return Derivative{L[1], L[2], L[3]}(∂, data(arg), ▶, grid)
end

@inline Base.getindex(d::Derivative, i, j, k) = d.▶(i, j, k, d.grid, d.∂, d.arg)

"""Return `Cell` if given `Face` or `Face` if given `Cell`."""
flip(::Type{Face}) = Cell
flip(::Type{Cell}) = Face

"""Return the x-derivative function acting at (`X`, `Any`, `Any`)."""
∂x(X::Union{Type{Face}, Type{Cell}}) = eval(Symbol(:∂x_, interpolation_code(flip(X)), :aa))

"""Return the y-derivative function acting at (`Any`, `Y`, `Any`)."""
∂y(Y::Union{Type{Face}, Type{Cell}}) = eval(Symbol(:∂y_a, interpolation_code(flip(Y)), :a))

"""Return the z-derivative function acting at (`Any`, `Any`, `Z`)."""
∂z(Z::Union{Type{Face}, Type{Cell}}) = eval(Symbol(:∂z_aa, interpolation_code(flip(Z))))

const derivative_operators = [:∂x, :∂y, :∂z]
append!(operators, derivative_operators)

"""
    ∂x(L::Tuple, a::Oceananigans.AbstractLocatedField)

Return an abstract representation of an x-derivative acting on `a` followed by
interpolation to `L`, where `L` is a 3-tuple of `Face`s and `Cell`s.
"""
∂x(L::Tuple, arg::ALF{X, Y, Z}) where {X, Y, Z} = 
    _derivative(L, ∂x(X), arg, (flip(X), Y, Z), arg.grid)

"""
    ∂y(L::Tuple, a::Oceananigans.AbstractLocatedField)

Return an abstract representation of a y-derivative acting on `a` followed by
interpolation to `L`, where `L` is a 3-tuple of `Face`s and `Cell`s.
"""
∂y(L::Tuple, arg::ALF{X, Y, Z}) where {X, Y, Z} = 
    _derivative(L, ∂y(Y), arg, (X, flip(Y), Z), arg.grid)

"""
    ∂z(L::Tuple, a::Oceananigans.AbstractLocatedField)

Return an abstract representation of a z-derivative acting on `a` followed by
interpolation to `L`, where `L` is a 3-tuple of `Face`s and `Cell`s.
"""
∂z(L::Tuple, arg::ALF{X, Y, Z}) where {X, Y, Z} = 
    _derivative(L, ∂z(Z), arg, (X, Y, flip(Z)), arg.grid)
    

# Defaults
"""
    ∂x(a::Oceananigans.AbstractLocatedField)

Return an abstract representation of a x-derivative acting on `a`.
"""
∂x(arg::ALF{X, Y, Z}) where {X, Y, Z} = ∂x((flip(X), Y, Z), arg)

"""
    ∂y(a::Oceananigans.AbstractLocatedField)

Return an abstract representation of a y-derivative acting on `a`.
"""
∂y(arg::ALF{X, Y, Z}) where {X, Y, Z} = ∂y((X, flip(Y), Z), arg)
"""
    ∂z(a::Oceananigans.AbstractLocatedField)

Return an abstract representation of a z-derivative acting on `a`.
"""
∂z(arg::ALF{X, Y, Z}) where {X, Y, Z} = ∂z((X, Y, flip(Z)), arg)

"Adapt `Derivative` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, deriv::Derivative{X, Y, Z}) where {X, Y, Z} =
    Derivative{X, Y, Z}(adapt(to, deriv.∂), adapt(to, deriv.arg), 
                        adapt(to, deriv.▶), deriv.grid)
