using Oceananigans.Operators: interpolation_code

struct Derivative{X, Y, Z, D, A, I, AD, R, G, T} <: AbstractOperation{X, Y, Z, R, G, T}
               ∂ :: D
             arg :: A
               ▶ :: I
      abstract_∂ :: AD
    architecture :: R
            grid :: G

    """
        Derivative{X, Y, Z}(∂, arg, ▶, grid)

    Returns an abstract representation of the derivative `∂` on `arg`,
    and subsequent interpolation by `▶` on `grid`.
    """
    function Derivative{X, Y, Z}(∂::D, arg::A, ▶::I, abstract_∂::AD,
                                 arch::R, grid::G) where {X, Y, Z, D, A, I, AD, R, G}
        T = eltype(grid)
        return new{X, Y, Z, D, A, I, AD, R, G, T}(∂, arg, ▶, abstract_∂, arch, grid)
    end
end

@inline Base.getindex(d::Derivative, i, j, k) = d.▶(i, j, k, d.grid, d.∂, d.arg)

#####
##### Derivative construction
#####

"""Create a derivative operator `∂` acting on `arg` at `L∂`, followed by
interpolation to `L` on `grid`."""
function _derivative(L, ∂, arg, L∂, abstract_∂, grid) where {X, Y, Z}
    ▶ = interpolation_operator(L∂, L)
    arch = architecture(arg)
    return Derivative{L[1], L[2], L[3]}(∂, arg, ▶, abstract_∂, arch, grid)
end

# Recompute location of derivative
@inline at(loc, d::Derivative) = d.abstract_∂(loc, d.arg)

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
    _derivative(L, ∂x(X, Y, Z), arg, (flip(X), Y, Z), ∂x, arg.grid)

"""
    ∂y(L::Tuple, a::AbstractField)

Return an abstract representation of a y-derivative acting on `a` followed by
interpolation to `L`, where `L` is a 3-tuple of `Face`s and `Center`s.
"""
∂y(L::Tuple, arg::AF{X, Y, Z}) where {X, Y, Z} =
    _derivative(L, ∂y(X, Y, Z), arg, (X, flip(Y), Z), ∂y, arg.grid)

"""
    ∂z(L::Tuple, a::AbstractField)

Return an abstract representation of a z-derivative acting on `a` followed by
interpolation to `L`, where `L` is a 3-tuple of `Face`s and `Center`s.
"""
∂z(L::Tuple, arg::AF{X, Y, Z}) where {X, Y, Z} =
    _derivative(L, ∂z(X, Y, Z), arg, (X, Y, flip(Z)), ∂z, arg.grid)

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

architecture(∂::Derivative) = ∂.architecture

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
                        Adapt.adapt(to, deriv.▶), nothing, nothing, Adapt.adapt(to, deriv.grid))

