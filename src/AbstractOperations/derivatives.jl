struct Derivative{X, Y, Z, A, I, G, D} <: AbstractOperation{X, Y, Z, G}
       ∂ :: D
       a :: A
       ▶ :: I
    grid :: G

    function Derivative{X, Y, Z}(∂, a, L∂, grid) where {X, Y, Z}
        ▶ = interpolation_operator(L∂, (X, Y, Z))
        return new{X, Y, Z, typeof(a), typeof(▶), typeof(grid), typeof(∂)}(∂, a, ▶, grid)
    end
end

@inline Base.getindex(d::Derivative, i, j, k) = d.▶(i, j, k, d.grid, d.∂, d.a)

flip(::Type{Face}) = Cell
flip(::Type{Cell}) = Face

∂x(X::Union{Type{Face}, Type{Cell}}) = eval(Symbol(:∂x_, interpolation_code(flip(X)), :aa))
∂y(Y::Union{Type{Face}, Type{Cell}}) = eval(Symbol(:∂y_a, interpolation_code(flip(Y)), :a))
∂z(Z::Union{Type{Face}, Type{Cell}}) = eval(Symbol(:∂z_aa, interpolation_code(flip(Z))))

const derivative_operators = [:∂x, :∂y, :∂z]
append!(operators, derivative_operators)

∂x(L::Tuple, a::ALF{X, Y, Z}) where {X, Y, Z} = 
    Derivative{L[1], L[2], L[3]}(∂x(X), data(a), (flip(X), Y, Z), a.grid)

∂y(L::Tuple, a::ALF{X, Y, Z}) where {X, Y, Z} = 
    Derivative{L[1], L[2], L[3]}(∂y(Y), data(a), (X, flip(Y), Z), a.grid)

∂z(L::Tuple, a::ALF{X, Y, Z}) where {X, Y, Z} = 
    Derivative{L[1], L[2], L[3]}(∂z(Z), data(a), (X, Y, flip(Z)), a.grid)

# Defaults
∂x(a::ALF{X, Y, Z}) where {X, Y, Z} = ∂x((flip(X), Y, Z), a)
∂y(a::ALF{X, Y, Z}) where {X, Y, Z} = ∂y((X, flip(Y), Z), a)
∂z(a::ALF{X, Y, Z}) where {X, Y, Z} = ∂z((X, Y, flip(Z)), a)

Adapt.adapt_structure(to, deriv::Derivative{X, Y, Z}) where {X, Y, Z} =
    Derivative{X, Y, Z}(adapt(to, deriv.∂), adapt(to, deriv.a), deriv.L∂, deriv.grid)
