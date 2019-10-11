struct Derivative{X, Y, Z, A, D, I, L, G} <: AbstractOperation{X, Y, Z, G}
       a :: A
       ∂ :: D
       ▶ :: I
      L∂ :: L
    grid :: G

    function Derivative{X, Y, Z}(a, ∂, L∂, grid) where {X, Y, Z}
         ▶ = interp_operator(L∂, (X, Y, Z))
        return new{X, Y, Z, typeof(a), typeof(∂), typeof(▶), typeof(L∂), typeof(grid)}(a, ∂, ▶, L∂, grid)
    end
end

@propagate_inbounds getindex(d::Derivative, i, j, k) = d.▶(i, j, k, d.grid, d.∂, d.a)

flip(::Type{Face}) = Cell
flip(::Type{Cell}) = Face

∂x(X::Union{Type{Face}, Type{Cell}}) = eval(Symbol(:∂x_, interp_code(flip(X)), :aa))
∂y(Y::Union{Type{Face}, Type{Cell}}) = eval(Symbol(:∂y_a, interp_code(flip(Y)), :a))
∂z(Z::Union{Type{Face}, Type{Cell}}) = eval(Symbol(:∂z_aa, interp_code(flip(Z))))

const ALF = AbstractLocatedField

∂x(a::ALF{X, Y, Z}) where {X, Y, Z} = Derivative{flip(X), Y, Z}(data(a), ∂x(X), (flip(X), Y, Z), a.grid)
∂y(a::ALF{X, Y, Z}) where {X, Y, Z} = Derivative{X, flip(Y), Z}(data(a), ∂y(Y), (X, flip(Y), Z), a.grid)
∂z(a::ALF{X, Y, Z}) where {X, Y, Z} = Derivative{X, Y, flip(Z)}(data(a), ∂z(Z), (X, Y, flip(Z)), a.grid)

∂x(L::Tuple, a::ALF{X, Y, Z}) where {X, Y, Z} = Derivative{L[1], L[2], L[3]}(data(a), ∂x(X), (flip(X), Y, Z), a.grid)
∂y(L::Tuple, a::ALF{X, Y, Z}) where {X, Y, Z} = Derivative{L[1], L[2], L[3]}(data(a), ∂y(Y), (X, flip(Y), Z), a.grid)
∂z(L::Tuple, a::ALF{X, Y, Z}) where {X, Y, Z} = Derivative{L[1], L[2], L[3]}(data(a), ∂z(Z), (X, Y, flip(Z)), a.grid)

Adapt.adapt_structure(to, deriv::Derivative{X, Y, Z}) where {X, Y, Z} =
    Derivative{X, Y, Z}(adapt(to, parent(deriv.a)), deriv.∂, deriv.L∂, deriv.grid)

Base.parent(deriv::Derivative) = parent(deriv.a)
