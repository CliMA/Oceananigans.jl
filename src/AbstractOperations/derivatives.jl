struct Derivative{X, Y, Z, A, D, I, L, G} <: AbstractOperation{X, Y, Z, G}
       a :: A
       ∂ :: D
       ▶ :: I
      L∂ :: L
    grid :: G

    function Derivative{X, Y, Z}(a, ∂, L∂) where {X, Y, Z}
         ▶ = interp_operator(L∂, (X, Y, Z))
        return new{X, Y, Z, typeof(data(a)), typeof(∂), 
                   typeof(▶), typeof(L∂), typeof(a.grid)}(data(a), ∂, ▶, L∂, a.grid)
    end
end

flip(::Type{Face}) = Cell
flip(::Type{Cell}) = Face

function ∂x(a::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}
    ∂ = eval(Symbol(:∂x_, interp_code(flip(X)), :aa))
    return Derivative{flip(X), Y, Z}(a, ∂, (flip(X), Y, Z))
end

function ∂y(a::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}
    ∂ = eval(Symbol(:∂y_a, interp_code(flip(Y)), :a))
    return Derivative{X, flip(Y), Z}(a, ∂, (X, flip(Y), Z))
end

function ∂z(a::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}
    ∂ = eval(Symbol(:∂z_aa, interp_code(flip(Z))))
    return Derivative{X, Y, flip(Z)}(a, ∂, (X, Y, flip(Z)))
end

function ∂x(L::Tuple, a::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}
    ∂ = eval(Symbol(:∂x_, interp_code(flip(X)), :aa))
    return Derivative{L[1], L[2], L[3]}(a, ∂, (flip(X), Y, Z))
end

function ∂y(L::Tuple, a::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}
    ∂ = eval(Symbol(:∂y_a, interp_code(flip(Y)), :a))
    return Derivative{L[1], L[2], L[3]}(a, ∂, (X, flip(Y), Z))
end

function ∂z(L::Tuple, a::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}
    ∂ = eval(Symbol(:∂z_aa, interp_code(flip(Z))))
    return Derivative{L[1], L[2], L[3]}(a, ∂, (X, Y, flip(Z)))
end

@propagate_inbounds getindex(d::Derivative, i, j, k) = d.▶(i, j, k, d.grid, d.∂, d.a)
