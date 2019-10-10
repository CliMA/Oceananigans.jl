struct Derivative{X, Y, Z, A, D, G} <: AbstractOperation{X, Y, Z, G}
       a :: A
       ∂ :: D
    grid :: G
    function Derivative{X, Y, Z}(a, ∂) where {X, Y, Z}
        return new{X, Y, Z, typeof(alldata(a)), typeof(∂), typeof(a.grid)}(alldata(a), ∂, a.grid)
    end
end

data(bop::BinaryOperation) = bop

flip(::Type{Face}) = Cell
flip(::Type{Cell}) = Face

function ∂x(a::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}
    ∂ = eval(Symbol(:∂x_, interp_code(flip(X)), :aa))
    return Derivative{flip(X), Y, Z}(a, ∂)
end

function ∂y(a::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}
    ∂ = eval(Symbol(:∂y_a, interp_code(flip(Y)), :a))
    return Derivative{X, flip(Y), Z}(a, ∂)
end

function ∂z(a::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}
    ∂ = eval(Symbol(:∂z_aa, interp_code(flip(Z))))
    return Derivative{X, Y, flip(Z)}(a, ∂)
end

@propagate_inbounds getindex(d::Derivative, i, j, k) = d.∂(i, j, k, d.grid, d.a)
