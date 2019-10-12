struct Derivative{X, Y, Z, D, A, I, G} <: AbstractOperation{X, Y, Z, G}
       ∂ :: D
     arg :: A
       ▶ :: I
    grid :: G

    function Derivative{X, Y, Z}(∂, arg, ▶, grid) where {X, Y, Z}
        return new{X, Y, Z, typeof(∂), typeof(arg), typeof(▶), typeof(grid)}(∂, arg, ▶, grid)
    end
end

function _derivative(L, ∂, arg, L∂, grid) where {X, Y, Z}
    ▶ = interpolation_operator(L∂, L)
    return Derivative{L[1], L[2], L[3]}(∂, arg, ▶, grid)
end

@inline Base.getindex(d::Derivative, i, j, k) = d.▶(i, j, k, d.grid, d.∂, d.arg)

flip(::Type{Face}) = Cell
flip(::Type{Cell}) = Face

∂x(X::Union{Type{Face}, Type{Cell}}) = eval(Symbol(:∂x_, interpolation_code(flip(X)), :aa))
∂y(Y::Union{Type{Face}, Type{Cell}}) = eval(Symbol(:∂y_a, interpolation_code(flip(Y)), :a))
∂z(Z::Union{Type{Face}, Type{Cell}}) = eval(Symbol(:∂z_aa, interpolation_code(flip(Z))))

const derivative_operators = [:∂x, :∂y, :∂z]
append!(operators, derivative_operators)

∂x(L::Tuple, arg::ALF{X, Y, Z}) where {X, Y, Z} = 
    _derivative(L, ∂x(X), arg, (flip(X), Y, Z), arg.grid)

∂y(L::Tuple, arg::ALF{X, Y, Z}) where {X, Y, Z} = 
    _derivative(L, ∂y(Y), arg, (X, flip(Y), Z), arg.grid)

∂z(L::Tuple, arg::ALF{X, Y, Z}) where {X, Y, Z} = 
    _derivative(L, ∂z(Z), arg, (X, Y, flip(Z)), arg.grid)

# Defaults
∂x(arg::ALF{X, Y, Z}) where {X, Y, Z} = ∂x((flip(X), Y, Z), arg)
∂y(arg::ALF{X, Y, Z}) where {X, Y, Z} = ∂y((X, flip(Y), Z), arg)
∂z(arg::ALF{X, Y, Z}) where {X, Y, Z} = ∂z((X, Y, flip(Z)), arg)

Adapt.adapt_structure(to, deriv::Derivative{X, Y, Z}) where {X, Y, Z} =
    Derivative{X, Y, Z}(adapt(to, deriv.∂), adapt(to, deriv.arg), 
                        adapt(to, deriv.▶), deriv.grid)

function tree_show(deriv::Derivative{X, Y, Z}, depth, nesting)  where {X, Y, Z}
    padding = "    "^(depth-nesting) * "│   "^nesting
    return string(deriv.∂, " at ", show_location(X, Y, Z), '\n',
                  padding, "└── ", tree_show(deriv.arg, depth+1, nesting))
end
