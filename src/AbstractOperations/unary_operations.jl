"""
    UnaryOperation{X, Y, Z, O, A, I, G} <: AbstractOperation{X, Y, Z, G}

An abstract representation of a unary operation on an `AbstractField`; or a function
`f(x)` with on argument acting on `x::AbstractField`.
"""
struct UnaryOperation{X, Y, Z, O, A, I, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
     arg :: A
       ▶ :: I
    grid :: G

    function UnaryOperation{X, Y, Z}(op, arg, ▶, grid) where {X, Y, Z}
        return new{X, Y, Z, typeof(op), typeof(arg), typeof(▶), typeof(grid)}(op, arg, ▶, grid)
    end
end

"""Create a unary operation for `operator` acting on `arg` which interpolates the
result from `Larg` to `L`."""
function _unary_operation(L, operator, arg, Larg, grid) where {X, Y, Z}
    ▶ = interpolation_operator(Larg, L)
    return UnaryOperation{L[1], L[2], L[3]}(operator, data(arg), ▶, grid)
end

@inline Base.getindex(υ::UnaryOperation, i, j, k) = υ.▶(i, j, k, υ.grid, υ.op, υ.arg)

const unary_operators = [:sqrt, :sin, :cos, :exp]
append!(operators, unary_operators)

for op in unary_operators
    @eval begin
        import Base: $op 

        @inline $op(i, j, k, grid::AbstractGrid, a) = @inbounds $op(a[i, j, k])
        @inline $op(i, j, k, grid::AbstractGrid, a::Number) = $op(a)

        function $op(Lop::Tuple, a::AbstractLocatedField)
            L = location(a)
            return _unary_operation(Lop, $op, a, L, a.grid)
        end

        $op(a::AbstractLocatedField) = $op(location(a), a)
    end
end

Adapt.adapt_structure(to, unary::UnaryOperation{X, Y, Z}) where {X, Y, Z} =
    UnaryOperation{X, Y, Z}(adapt(to, unary.op), adapt(to, unary.arg), 
                            adapt(to, unary.▶), unary.grid)

function tree_show(unary::UnaryOperation{X, Y, Z}, depth, nesting)  where {X, Y, Z}
    padding = "    "^(depth-nesting) * "│   "^nesting

    return string(unary.op, " at ", show_location(X, Y, Z), " via ", unary.▶, '\n',
                  padding, "└── ", tree_show(unary.arg, depth+1, nesting))
end
