struct UnaryOperation{X, Y, Z, O, A, I, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
     arg :: A
       ▶ :: I
    grid :: G

    function UnaryOperation{X, Y, Z}(op, arg, ▶, grid) where {X, Y, Z}
        return new{X, Y, Z, typeof(op), typeof(arg), typeof(▶), typeof(grid)}(op, arg, ▶, grid)
    end
end

function UnaryOperation{X, Y, Z}(op, arg, L::Tuple, grid) where {X, Y, Z}
    ▶ = interpolation_operator(L, (X, Y, Z))
    return UnaryOperation{X, Y, Z}(op, arg, ▶, grid)
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
            return UnaryOperation{L[1], L[2], L[3]}($op, data(a), L, a.grid)
        end

        $op(a::AbstractLocatedField) = $op(location(a), a)
    end
end

Adapt.adapt_structure(to, unary::UnaryOperation{X, Y, Z}) where {X, Y, Z} =
    UnaryOperation{X, Y, Z}(adapt(to, unary.op), adapt(to, unary.arg), 
                            adapt(to, unary.▶), unary.grid)
