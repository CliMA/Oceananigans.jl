struct UnaryOperation{X, Y, Z, A, I, L, G, O} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
       ▶ :: I
      La :: L
    grid :: G
    function UnaryOperation{X, Y, Z}(op, a, La, grid) where {X, Y, Z}
        ▶ = interpolation_operator(La, (X, Y, Z))
        return new{X, Y, Z, typeof(a), typeof(▶), typeof(La), typeof(grid), typeof(op)}(op, a, ▶, La, grid)
    end
end

@inline Base.getindex(υ::UnaryOperation, i, j, k) = υ.▶(i, j, k, υ.grid, υ.op, υ.a)

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
    UnaryOperation{X, Y, Z}(unary.op, adapt(to, unary.a), unary.La, unary.grid)
