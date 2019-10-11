struct BinaryOperation{X, Y, Z, A, B, IA, IB, LA, LB, O, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
       b :: B
      ▶a :: IA
      ▶b :: IB
      La :: LA
      Lb :: LB
    grid :: G

    function BinaryOperation{X, Y, Z}(op, a, b, ▶a, ▶b) where {X, Y, Z}
        grid = validate_grid(a, b)
        La = location(a)
        Lb = location(b)
        return new{X, Y, Z, typeof(data(a)), typeof(data(b)), 
                   typeof(▶a), typeof(▶b), typeof(La), typeof(Lb),
                   typeof(op), typeof(grid)}(op, data(a), data(b), ▶a, ▶b, La, Lb, grid)
    end
end

@inline getindex(β::BinaryOperation, i, j, k) =
    interpolate_then_operate(i, j, k, β.grid, β.op, β.▶a, β.▶b, β.a, β.b)

@inline interpolate_then_operate(i, j, k, grid, op, ▶a, ▶b, a, b) =
    op(▶a(i, j, k, grid, a), ▶b(i, j, k, grid, b))

for op in (:+, :-, :/, :*)
    @eval begin
        function $op(a::AbstractLocatedField, b::AbstractLocatedField)
            La = location(a)
            Lb = location(b)
            ▶a = identity
            ▶b = interp_operator(Lb, La)
            return BinaryOperation{La[1], La[2], La[3]}($op, a, b, ▶a, ▶b)
        end

        $op(a::AbstractLocatedField{X, Y, Z}, b::Number) where {X, Y, Z} =
            BinaryOperation{X, Y, Z}($op, a, b, identity, identity)

        $op(a::Number, b::AbstractLocatedField{X, Y, Z}) where {X, Y, Z} =
            BinaryOperation{X, Y, Z}($op, a, b, identity, identity)

        function $op(Lop::Tuple, a, b)
            La = location(a)
            Lb = location(b)
            ▶a = interp_operator(La, Lop)
            ▶b = interp_operator(Lb, Lop)
            return BinaryOperation{Lop[1], Lop[2], Lop[3]}($op, a, b, ▶a, ▶b)
        end
    end
end

for op in (:^,)
    @eval begin
        $op(a::AbstractLocatedField{X, Y, Z}, b::Number) where {X, Y, Z} =
            BinaryOperation{X, Y, Z}($op, a, b, identity, identity)
    end
end
