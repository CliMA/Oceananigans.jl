struct BinaryOperation{X, Y, Z, A, B, IA, IB, LA, LB, G, O} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
       b :: B
      ▶a :: IA
      ▶b :: IB
      La :: LA
      Lb :: LB
    grid :: G

    function BinaryOperation{X, Y, Z}(op, a, b, La, Lb, grid) where {X, Y, Z}
        ▶a = interpolation_operator(La, (X, Y, Z))
        ▶b = interpolation_operator(Lb, (X, Y, Z))
        return new{X, Y, Z, typeof(a), typeof(b), 
                   typeof(▶a), typeof(▶b), typeof(La), typeof(Lb),
                   typeof(grid), typeof(op)}(op, a, b, ▶a, ▶b, La, Lb, grid)
    end
end

@inline Base.getindex(β::BinaryOperation, i, j, k) = 
    β.op(β.▶a(i, j, k, β.grid, β.a), β.▶b(i, j, k, β.grid, β.b))

const binary_operators = [:+, :-, :/, :*, :^]
append!(operators, binary_operators)

for op in binary_operators
    @eval begin
        import Base: $op 

        function $op(Lop::Tuple, a, b)
            La = location(a)
            Lb = location(b)
            grid = validate_grid(a, b)
            return BinaryOperation{Lop[1], Lop[2], Lop[3]}($op, data(a), data(b), La, Lb, grid)
        end

        # Sugary versions with default locations
        $op(a::AbstractLocatedField, b::AbstractLocatedField) = $op(location(a), a, b)
        $op(a::AbstractLocatedField, b::Number) = $op(location(a), a, b)
        $op(a::Number, b::AbstractLocatedField) = $op(location(b), a, b)

        # Sugar for mixing in functions of (x, y, z)
        $op(Lop::Tuple, a::Function, b::AbstractField) = $op(Lop, FunctionField(Lop, a, b.grid), b) 
        $op(Lop::Tuple, a::AbstractField, b::Function) = $op(Lop, a, FunctionField(Lop, b, a.grid))

        $op(a::AbstractLocatedField, b::Function) = $op(location(a), a, FunctionField(location(a), b, a.grid))
        $op(a::Function, b::AbstractLocatedField) = $op(location(b), FunctionField(location(b), a, b.grid), b)
    end
end

Adapt.adapt_structure(to, binary::BinaryOperation{X, Y, Z}) where {X, Y, Z} =
    BinaryOperation{X, Y, Z}(binary.op, adapt(to, binary.a), adapt(to, binary.b), 
                            binary.La,  binary.Lb, binary.grid)
