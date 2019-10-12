struct BinaryOperation{X, Y, Z, O, A, B, IA, IB, IΩ, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
       b :: B
      ▶a :: IA
      ▶b :: IB
     ▶op :: IΩ
    grid :: G

    function BinaryOperation{X, Y, Z}(op, a, b, ▶a, ▶b, ▶op, grid) where {X, Y, Z}
        return new{X, Y, Z, typeof(op), typeof(a), typeof(b), typeof(▶a), typeof(▶b), 
                   typeof(▶op), typeof(grid)}(op, a, b, ▶a, ▶b, ▶op, grid)
    end
end

function BinaryOperation{X, Y, Z}(op, a, b, La::Tuple, Lb::Tuple, Lop::Tuple, grid) where {X, Y, Z}
     ▶a = interpolation_operator(La, Lop)
     ▶b = interpolation_operator(Lb, Lop)
    ▶op = interpolation_operator(Lop, (X, Y, Z))
    return BinaryOperation{X, Y, Z}(op, a, b, ▶a, ▶b, ▶op, grid)
end


@inline Base.getindex(β::BinaryOperation, i, j, k) = 
    β.▶op(i, j, k, β.grid, β.op, β.▶a, β.▶b, β.a, β.b)

const binary_operators = [:+, :-, :/, :*, :^]
append!(operators, binary_operators)

for op in binary_operators
    @eval begin
        import Base: $op 

        @inline $op(i, j, k, grid::AbstractGrid, ▶a, ▶b, a, b) = 
            @inbounds $op(▶a(i, j, k, grid, a), ▶b(i, j, k, grid, b))

        function $op(Lc::Tuple, Lop::Tuple, a, b)
            La = location(a)
            Lb = location(b)
            grid = validate_grid(a, b)
            return BinaryOperation{Lc[1], Lc[2], Lc[3]}($op, data(a), data(b), La, Lb, Lop, grid)
        end

        $op(Lc::Tuple, a, b) = $op(Lc, Lc, a, b)
        $op(Lc::Tuple, a::Number, b) = $op(Lc, location(b), a, b)
        $op(Lc::Tuple, a, b::Number) = $op(Lc, location(a), a, b)
        $op(Lc::Tuple, a::ALF{X, Y, Z}, b::ALF{X, Y, Z}) where {X, Y, Z} = $op(Lc, location(a), a, b)

        # Sugar for mixing in functions of (x, y, z)
        $op(Lc::Tuple, a::Function, b::AbstractField) = $op(Lc, FunctionField(Lc, a, b.grid), b) 
        $op(Lc::Tuple, a::AbstractField, b::Function) = $op(Lc, a, FunctionField(Lc, b, a.grid))

        # Sugary versions with default locations
        $op(a::AbstractLocatedField, b::AbstractLocatedField) = $op(location(a), a, b)

        $op(a::AbstractLocatedField, b::Number) = $op(location(a), a, b)
        $op(a::Number, b::AbstractLocatedField) = $op(location(b), a, b)

        $op(a::AbstractLocatedField, b::Function) = $op(location(a), a, FunctionField(location(a), b, a.grid))
        $op(a::Function, b::AbstractLocatedField) = $op(location(b), FunctionField(location(b), a, b.grid), b)
    end
end

Adapt.adapt_structure(to, binary::BinaryOperation{X, Y, Z}) where {X, Y, Z} =
    BinaryOperation{X, Y, Z}(adapt(to, binary.op), adapt(to, binary.a), adapt(to, binary.b), 
                             adapt(to, binary.▶a), adapt(to, binary.▶b), adapt(to, binary.▶op),  
                             binary.grid)
