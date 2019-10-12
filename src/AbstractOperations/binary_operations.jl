struct BinaryOperation{X, Y, Z, A, B, IA, IB, IΩ, LA, LB, LΩ, G, O} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
       b :: B
      ▶a :: IA
      ▶b :: IB
     ▶op :: IΩ
      La :: LA
      Lb :: LB
     Lop :: LΩ
    grid :: G

    function BinaryOperation{X, Y, Z}(op, a, b, La, Lb, Lop, grid) where {X, Y, Z}
         ▶a = interpolation_operator(La, Lop)
         ▶b = interpolation_operator(Lb, Lop)
        ▶op = interpolation_operator(Lop, (X, Y, Z))
         La = instantiate(La)
         Lb = instantiate(Lb)
        Lop = instantiate(Lop)
        return new{X, Y, Z, typeof(a), typeof(b), 
                   typeof(▶a), typeof(▶b), typeof(▶op), typeof(La), typeof(Lb),
                   typeof(Lop), typeof(grid), typeof(op)}(op, a, b, ▶a, ▶b, ▶op, La, Lb, Lop, grid)
    end
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
    BinaryOperation{X, Y, Z}(adapt(to, binary.op), adapt(to, binary.a), adapt(to, binary.b), 
                             binary.La,  binary.Lb, binary.Lop, binary.grid)
