struct BinaryOperation{X, Y, Z, O, A, B, IA, IB, IΩ, LA, LB, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
       b :: B
      ▶a :: IA
      ▶b :: IB
     ▶op :: IΩ
      La :: LA
      Lb :: LB
    grid :: G

    function BinaryOperation{X, Y, Z}(op, a, b, ▶a, ▶b, ▶op, La, Lb, grid) where {X, Y, Z}
        return new{X, Y, Z, typeof(op), typeof(a), typeof(b), typeof(▶a), typeof(▶b), 
                   typeof(▶op), typeof(La), typeof(Lb), typeof(grid)}(op, a, b, ▶a, ▶b, ▶op, La, Lb, grid)
    end
end

function _binary_operation(Lc, op, a, b, La, Lb, Lop, grid) where {X, Y, Z}
     ▶a = interpolation_operator(La, Lop)
     ▶b = interpolation_operator(Lb, Lop)
    ▶op = interpolation_operator(Lop, Lc)
     La = instantiate(La)
     Lb = instantiate(La)
    return BinaryOperation{Lc[1], Lc[2], Lc[3]}(op, data(a), data(b), ▶a, ▶b, ▶op, La, Lb, grid)
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
            return _binary_operation(Lc, $op, a, b, La, Lb, Lop, grid)
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
                             binary.La, binary.Lb, binary.grid)

function tree_show(binary::BinaryOperation{X, Y, Z}, depth, nesting) where {X, Y, Z}
    padding = "    "^(depth-nesting) * "│   "^nesting

    return string(binary.op, " at ", show_location(X, Y, Z), " via ", binary.▶op, '\n',
                  padding, "├── ", tree_show(binary.a, depth+1, nesting+1), '\n',
                  padding, "└── ", tree_show(binary.b, depth+1, nesting))
end
