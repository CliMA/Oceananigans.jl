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

function _binary_operation(Lc, op, a, b, La, Lb, Lab, grid) where {X, Y, Z}
     ▶a = interpolation_operator(La, Lab)
     ▶b = interpolation_operator(Lb, Lab)
    ▶op = interpolation_operator(Lab, Lc)
    return BinaryOperation{Lc[1], Lc[2], Lc[3]}(op, data(a), data(b), ▶a, ▶b, ▶op, grid)
end

@inline Base.getindex(β::BinaryOperation, i, j, k) = β.▶op(i, j, k, β.grid, β.op, β.▶a, β.▶b, β.a, β.b)

const binary_operators = [:+, :-, :/, :*, :^]
append!(operators, binary_operators)

import Base: +, -, /, *, ^

for op in binary_operators
    @eval begin
        import Oceananigans

        @inline $op(i, j, k, grid::Oceananigans.AbstractGrid, ▶a, ▶b, a, b) =
            @inbounds $op(▶a(i, j, k, grid, a), ▶b(i, j, k, grid, b))

        """
            $($op)(Lc, Lab, a, b)

        Returns an abstract representation of the operator `$($op)` acting on `a` and `b` at
        location `Lab`, and subsequently interpolated to location `Lc`.
        """
        function $op(Lc::Tuple, Lop::Tuple, a, b)
            La = Oceananigans.location(a)
            Lb = Oceananigans.location(b)
            grid = Oceananigans.AbstractOperations.validate_grid(a, b)
            return Oceananigans.AbstractOperations._binary_operation(Lc, $op, a, b, La, Lb, Lop, grid)
        end

        $op(Lc::Tuple, a, b) = $op(Lc, Lc, a, b)
        $op(Lc::Tuple, a::Number, b) = $op(Lc, Oceananigans.location(b), a, b)
        $op(Lc::Tuple, a, b::Number) = $op(Lc, Oceananigans.location(a), a, b)

        $op(Lc::Tuple, 
            a::Oceananigans.AbstractLocatedField{X, Y, Z}, 
            b::Oceananigans.AbstractLocatedField{X, Y, Z}) where {X, Y, Z} = $op(Lc, Oceananigans.location(a), a, b)

        # Sugar for mixing in functions of (x, y, z)
        $op(Lc::Tuple, a::Function, b::Oceananigans.AbstractField) = 
            $op(Lc, Oceananigans.AbstractOperations.FunctionField(Lc, a, b.grid), b)

        $op(Lc::Tuple, a::Oceananigans.AbstractField, b::Function) = 
            $op(Lc, a, Oceananigans.AbstractOperations.FunctionField(Lc, b, a.grid))

        # Sugary versions with default locations
        $op(a::Oceananigans.AbstractLocatedField, b::Oceananigans.AbstractLocatedField) = 
            $op(Oceananigans.location(a), a, b)

        $op(a::Oceananigans.AbstractLocatedField, b::Number) = $op(Oceananigans.location(a), a, b)
        $op(a::Number, b::Oceananigans.AbstractLocatedField) = $op(Oceananigans.location(b), a, b)

        $op(a::Oceananigans.AbstractLocatedField, b::Function) = 
            $op(Oceananigans.location(a), 
                a, Oceananigans.AbstractOperations.FunctionField(Oceananigans.location(a), b, a.grid))

        $op(a::Function, b::Oceananigans.AbstractLocatedField) = 
            $op(Oceananigans.location(b), 
                Oceananigans.AbstractOperations.FunctionField(Oceananigans.location(b), a, b.grid), b)
    end
end

#=
const binary_operators = []
macro binary(ops...)
    expr = Expr(:block)

    for op in ops
        define_binary_operator = quote
            import Oceananigans

            @inline $op(i, j, k, grid::Oceananigans.AbstractGrid, ▶a, ▶b, a, b) = 
                @inbounds $op(▶a(i, j, k, grid, a), ▶b(i, j, k, grid, b))

            """
                $($op)(Lc, Lab, a, b)

            Returns an abstract representation of the operator `$($op)` acting on `a` and `b` at
            location `Lab`, and subsequently interpolated to location `Lc`.
            """
            function $op(Lc::Tuple, Lab::Tuple, a, b)
                La = Oceananigans.location(a)
                Lb = Oceananigans.location(b)
                grid = Oceananigans.AbstractOperations.validate_grid(a, b)
                return Oceananigans.AbstractOperations._binary_operation(Lc, $op, a, b, La, Lb, Lab, grid)
            end

            # Functions for avoiding over-interpolation when multiplying fields with numbers
            # or other fields with like-locations
            $op(Lc::Tuple, a, b) = $op(Lc, Lc, a, b)
            $op(Lc::Tuple, a::Number, b) = $op(Lc, Oceananigans.location(b), a, b)
            $op(Lc::Tuple, a, b::Number) = $op(Lc, Oceananigans.location(a), a, b)

            $op(Lc::Tuple, a::Oceananigans.AbstractLocatedField{X, Y, Z}, 
                           b::Oceananigans.AbstractLocatedField{X, Y, Z}) where {X, Y, Z} = 
                $op(Lc, Oceananigans.location(a), a, b)

            # Sugar for mixing in functions of (x, y, z)
            $op(Lc::Tuple, a::Function, b::Oceananigans.AbstractField) = 
                $op(Lc, Oceananigans.AbstractOperations.FunctionField(Lc, a, b.grid), b) 
                
            $op(Lc::Tuple, a::Oceananigans.AbstractField, b::Function) = 
                $op(Lc, a, Oceananigans.AbstractOperations.FunctionField(Lc, b, a.grid))

            # Sugary operators with default locations
            $op(a::Oceananigans.AbstractLocatedField, b::Oceananigans.AbstractLocatedField) = 
                $op(Oceananigans.location(a), a, b)

            $op(a::Oceananigans.AbstractLocatedField, b::Number) = $op(Oceananigans.location(a), a, b)
            $op(a::Number, b::Oceananigans.AbstractLocatedField) = $op(Oceananigans.location(b), a, b)

            $op(a::Oceananigans.AbstractLocatedField, b::Function) = 
                $op(Oceananigans.location(a), 
                        a, Oceananigans.AbstractOperations.FunctionField(Oceananigans.location(a), b, a.grid))

            $op(a::Function, b::Oceananigans.AbstractLocatedField) = 
                $op(Oceananigans.location(b), 
                        Oceananigans.AbstractOperations.FunctionField(Oceananigans.location(b), a, b.grid), b)

            push!(Oceananigans.AbstractOperations.binary_operators, $op)
            push!(Oceananigans.AbstractOperations.operators, $op)
        end

        push!(expr.args, :($(esc(define_binary_operator))))
    end

    push!(expr.args, :(nothing))

    return expr
end
=#

"Adapt `BinaryOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, binary::BinaryOperation{X, Y, Z}) where {X, Y, Z} =
    BinaryOperation{X, Y, Z}(adapt(to, binary.op), adapt(to, binary.a), adapt(to, binary.b), 
                             adapt(to, binary.▶a), adapt(to, binary.▶b), adapt(to, binary.▶op),  
                             binary.grid)
