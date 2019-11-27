"""
    BinaryOperation{X, Y, Z, O, A, B, IA, IB, IΩ, G} <: AbstractOperation{X, Y, Z, G}

An abstract representation of a binary operation on `AbstractField`s.
"""
struct BinaryOperation{X, Y, Z, O, A, B, IA, IB, IΩ, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
       b :: B
      ▶a :: IA
      ▶b :: IB
     ▶op :: IΩ
    grid :: G

    """
        BinaryOperation{X, Y, Z}(op, a, b, ▶a, ▶b, ▶op, grid)

    Returns an abstract representation of the binary operation `op(▶a(a), ▶b(b))`,
    followed by interpolation by `▶op` to `(X, Y, Z)`, where `▶a` and `▶b` interpolate
    `a` and `b` to a common location.
    """
    function BinaryOperation{X, Y, Z}(op, a, b, ▶a, ▶b, ▶op, grid) where {X, Y, Z}
        return new{X, Y, Z, typeof(op), typeof(a), typeof(b), typeof(▶a), typeof(▶b),
                   typeof(▶op), typeof(grid)}(op, a, b, ▶a, ▶b, ▶op, grid)
    end
end

"""Create a binary operation for `op` acting on `a` and `b` with locations `La` and `Lb`.
The operator acts at `Lab` and the result is interpolated to `Lc`."""
function _binary_operation(Lc, op, a, b, La, Lb, Lab, grid)
     ▶a = interpolation_operator(La, Lab)
     ▶b = interpolation_operator(Lb, Lab)
    ▶op = interpolation_operator(Lab, Lc)
    return BinaryOperation{Lc[1], Lc[2], Lc[3]}(op, data(a), data(b), ▶a, ▶b, ▶op, grid)
end

@inline Base.getindex(β::BinaryOperation, i, j, k) = β.▶op(i, j, k, β.grid, β.op, β.▶a, β.▶b, β.a, β.b)

"""Return an expression that defines an abstract `BinaryOperator` named `op` for `AbstractLocatedField`."""
function define_binary_operator(op)
    return quote
        import Oceananigans

        local location = Oceananigans.location
        local FunctionField = Oceananigans.AbstractOperations.FunctionField
        local ALF = Oceananigans.AbstractLocatedField

        @inline $op(i, j, k, grid::Oceananigans.AbstractGrid, ▶a, ▶b, a, b) =
            @inbounds $op(▶a(i, j, k, grid, a), ▶b(i, j, k, grid, b))

        """
            $($op)(Lc, Lab, a, b)

        Returns an abstract representation of the operator `$($op)` acting on `a` and `b` at
        location `Lab`, and subsequently interpolated to location `Lc`.
        """
        function $op(Lc::Tuple, Lop::Tuple, a, b)
            La = location(a)
            Lb = location(b)
            grid = Oceananigans.AbstractOperations.validate_grid(a, b)
            return Oceananigans.AbstractOperations._binary_operation(Lc, $op, a, b, La, Lb, Lop, grid)
        end

        $op(Lc::Tuple, a, b) = $op(Lc, Lc, a, b)
        $op(Lc::Tuple, a::Number, b) = $op(Lc, location(b), a, b)
        $op(Lc::Tuple, a, b::Number) = $op(Lc, location(a), a, b)
        $op(Lc::Tuple, a::ALF{X, Y, Z}, b::ALF{X, Y, Z}) where {X, Y, Z} = $op(Lc, location(a), a, b)

        # Sugar for mixing in functions of (x, y, z)
        $op(Lc::Tuple, a::Function, b::Oceananigans.AbstractField) = $op(Lc, FunctionField(Lc, a, b.grid), b)
        $op(Lc::Tuple, a::Oceananigans.AbstractField, b::Function) = $op(Lc, a, FunctionField(Lc, b, a.grid))

        # Sugary versions with default locations
        $op(a::ALF, b::ALF) = $op(location(a), a, b)
        $op(a::ALF, b::Number) = $op(location(a), a, b)
        $op(a::Number, b::ALF) = $op(location(b), a, b)

        $op(a::ALF, b::Function) = $op(location(a), a, FunctionField(location(a), b, a.grid))
        $op(a::Function, b::ALF) = $op(location(b), FunctionField(location(b), a, b.grid), b)
    end
end

"""
    @binary op1 op2 op3...

Turn each binary function in the list `(op1, op2, op3...)`
into a binary operator on `Oceananigans.Fields` for use in `AbstractOperations`.

Note: a binary function is a function with two arguments: for example, `+(x, y)` is a binary function.

Also note: a binary function in `Base` must be imported to be extended: use `import Base: op; @binary op`.

Example
=======

```jldoctest
julia> plus_or_times(x, y) = x < 0 ? x + y : x * y
plus_or_times (generic function with 1 method)

julia> @binary plus_or_times
6-element Array{Any,1}:
 :+
 :-
 :/
 :^
 :*
 :plus_or_times

julia> c, d = (Field(Cell, Cell, Cell, CPU(), RegularCartesianGrid((1, 1, 16), (1, 1, 1))) for i = 1:2);

julia> plus_or_times(c, d)
BinaryOperation at (Cell, Cell, Cell)
├── grid: RegularCartesianGrid{Float64,StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}}
│   ├── size: (1, 1, 16)
│   └── domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [0.0, -1.0]
└── tree:

plus_or_times at (Cell, Cell, Cell) via Oceananigans.AbstractOperations.identity
├── OffsetArrays.OffsetArray{Float64,3,Array{Float64,3}}
└── OffsetArrays.OffsetArray{Float64,3,Array{Float64,3}}
"""
macro binary(ops...)
    expr = Expr(:block)

    for op in ops
        defexpr = define_binary_operator(op)
        push!(expr.args, :($(esc(defexpr))))

        add_to_operator_lists = quote
            push!(Oceananigans.AbstractOperations.operators, Symbol($op))
            push!(Oceananigans.AbstractOperations.binary_operators, Symbol($op))
        end

        push!(expr.args, :($(esc(add_to_operator_lists))))
    end

    return expr
end

const binary_operators = Set()

"Adapt `BinaryOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, binary::BinaryOperation{X, Y, Z}) where {X, Y, Z} =
    BinaryOperation{X, Y, Z}(adapt(to, binary.op), adapt(to, binary.a), adapt(to, binary.b),
                             adapt(to, binary.▶a), adapt(to, binary.▶b), adapt(to, binary.▶op),
                             binary.grid)
