struct UnaryOperation{X, Y, Z, O, A, I, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
     arg :: A
       ▶ :: I
    grid :: G

    function UnaryOperation{X, Y, Z}(op, arg, ▶, grid) where {X, Y, Z}
        return new{X, Y, Z, typeof(op), typeof(arg), typeof(▶), typeof(grid)}(op, arg, ▶, grid)
    end
end

function _unary_operation(L, op, arg, Larg, grid) where {X, Y, Z}
    ▶ = interpolation_operator(Larg, L)
    return UnaryOperation{L[1], L[2], L[3]}(op, data(arg), ▶, grid)
end

@inline Base.getindex(υ::UnaryOperation, i, j, k) = υ.▶(i, j, k, υ.grid, υ.op, υ.arg)

const unary_operators = [:sqrt, :sin, :cos, :exp]
append!(operators, unary_operators)

"""
    @unary op

    @unary op1 op2 op3...

Turn the unary function `op`, or each unary function in the list `(op1, op2, op3...)` 
into a unary operator on `Oceananigans.Fields` for use in `AbstractOperations`. 

Note: a unary function is a function with one argument: for example, `sin(x)` is a unary function.

Also note: a unary function in `Base` must be imported to be extended: use `import Base: op; @unary op`.

Example
=======

julia> square_it(x) = x^2
square_it (generic function with 1 method)

julia> @unary square_it

julia> c = Field(Cell, Cell, Cell, CPU(), RegularCartesianGrid((1, 1, 16), (1, 1, 1)))
Field at (Cell, Cell, Cell)
├── data: OffsetArrays.OffsetArray{Float64,3,Array{Float64,3}}
└── grid: RegularCartesianGrid{Float64,StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}}
    ├── size: (1, 1, 16)
    └── domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [0.0, -1.0]

julia> square_it(c)
UnaryOperation at (Cell, Cell, Cell)
├── grid: RegularCartesianGrid{Float64,StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}}
│   ├── size: (1, 1, 16)
│   └── domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [0.0, -1.0]
└── tree:

square_it at (Cell, Cell, Cell) via identity
└── OffsetArrays.OffsetArray{Float64,3,Array{Float64,3}}

"""
macro unary(op)
    esc(quote
        import Oceananigans

        @inline $op(i, j, k, grid::Oceananigans.AbstractGrid, a) = @inbounds $op(a[i, j, k])
        @inline $op(i, j, k, grid::Oceananigans.AbstractGrid, a::Number) = $op(a)

        function $op(Lop::Tuple, a::Oceananigans.AbstractLocatedField)
            L = Oceananigans.location(a)
            return Oceananigans.AbstractOperations._unary_operation(Lop, $op, a, L, a.grid)
        end

        $op(a::Oceananigans.AbstractLocatedField) = $op(Oceananigans.location(a), a)

        push!(Oceananigans.AbstractOperations.operators, $op)

        nothing
    end)
end

macro unary(op1, ops...)
    expr = Expr(:block)
    push!(expr.args, :(@unary $(esc(op1));))
    append!(expr.args, [:(@unary $(esc(op));) for op in ops])
    return expr
end

# Make some unary operators
import Base: sqrt, sin, cos, exp, tanh, -

@unary sqrt sin cos exp tanh
@unary -

Adapt.adapt_structure(to, unary::UnaryOperation{X, Y, Z}) where {X, Y, Z} =
    UnaryOperation{X, Y, Z}(adapt(to, unary.op), adapt(to, unary.arg), 
                            adapt(to, unary.▶), unary.grid)

function tree_show(unary::UnaryOperation{X, Y, Z}, depth, nesting)  where {X, Y, Z}
    padding = "    "^(depth-nesting) * "│   "^nesting

    return string(unary.op, " at ", show_location(X, Y, Z), " via ", unary.▶, '\n',
                  padding, "└── ", tree_show(unary.arg, depth+1, nesting))
end

