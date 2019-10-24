"""
    UnaryOperation{X, Y, Z, O, A, I, G} <: AbstractOperation{X, Y, Z, G}

An abstract representation of a unary operation on an `AbstractField`; or a function
`f(x)` with on argument acting on `x::AbstractField`.
"""
struct UnaryOperation{X, Y, Z, O, A, I, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
     arg :: A
       ▶ :: I
    grid :: G

    """
        UnaryOperation{X, Y, Z}(op, arg, ▶, grid)

    Returns an abstract `UnaryOperation` representing the action of `op` on `arg`,
    and subsequent interpolation by `▶` on `grid`.
    """
    function UnaryOperation{X, Y, Z}(op, arg, ▶, grid) where {X, Y, Z}
        return new{X, Y, Z, typeof(op), typeof(arg), typeof(▶), typeof(grid)}(op, arg, ▶, grid)
    end
end

"""Create a unary operation for `operator` acting on `arg` which interpolates the
result from `Larg` to `L`."""
function _unary_operation(L, operator, arg, Larg, grid)
    ▶ = interpolation_operator(Larg, L)
    return UnaryOperation{L[1], L[2], L[3]}(operator, data(arg), ▶, grid)
end

@inline Base.getindex(υ::UnaryOperation, i, j, k) = υ.▶(i, j, k, υ.grid, υ.op, υ.arg)

"""
    @unary op1 op2 op3...

Turn each unary function in the list `(op1, op2, op3...)` 
into a unary operator on `Oceananigans.Fields` for use in `AbstractOperations`. 

Note: a unary function is a function with one argument: for example, `sin(x)` is a unary function.

Also note: a unary function in `Base` must be imported to be extended: use `import Base: op; @unary op`.

Example
=======

julia> square_it(x) = x^2
square_it (generic function with 1 method)

julia> @unary square_it
7-element Array{Any,1}:
 :sqrt
 :sin
 :cos
 :exp
 :tanh
 :-
 :square_it

julia> c = Field(Cell, Cell, Cell, CPU(), RegularCartesianGrid((1, 1, 16), (1, 1, 1)));

julia> square_it(c)
UnaryOperation at (Cell, Cell, Cell)
├── grid: RegularCartesianGrid{Float64,StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}}
│   ├── size: (1, 1, 16)
│   └── domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [0.0, -1.0]
└── tree:

square_it at (Cell, Cell, Cell) via identity
└── OffsetArrays.OffsetArray{Float64,3,Array{Float64,3}}
"""
macro unary(ops...)
    expr = Expr(:block)

    for op in ops
        define_unary_operator = quote
            import Oceananigans

            @inline $op(i, j, k, grid::Oceananigans.AbstractGrid, a) = @inbounds $op(a[i, j, k])
            @inline $op(i, j, k, grid::Oceananigans.AbstractGrid, a::Number) = $op(a)

            """
                $($op)(Lop::Tuple, a::Oceananigans.AbstractLocatedField)

            Returns an abstract representation of the operator `$($op)` acting on the Oceananigans `Field`
            `a`, and subsequently interpolated to the location indicated by `Lop`.
            """
            function $op(Lop::Tuple, a::Oceananigans.AbstractLocatedField)
                L = Oceananigans.location(a)
                return Oceananigans.AbstractOperations._unary_operation(Lop, $op, a, L, a.grid)
            end

            $op(a::Oceananigans.AbstractLocatedField) = $op(Oceananigans.location(a), a)

            push!(Oceananigans.AbstractOperations.operators, Symbol($op))
            push!(Oceananigans.AbstractOperations.unary_operators, Symbol($op))
        end

        push!(expr.args, :($(esc(define_unary_operator))))
    end

    return expr
end

const unary_operators = Set()

"Adapt `UnaryOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, unary::UnaryOperation{X, Y, Z}) where {X, Y, Z} =
    UnaryOperation{X, Y, Z}(adapt(to, unary.op), adapt(to, unary.arg), 
                            adapt(to, unary.▶), unary.grid)
