struct PolynaryOperation{X, Y, Z, N, O, A, I, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
    args :: A
       ▶ :: I
    grid :: G

    function PolynaryOperation{X, Y, Z}(op, args, ▶, grid) where {X, Y, Z}
        return new{X, Y, Z, length(args), typeof(op), typeof(args), typeof(▶), typeof(grid)}(op, args, ▶, grid)
    end
end

function _polynary_operation(L, op, args, Largs, grid) where {X, Y, Z}
    ▶ = Tuple(interpolation_operator(La, L) for La in Largs)
    return PolynaryOperation{L[1], L[2], L[3]}(op, Tuple(data(a) for a in args), ▶, grid)
end

@inline Base.getindex(Π::PolynaryOperation{X, Y, Z, N}, i, j, k)  where {X, Y, Z, N} =
    Π.op(ntuple(γ -> Π.▶[γ](i, j, k, Π.grid, Π.args[γ]), Val(N))...)

"""Return `a`, or convert `a` to `FunctionField` if `a::Function`"""
fieldify(L, a, grid) = a
fieldify(L, a::Function, grid) = FunctionField(L, a, grid)

"""Return an expression that defines an abstract `PolynaryOperator` named `op` for `AbstractLocatedField`."""
function define_polynary_operator(op)
    return quote
        import Oceananigans

        local location = Oceananigans.location

        # "Function, or Field"
        local FuFi = Union{Function, Oceananigans.AbstractField}

        function $op(Lop::Tuple, a::FuFi, b::FuFi, c::FuFi...)
            args = tuple(a, b, c...)
            grid = Oceananigans.AbstractOperations.validate_grid(args...)

            # Convert any functions to FunctionFields
            args = Tuple(Oceananigans.AbstractOperations.fieldify(Lop, a, grid) for a in args)
            Largs = Tuple(location(a) for a in args)

            return Oceananigans.AbstractOperations._polynary_operation(Lop, $op, args, Largs, grid)
        end

        $op(a::FuFi, b::FuFi, c::FuFi...) = $op(location(a), a, b, c...)
    end
end

"""
    @polynary op1 op2 op3...

Turn each polynary operator in the list `(op1, op2, op3...)` 
into a polynary operator on `Oceananigans.Fields` for use in `AbstractOperations`. 

Note that a polynary operator:
    * is a function with two or more arguments: for example, `+(x, y, z)` is a polynary function;
    * must be imported to be extended if part of `Base`: use `import Base: op; @polynary op`;
    * can only be called on `Oceananigans.Field`s if the "location" is noted explicitly; see example.

Example
=======

```jldoctest
julia> harmonic_plus(a, b, c) = 1/3 * (1/a + 1/b + 1/c)
harmonic_plus(generic function with 1 method)

julia> @polynary harmonic_plus
3-element Array{Any,1}:
 :+
 :*
 :harmonic_plus

julia> c, d, e = Tuple(Field(Cell, Cell, Cell, CPU(), RegularCartesianGrid((1, 1, 16), (1, 1, 1))) for i = 1:3);

julia> harmonic_plus(c, d, e) # this calls the original function, which in turn returns a (correct) operation tree
BinaryOperation at (Cell, Cell, Cell)
├── grid: RegularCartesianGrid{Float64,StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}}
│   ├── size: (1, 1, 16)
│   └── domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [0.0, -1.0]
└── tree:

* at (Cell, Cell, Cell) via Oceananigans.AbstractOperations.identity
├── 0.3333333333333333
└── + at (Cell, Cell, Cell) via Oceananigans.AbstractOperations.identity
    ├── + at (Cell, Cell, Cell) via Oceananigans.AbstractOperations.identity
    │   ├── / at (Cell, Cell, Cell) via Oceananigans.AbstractOperations.identity
    │   │   ├── 1
    │   │   └── OffsetArrays.OffsetArray{Float64,3,Array{Float64,3}}
    │   └── / at (Cell, Cell, Cell) via Oceananigans.AbstractOperations.identity
        │   ├── 1
        │   └── OffsetArrays.OffsetArray{Float64,3,Array{Float64,3}}
    └── / at (Cell, Cell, Cell) via Oceananigans.AbstractOperations.identity
        ├── 1
        └── OffsetArrays.OffsetArray{Float64,3,Array{Float64,3}}

julia> @at (Cell, Cell, Cell) harmonic_plus(c, d, e) # this returns a `PolynaryOperation` as expected
PolynaryOperation at (Cell, Cell, Cell)
├── grid: RegularCartesianGrid{Float64,StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}}
│   ├── size: (1, 1, 16)
│   └── domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [0.0, -1.0]
└── tree: 

harmonic_plus at (Cell, Cell, Cell)
├── OffsetArrays.OffsetArray{Float64,3,Array{Float64,3}}
├── OffsetArrays.OffsetArray{Float64,3,Array{Float64,3}}
└── OffsetArrays.OffsetArray{Float64,3,Array{Float64,3}}
"""
macro polynary(ops...)
    expr = Expr(:block)

    for op in ops
        defexpr = define_polynary_operator(op)
        push!(expr.args, :($(esc(defexpr))))

        add_to_operator_lists = quote
            Oceananigans.AbstractOperations.uniquepush!(Oceananigans.AbstractOperations.operators, Symbol($op))
            Oceananigans.AbstractOperations.uniquepush!(Oceananigans.AbstractOperations.polynary_operators, Symbol($op))
        end

        push!(expr.args, :($(esc(add_to_operator_lists))))
    end

    return expr
end

const polynary_operators = []

"Adapt `PolynaryOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, polynary::PolynaryOperation{X, Y, Z}) where {X, Y, Z} =
    PolynaryOperation{X, Y, Z}(adapt(to, polynary.op), adapt(to, polynary.args), 
                               adapt(to, polynary.▶), polynary.grid)
