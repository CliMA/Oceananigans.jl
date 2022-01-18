const unary_operators = Set()

struct UnaryOperation{LX, LY, LZ, O, A, I, G, T} <: AbstractOperation{LX, LY, LZ, G, T}
    op :: O
    arg :: A
    ▶ :: I
    grid :: G

    @doc """
        UnaryOperation{LX, LY, LZ}(op, arg, ▶, grid)

    Returns an abstract `UnaryOperation` representing the action of `op` on `arg`,
    and subsequent interpolation by `▶` on `grid`.
    """
    function UnaryOperation{LX, LY, LZ}(op::O, arg::A, ▶::I, grid::G) where {LX, LY, LZ, O, A, I, G}
        T = eltype(grid)
        return new{LX, LY, LZ, O, A, I, G, T}(op, arg, ▶, grid)
    end
end

@inline Base.getindex(υ::UnaryOperation, i, j, k) = υ.▶(i, j, k, υ.grid, υ.op, υ.arg)

#####
##### UnaryOperation construction
#####

"""Create a unary operation for `operator` acting on `arg` which interpolates the
result from `Larg` to `L`."""
function _unary_operation(L, operator, arg, Larg, grid)
    ▶ = interpolation_operator(Larg, L)
    return UnaryOperation{L[1], L[2], L[3]}(operator, arg, ▶, grid)
end

# Recompute location of unary operation
@inline at(loc, υ::UnaryOperation) = υ.op(loc, at(loc, υ.arg))

"""
    @unary op1 op2 op3...

Turn each unary function in the list `(op1, op2, op3...)`
into a unary operator on `Oceananigans.Fields` for use in `AbstractOperations`.

Note: a unary function is a function with one argument: for example, `sin(x)` is a unary function.

Also note: a unary function in `Base` must be imported to be extended: use `import Base: op; @unary op`.

Example
=======

```jldoctest
julia> using Oceananigans, Oceananigans.Grids, Oceananigans.AbstractOperations

julia> square_it(x) = x^2
square_it (generic function with 1 method)

julia> @unary square_it
Set{Any} with 8 elements:
  :sqrt
  :square_it
  :cos
  :exp
  :interpolate_identity
  :-
  :tanh
  :sin

julia> c = CenterField(RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)));

julia> square_it(c)
UnaryOperation at (Center, Center, Center)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
│   └── domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [-1.0, 0.0]
└── tree:
    square_it at (Center, Center, Center) via identity
    └── Field located at (Center, Center, Center)
```
"""
macro unary(ops...)
    expr = Expr(:block)

    for op in ops
        define_unary_operator = quote
            import Oceananigans.Grids: AbstractGrid
            import Oceananigans.Fields: AbstractField

            local location = Oceananigans.Fields.location

            @inline $op(i, j, k, grid::AbstractGrid, a) = @inbounds $op(a[i, j, k])
            @inline $op(i, j, k, grid::AbstractGrid, a::Number) = $op(a)

            """
                $($op)(Lop::Tuple, a::AbstractField)

            Returns an abstract representation of the operator `$($op)` acting on the Oceananigans `Field`
            `a`, and subsequently interpolated to the location indicated by `Lop`.
            """
            function $op(Lop::Tuple, a::AbstractField)
                L = location(a)
                return Oceananigans.AbstractOperations._unary_operation(Lop, $op, a, L, a.grid)
            end

            $op(a::AbstractField) = $op(location(a), a)

            push!(Oceananigans.AbstractOperations.operators, Symbol($op))
            push!(Oceananigans.AbstractOperations.unary_operators, Symbol($op))
        end

        push!(expr.args, :($(esc(define_unary_operator))))
    end

    return expr
end

#####
##### Nested computations
#####

compute_at!(υ::UnaryOperation, time) = compute_at!(υ.arg, time)

#####
##### GPU capabilities
#####

"Adapt `UnaryOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, unary::UnaryOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    UnaryOperation{LX, LY, LZ}(Adapt.adapt(to, unary.op),
                               Adapt.adapt(to, unary.arg),
                               Adapt.adapt(to, unary.▶),
                               Adapt.adapt(to, unary.grid))

