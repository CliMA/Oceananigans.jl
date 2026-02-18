const binary_operators = Set()

struct BinaryOperation{LX, LY, LZ, O, A, B, IA, IB, G, T} <: AbstractOperation{LX, LY, LZ, G, T}
    op :: O
    a :: A
    b :: B
    ▶a :: IA
    ▶b :: IB
    grid :: G

    @doc """
        BinaryOperation{LX, LY, LZ}(op, a, b, ▶a, ▶b, grid)

    Return an abstract representation of the binary operation `op(▶a(a), ▶b(b))` on
    `grid`, where `▶a` and `▶b` interpolate `a` and `b` to locations `(LX, LY, LZ)`.
    """
    function BinaryOperation{LX, LY, LZ}(op::O, a::A, b::B, ▶a::IA, ▶b::IB, grid::G) where {LX, LY, LZ, O, A, B, IA, IB, G}
        T = Base.promote_op(op, eltype(a), eltype(b))
        return new{LX, LY, LZ, O, A, B, IA, IB, G, T}(op, a, b, ▶a, ▶b, grid)
    end
end

@inline Base.getindex(β::BinaryOperation, i, j, k) = β.op(i, j, k, β.grid, β.▶a, β.▶b, β.a, β.b)

#####
##### BinaryOperation construction
#####

# Recompute location of binary operation
@inline at(loc, β::BinaryOperation) = β.op(loc, at(loc, β.a), at(loc, β.b))

indices(β::BinaryOperation) = construct_regionally(intersect_indices, location(β), β.a, β.b)

"""Create a binary operation for `op` acting on `a` and `b` at `Lc`, where
`a` and `b` have location `La` and `Lb`."""
function _binary_operation(Lc::Tuple{LX, LY, LZ}, op, a, b, La, Lb, grid) where {LX<:Location, LY<:Location, LZ<:Location}
    ▶a = interpolation_operator(La, Lc)
    ▶b = interpolation_operator(Lb, Lc)

    return BinaryOperation{LX, LY, LZ}(op, a, b, ▶a, ▶b, grid)
end

const ConcreteLocationType = Union{Face, Center}

# Precedence rules for choosing operation location:
choose_location(La, Lb, Lc) = Lc                              # Fallback to the specification Lc, but also...
choose_location(::Face,   ::Face,   Lc) = Face()              # keep common locations; and
choose_location(::Center, ::Center, Lc) = Center()            #
choose_location(La::ConcreteLocationType, ::Nothing, Lc) = La # don't interpolate unspecified locations.
choose_location(::Nothing, Lb::ConcreteLocationType, Lc) = Lb #

# Apply the function if the inputs are scalars, otherwise broadcast it over the inputs
# This can occur in the binary operator code if we index into with an array, e.g. array[1:10]
@inline @propagate_inbounds apply_op(op, a, b) = op(a, b)
@inline @propagate_inbounds apply_op(op, a::AbstractArray, b::AbstractArray) = op.(a, b)

"""Return an expression that defines an abstract `BinaryOperator` named `op` for `AbstractField`."""
function define_binary_operator(op)
    return quote
        import Oceananigans.Grids: AbstractGrid
        import Oceananigans.Fields: AbstractField

        local location = Oceananigans.Fields.location
        local FunctionField = Oceananigans.Fields.FunctionField
        local ConstantField = Oceananigans.Fields.ConstantField
        local AF = AbstractField

        @inline $op(i, j, k, grid::AbstractGrid, ▶a, ▶b, a, b) =
            @inbounds apply_op($op, ▶a(i, j, k, grid, a), ▶b(i, j, k, grid, b))

        # These shenanigans seem to help / encourage the compiler to infer types of objects
        # buried in deep AbstractOperations trees.
        @inline function $op(i, j, k, grid::AbstractGrid, ▶a, ▶b, A::BinaryOperation, B::BinaryOperation)
            @inline a(ii, jj, kk, grid) = apply_op(A.op, A.▶a(ii, jj, kk, grid, A.a), A.▶b(ii, jj, kk, grid, A.b))
            @inline b(ii, jj, kk, grid) = apply_op(B.op, B.▶a(ii, jj, kk, grid, B.a), B.▶b(ii, jj, kk, grid, B.b))
            return @inbounds apply_op($op, ▶a(i, j, k, grid, a), ▶b(i, j, k, grid, b))
        end

        @inline function $op(i, j, k, grid::AbstractGrid, ▶a, ▶b, A::BinaryOperation, B::AbstractField)
            @inline a(ii, jj, kk, grid) = apply_op(A.op, A.▶a(ii, jj, kk, grid, A.a), A.▶b(ii, jj, kk, grid, A.b))
            return @inbounds apply_op($op, ▶a(i, j, k, grid, a), ▶b(i, j, k, grid, B))
        end

        @inline function $op(i, j, k, grid::AbstractGrid, ▶a, ▶b, A::AbstractField, B::BinaryOperation)
            @inline b(ii, jj, kk, grid) = apply_op(B.op, B.▶a(ii, jj, kk, grid, B.a), B.▶b(ii, jj, kk, grid, B.b))
            return @inbounds apply_op($op, ▶a(i, j, k, grid, A), ▶b(i, j, k, grid, b))
        end

        @inline function $op(i, j, k, grid::AbstractGrid, ▶a, ▶b, A::BinaryOperation, B::Number)
            @inline a(ii, jj, kk, grid) = apply_op(A.op, A.▶a(ii, jj, kk, grid, A.a), A.▶b(ii, jj, kk, grid, A.b))
            return @inbounds apply_op($op, ▶a(i, j, k, grid, a), B)
        end

        @inline function $op(i, j, k, grid::AbstractGrid, ▶a, ▶b, A::Number, B::BinaryOperation)
            @inline b(ii, jj, kk, grid) = apply_op(B.op, B.▶a(ii, jj, kk, grid, B.a), B.▶b(ii, jj, kk, grid, B.b))
            return @inbounds apply_op($op, A, ▶b(i, j, k, grid, b))
        end

        """
            $($op)(Lc, a, b)

        Return an abstract representation of the operator `$($op)` acting on `a` and `b`.
        The operation occurs at `location(a)` except for Nothing dimensions. In that case,
        the location of the dimension in question is supplied either by `location(b)` or
        if that is also Nothing, `Lc`.
        """
        function $op(Lc::Tuple{<:$Location, <:$Location, <:$Location}, a, b)
            La = Oceananigans.Fields.instantiated_location(a)
            Lb = Oceananigans.Fields.instantiated_location(b)
            Lab = choose_location.(La, Lb, Lc)

            grid = $(validate_grid)(a, b)

            return $(_binary_operation)(Lab, $op, a, b, La, Lb, grid)
        end

        # Numbers are not fields...
        $op(Lc::Tuple{<:$Location, <:$Location, <:$Location}, a::Number, b::Number) = $op(a, b)

        # Sugar for mixing in functions of (x, y, z)
        $op(Lc::Tuple{<:$Location, <:$Location, <:$Location}, f::Function, b::AbstractField) = $op(Lc, FunctionField(location(b), f, b.grid), b)
        $op(Lc::Tuple{<:$Location, <:$Location, <:$Location}, a::AbstractField, f::Function) = $op(Lc, a, FunctionField(location(a), f, a.grid))

        $op(Lc::Tuple{<:$Location, <:$Location, <:$Location}, m::GridMetric, b::AbstractField) = $op(Lc, grid_metric_operation(Oceananigans.Fields.instantiated_location(b), m, b.grid), b)
        $op(Lc::Tuple{<:$Location, <:$Location, <:$Location}, a::AbstractField, m::GridMetric) = $op(Lc, a, grid_metric_operation(Oceananigans.Fields.instantiated_location(a), m, a.grid))

        # instantiate location if types are passed
        $op(Lc::Tuple, a, b) = $op((Lc[1](), Lc[2](), Lc[3]()), a, b)

        $op(Lc::Tuple, f::Function, b::AbstractField) = $op((Lc[1](), Lc[2](), Lc[3]()), f, b)
        $op(Lc::Tuple, a::AbstractField, f::Function) = $op((Lc[1](), Lc[2](), Lc[3]()), a, f)

        $op(Lc::Tuple, m::GridMetric, b::AbstractField) = $op((Lc[1](), Lc[2](), Lc[3]()), m, b)
        $op(Lc::Tuple, a::AbstractField, m::GridMetric) = $op((Lc[1](), Lc[2](), Lc[3]()), a, m)

        # Sugary versions with default locations
        $op(a::AF, b::AF) = $op(Oceananigans.Fields.instantiated_location(a), a, b)
        $op(a::AF, b) = $op(Oceananigans.Fields.instantiated_location(a), a, b)
        $op(a, b::AF) = $op(Oceananigans.Fields.instantiated_location(b), a, b)

        $op(a::AF, b::Number) = $op(Oceananigans.Fields.instantiated_location(a), a, b)
        $op(a::Number, b::AF) = $op(Oceananigans.Fields.instantiated_location(b), a, b)

        $op(a::AF, b::ConstantField) = $op(Oceananigans.Fields.instantiated_location(a), a, b.constant)
        $op(a::ConstantField, b::AF) = $op(Oceananigans.Fields.instantiated_location(b), a.constant, b)

        $op(a::Number, b::ConstantField) = ConstantField($op(a, b.constant))
        $op(a::ConstantField, b::Number) = ConstantField($op(a.constant, b))
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
julia> using Oceananigans, Oceananigans.AbstractOperations

julia> using Oceananigans.AbstractOperations: BinaryOperation, GridMetric, choose_location

julia> plus_or_times(x, y) = x < 0 ? x + y : x * y
plus_or_times (generic function with 1 method)

julia> @binary plus_or_times;

julia> c, d = (CenterField(RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))) for i = 1:2);

julia> plus_or_times(c, d)
BinaryOperation at (Center, Center, Center)
├── grid: 1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
└── tree:
    plus_or_times at (Center, Center, Center)
    ├── 1×1×1 Field{Center, Center, Center} on RectilinearGrid on CPU
    └── 1×1×1 Field{Center, Center, Center} on RectilinearGrid on CPU
```
"""
macro binary(ops...)
    expr = Expr(:block)

    for op in ops
        defexpr = define_binary_operator(op)
        push!(expr.args, :($(esc(defexpr))))

        add_to_operator_lists = quote
            push!($(operators), Symbol($op))
            push!($(binary_operators), Symbol($op))
        end

        push!(expr.args, :($(esc(add_to_operator_lists))))
    end

    return expr
end

#####
##### Nested computations
#####

function compute_at!(β::BinaryOperation, time)
    compute_at!(β.a, time)
    compute_at!(β.b, time)
    return nothing
end

#####
##### GPU capabilities
#####

"Adapt `BinaryOperation` to work on the GPU via KernelAbstractions."
Adapt.adapt_structure(to, binary::BinaryOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    BinaryOperation{LX, LY, LZ}(Adapt.adapt(to, binary.op),
                                Adapt.adapt(to, binary.a),
                                Adapt.adapt(to, binary.b),
                                Adapt.adapt(to, binary.▶a),
                                Adapt.adapt(to, binary.▶b),
                                Adapt.adapt(to, binary.grid))


Architectures.on_architecture(to, binary::BinaryOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    BinaryOperation{LX, LY, LZ}(on_architecture(to, binary.op),
                                on_architecture(to, binary.a),
                                on_architecture(to, binary.b),
                                on_architecture(to, binary.▶a),
                                on_architecture(to, binary.▶b),
                                on_architecture(to, binary.grid))
