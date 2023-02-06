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
        T = eltype(grid)
        return new{LX, LY, LZ, O, A, B, IA, IB, G, T}(op, a, b, ▶a, ▶b, grid)
    end
end

@inline Base.getindex(β::BinaryOperation, i, j, k) = β.op(i, j, k, β.grid, β.▶a, β.▶b, β.a, β.b)

#####
##### BinaryOperation construction
#####

# Recompute location of binary operation
@inline at(loc, β::BinaryOperation) = β.op(loc, at(loc, β.a), at(loc, β.b))

indices(β::BinaryOperation) = interpolate_indices(β.a, β.b; loc_operation = location(β))

"""Create a binary operation for `op` acting on `a` and `b` at `Lc`, where
`a` and `b` have location `La` and `Lb`."""
function _binary_operation(Lc, op, a, b, La, Lb, grid)
     ▶a = interpolation_operator(La, Lc)
     ▶b = interpolation_operator(Lb, Lc)

    return BinaryOperation{Lc[1], Lc[2], Lc[3]}(op, a, b, ▶a, ▶b, grid)
end

const ConcreteLocationType = Union{Type{Face}, Type{Center}}

# Precedence rules for choosing operation location:
choose_location(La, Lb, Lc) = Lc                                    # Fallback to the specification Lc, but also...
choose_location(::Type{Face},   ::Type{Face},   Lc) = Face          # keep common locations; and
choose_location(::Type{Center}, ::Type{Center}, Lc) = Center        #
choose_location(La::ConcreteLocationType, ::Type{Nothing}, Lc) = La # don't interpolate unspecified locations.
choose_location(::Type{Nothing}, Lb::ConcreteLocationType, Lc) = Lb #

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
            @inbounds $op(▶a(i, j, k, grid, a), ▶b(i, j, k, grid, b))

        # These shenanigans seem to help / encourage the compiler to infer types of objects
        # buried in deep AbstractOperations trees.
        @inline function $op(i, j, k, grid::AbstractGrid, ▶a, ▶b, A::BinaryOperation, B::BinaryOperation)
            @inline a(ii, jj, kk, grid) = A.op(A.▶a(ii, jj, kk, grid, A.a), A.▶b(ii, jj, kk, grid, A.b))
            @inline b(ii, jj, kk, grid) = B.op(B.▶a(ii, jj, kk, grid, B.a), B.▶b(ii, jj, kk, grid, B.b))
            return @inbounds $op(▶a(i, j, k, grid, a), ▶b(i, j, k, grid, b))
        end

        @inline function $op(i, j, k, grid::AbstractGrid, ▶a, ▶b, A::BinaryOperation, B::AbstractField)
            @inline a(ii, jj, kk, grid) = A.op(A.▶a(ii, jj, kk, grid, A.a), A.▶b(ii, jj, kk, grid, A.b))
            return @inbounds $op(▶a(i, j, k, grid, a), ▶b(i, j, k, grid, B))
        end

        @inline function $op(i, j, k, grid::AbstractGrid, ▶a, ▶b, A::AbstractField, B::BinaryOperation)
            @inline b(ii, jj, kk, grid) = B.op(B.▶a(ii, jj, kk, grid, B.a), B.▶b(ii, jj, kk, grid, B.b))
            return @inbounds $op(▶a(i, j, k, grid, A), ▶b(i, j, k, grid, b))
        end

        @inline function $op(i, j, k, grid::AbstractGrid, ▶a, ▶b, A::BinaryOperation, B::Number)
            @inline a(ii, jj, kk, grid) = A.op(A.▶a(ii, jj, kk, grid, A.a), A.▶b(ii, jj, kk, grid, A.b))
            return @inbounds $op(▶a(i, j, k, grid, a), B)
        end

        @inline function $op(i, j, k, grid::AbstractGrid, ▶a, ▶b, A::Number, B::BinaryOperation)
            @inline b(ii, jj, kk, grid) = B.op(B.▶a(ii, jj, kk, grid, B.a), B.▶b(ii, jj, kk, grid, B.b))
            return @inbounds $op(A, ▶b(i, j, k, grid, b))
        end

        """
            $($op)(Lc, a, b)

        Return an abstract representation of the operator `$($op)` acting on `a` and `b`.
        The operation occurs at `location(a)` except for Nothing dimensions. In that case,
        the location of the dimension in question is supplied either by `location(b)` or
        if that is also Nothing, `Lc`.
        """
        function $op(Lc::Tuple, a, b)
            La = location(a)
            Lb = location(b)
            Lab = choose_location.(La, Lb, Lc)

            grid = Oceananigans.AbstractOperations.validate_grid(a, b)

            return Oceananigans.AbstractOperations._binary_operation(Lab, $op, a, b, La, Lb, grid)
        end

        # Numbers are not fields...
        $op(Lc::Tuple, a::Number, b::Number) = $op(a, b)

        # Sugar for mixing in functions of (x, y, z)
        $op(Lc::Tuple, f::Function, b::AbstractField) = $op(Lc, FunctionField(location(b), f, b.grid), b)
        $op(Lc::Tuple, a::AbstractField, f::Function) = $op(Lc, a, FunctionField(location(a), f, a.grid))

        $op(Lc::Tuple, m::AbstractGridMetric, b::AbstractField) = $op(Lc, GridMetricOperation(location(b), m, b.grid), b)
        $op(Lc::Tuple, a::AbstractField, m::AbstractGridMetric) = $op(Lc, a, GridMetricOperation(location(a), m, a.grid))

        # Sugary versions with default locations
        $op(a::AF, b::AF) = $op(location(a), a, b)
        $op(a::AF, b) = $op(location(a), a, b)
        $op(a, b::AF) = $op(location(b), a, b)

        $op(a::AF, b::Number) = $op(location(a), a, b)
        $op(a::Number, b::AF) = $op(location(b), a, b)

        $op(a::AF, b::ConstantField) = $op(location(a), a, b.constant)
        $op(a::ConstantField, b::AF) = $op(location(b), a.constant, b)

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

julia> using Oceananigans.AbstractOperations: BinaryOperation, AbstractGridMetric, choose_location

julia> plus_or_times(x, y) = x < 0 ? x + y : x * y
plus_or_times (generic function with 1 method)

julia> @binary plus_or_times
Set{Any} with 6 elements:
  :+
  :/
  :^
  :-
  :*
  :plus_or_times

julia> c, d = (CenterField(RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))) for i = 1:2);

julia> plus_or_times(c, d)
BinaryOperation at (Center, Center, Center)
├── grid: 1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
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
            push!(Oceananigans.AbstractOperations.operators, Symbol($op))
            push!(Oceananigans.AbstractOperations.binary_operators, Symbol($op))
        end

        push!(expr.args, :($(esc(add_to_operator_lists))))
    end

    return expr
end

#####
##### BinaryOperations with ZeroField
#####

import Base: +, -, *, /, ==
using Oceananigans.Fields: ZeroField, ConstantField

==(::ZeroField, ::ZeroField) = true

==(zf::ZeroField, cf::ConstantField) = 0 == cf.constant
==(cf::ConstantField, zf::ZeroField) = 0 == cf.constant
==(c1::ConstantField, c2::ConstantField) = c1.constant == c2.constant

+(a::ZeroField, b::AbstractField) = b
+(a::AbstractField, b::ZeroField) = a
+(a::ZeroField, b::Number) = ConstantField(b)
+(a::Number, b::ZeroField) = ConstantField(a)

-(a::ZeroField, b::AbstractField) = -b
-(a::AbstractField, b::ZeroField) = a
-(a::ZeroField, b::Number) = ConstantField(-b)
-(a::Number, b::ZeroField) = ConstantField(a)

*(a::ZeroField, b::AbstractField) = a
*(a::AbstractField, b::ZeroField) = b
*(a::ZeroField, b::Number) = a
*(a::Number, b::ZeroField) = b

/(a::ZeroField, b::AbstractField) = a
/(a::AbstractField, b::ZeroField) = ConstantField(convert(eltype(a), Inf))
/(a::ZeroField, b::Number) = a
/(a::Number, b::ZeroField) = ConstantField(a / convert(eltype(a), 0))


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

"Adapt `BinaryOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, binary::BinaryOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    BinaryOperation{LX, LY, LZ}(Adapt.adapt(to, binary.op),
                                Adapt.adapt(to, binary.a),
                                Adapt.adapt(to, binary.b),
                                Adapt.adapt(to, binary.▶a),
                                Adapt.adapt(to, binary.▶b),
                                Adapt.adapt(to, binary.grid))
