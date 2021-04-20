const binary_operators = Set()

struct BinaryOperation{X, Y, Z, O, A, B, IA, IB, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
       b :: B
      ▶a :: IA
      ▶b :: IB
    grid :: G

    """
        BinaryOperation{X, Y, Z}(op, a, b, ▶a, ▶b, grid)

    Returns an abstract representation of the binary operation `op(▶a(a), ▶b(b))`.
    where `▶a` and `▶b` interpolate `a` and `b` to (X, Y, Z).
    """
    function BinaryOperation{X, Y, Z}(op, a, b, ▶a, ▶b, grid) where {X, Y, Z}

        any((X, Y, Z) .=== Nothing) && throw(ArgumentError("Nothing locations are invalid! " *
                                                           "Cannot construct BinaryOperation at ($X, $Y, $Z)."))

        return new{X, Y, Z, typeof(op), typeof(a), typeof(b), typeof(▶a), typeof(▶b),
                   typeof(grid)}(op, a, b, ▶a, ▶b, grid)
    end
end

@inline Base.getindex(β::BinaryOperation, i, j, k) = @inbounds β.op(β.▶a(i, j, k, β.grid, β.a), β.▶b(i, j, k, β.grid, β.b))

#####
##### BinaryOperation construction
#####

"""Create a binary operation for `op` acting on `a` and `b` with locations `La` and `Lb`.
The operator acts at `Lab` and the result is interpolated to `Lc`."""
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
        local AF = AbstractField

        """
            $($op)(Lc, a, b)

        Returns an abstract representation of the operator `$($op)` acting on `a` and `b`.
        The operation occurs at location(a) except for Nothing dimensions. In that case,
        the location of the dimension in question is supplied either by location(b) or
        if that is also Nothing, Lc.
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

        # Sugary versions with default locations
        $op(a::AF, b::AF) = $op(location(a), a, b)
        $op(a::AF, b::Number) = $op(location(a), a, b)
        $op(a::Number, b::AF) = $op(location(b), a, b)

        $op(a::AF, b::Function) = $op(location(a), a, FunctionField(location(a), b, a.grid))
        $op(a::Function, b::AF) = $op(location(b), FunctionField(location(b), a, b.grid), b)
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
julia> using Oceananigans, Oceananigans.AbstractOperations, Oceananigans.Grids

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

julia> c, d = (Field(Center, Center, Center, CPU(), RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))) for i = 1:2);

julia> plus_or_times(c, d)
BinaryOperation at (Center, Center, Center)
├── grid: RegularRectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
│   └── domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [-1.0, 0.0]
└── tree:
    plus_or_times at (Center, Center, Center) via identity
    ├── Field located at (Center, Center, Center)
    └── Field located at (Center, Center, Center)
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
##### Architecture inference for BinaryOperation
#####

architecture(β::BinaryOperation) = architecture(β.a, β.b)

function architecture(a, b)
    arch_a = architecture(a)
    arch_b = architecture(b)

    arch_a === arch_b && return arch_a
    isnothing(arch_a) && return arch_b
    isnothing(arch_b) && return arch_a

    throw(ArgumentError("Operation involves fields on architectures $arch_a and $arch_b"))

    return nothing
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

"Adapt `BinaryOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, binary::BinaryOperation{X, Y, Z}) where {X, Y, Z} =
    BinaryOperation{X, Y, Z}(Adapt.adapt(to, binary.op), Adapt.adapt(to, binary.a),  Adapt.adapt(to, binary.b),
                             Adapt.adapt(to, binary.▶a), Adapt.adapt(to, binary.▶b), Adapt.adapt(to, binary.grid))
                             
