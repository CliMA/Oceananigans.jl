const multiary_operators = Set()

struct MultiaryOperation{X, Y, Z, N, O, A, I, R, G, T} <: AbstractOperation{X, Y, Z, R, G, T}
              op :: O
            args :: A
               ▶ :: I
    architecture :: R
            grid :: G

    function MultiaryOperation{X, Y, Z}(op::O, args::A, ▶::I, arch::R, grid::G) where {X, Y, Z, O, A, I, R, G}
        T = eltype(grid)
        N = length(args)
        return new{X, Y, Z, N, O, A, I, R, G, T}(op, args, ▶, arch, grid)
    end
end

@inline Base.getindex(Π::MultiaryOperation{X, Y, Z, N}, i, j, k)  where {X, Y, Z, N} =
    Π.op(ntuple(γ -> Π.▶[γ](i, j, k, Π.grid, Π.args[γ]), Val(N))...)

#####
##### MultiaryOperation construction
#####

function _multiary_operation(L, op, args, Largs, grid) where {X, Y, Z}
    ▶ = Tuple(interpolation_operator(La, L) for La in Largs)
    arch = architecture(args...)
    return MultiaryOperation{L[1], L[2], L[3]}(op, Tuple(a for a in args), ▶, arch, grid)
end

# Recompute location of multiary operation
@inline at(loc, Π::MultiaryOperation) = Π.op(loc, Tuple(at(loc, a) for a in Π.args)...)

"""Return an expression that defines an abstract `MultiaryOperator` named `op` for `AbstractField`."""
function define_multiary_operator(op)
    return quote
        function $op(Lop::Tuple,
                     a::Union{Function, Number, Oceananigans.Fields.AbstractField},
                     b::Union{Function, Number, Oceananigans.Fields.AbstractField},
                     c::Union{Function, Number, Oceananigans.Fields.AbstractField},
                     d::Union{Function, Number, Oceananigans.Fields.AbstractField}...)

            args = tuple(a, b, c, d...)
            grid = Oceananigans.AbstractOperations.validate_grid(args...)

            # Convert any functions to FunctionFields
            args = Tuple(Oceananigans.Fields.fieldify(Lop, a, grid) for a in args)
            Largs = Tuple(Oceananigans.Fields.location(a) for a in args)

            return Oceananigans.AbstractOperations._multiary_operation(Lop, $op, args, Largs, grid)
        end

        $op(a::Oceananigans.Fields.AbstractField,
            b::Union{Function, Oceananigans.Fields.AbstractField},
            c::Union{Function, Oceananigans.Fields.AbstractField},
            d::Union{Function, Oceananigans.Fields.AbstractField}...) = $op(Oceananigans.Fields.location(a), a, b, c, d...)
    end
end

"""
    @multiary op1 op2 op3...

Turn each multiary operator in the list `(op1, op2, op3...)`
into a multiary operator on `Oceananigans.Fields` for use in `AbstractOperations`.

Note that a multiary operator:
  * is a function with two or more arguments: for example, `+(x, y, z)` is a multiary function;
  * must be imported to be extended if part of `Base`: use `import Base: op; @multiary op`;
  * can only be called on `Oceananigans.Field`s if the "location" is noted explicitly; see example.

Example
=======

```jldoctest
julia> using Oceananigans, Oceananigans.AbstractOperations

julia> harmonic_plus(a, b, c) = 1/3 * (1/a + 1/b + 1/c)
harmonic_plus (generic function with 1 method)

julia> c, d, e = Tuple(Field(Center, Center, Center, CPU(), RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))) for i = 1:3);

julia> harmonic_plus(c, d, e) # before magic @multiary transformation
BinaryOperation at (Center, Center, Center)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
│   └── domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [-1.0, 0.0]
└── tree:
    * at (Center, Center, Center)
    ├── 0.3333333333333333
    └── + at (Center, Center, Center)
        ├── / at (Center, Center, Center)
        │   ├── 1
        │   └── Field located at (Center, Center, Center)
        ├── / at (Center, Center, Center)
        │   ├── 1
        │   └── Field located at (Center, Center, Center)
        └── / at (Center, Center, Center)
            ├── 1
            └── Field located at (Center, Center, Center)

julia> @multiary harmonic_plus
Set{Any} with 3 elements:
  :+
  :harmonic_plus
  :*

julia> harmonic_plus(c, d, e)
MultiaryOperation at (Center, Center, Center)
├── grid: RectilinearGrid{Float64, Periodic, Periodic, Bounded}(Nx=1, Ny=1, Nz=1)
│   └── domain: x ∈ [0.0, 1.0], y ∈ [0.0, 1.0], z ∈ [-1.0, 0.0]
└── tree:
    harmonic_plus at (Center, Center, Center)
    ├── Field located at (Center, Center, Center)
    ├── Field located at (Center, Center, Center)
    └── Field located at (Center, Center, Center)
```
"""
macro multiary(ops...)
    expr = Expr(:block)

    for op in ops
        defexpr = define_multiary_operator(op)
        push!(expr.args, :($(esc(defexpr))))

        add_to_operator_lists = quote
            push!(Oceananigans.AbstractOperations.operators, Symbol($op))
            push!(Oceananigans.AbstractOperations.multiary_operators, Symbol($op))
        end

        push!(expr.args, :($(esc(add_to_operator_lists))))
    end

    return expr
end

#####
##### Architecture inference for MultiaryOperation
#####

architecture(Π::MultiaryOperation) = Π.architecture

function architecture(a, b, c, d...)

    archs = []

    push!(archs, architecture(a, b))
    push!(archs, architecture(a, c))
    append!(archs, [architecture(a, di) for di in d])

    for arch in archs
        if !(arch === nothing)
            return arch
        end
    end

    return nothing
end

#####
##### Nested computations
#####

function compute_at!(Π::MultiaryOperation, time)
    c = Tuple(compute_at!(a, time) for a in Π.args)
    return nothing
end

#####
##### GPU capabilities
#####

"Adapt `MultiaryOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, multiary::MultiaryOperation{X, Y, Z}) where {X, Y, Z} =
    MultiaryOperation{X, Y, Z}(Adapt.adapt(to, multiary.op), Adapt.adapt(to, multiary.args),
                               Adapt.adapt(to, multiary.▶),  nothing, Adapt.adapt(to, multiary.grid))

