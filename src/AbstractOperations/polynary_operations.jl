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

#=
const polynary_operators = [:+, :*]
append!(operators, polynary_operators)

for op in (:+, :*)
    @eval begin
        import Base: $op

        function $op(Lop::Tuple, b, c, d...)
            a = tuple(b, c, d...)
            Largs = tuple(location(b), location(c), (location(di) for di in d)...)
            grid = validate_grid(a...)
            return _polynary_operation(Lop, $op, a, Largs, grid)
        end
    end
end
=#

macro polynary(ops...)
    expr = Expr(:block)

    for op in ops
        define_polynary_operator = quote
            import Oceananigans

            function $op(Lop::Tuple, a, b, c...)
                args = tuple(a, b, c...)
                Largs = tuple(Oceananigans.location(a), Oceananigans.location(b), 
                              (Oceananigans.location(ci) for ci in c)...)
                grid = Oceananigans.AbstractOperations.validate_grid(args...)
                return Oceananigans.AbstractOperations._polynary_operation(Lop, $op, args, Largs, grid)
            end

            push!(Oceananigans.AbstractOperations.polynary_operators, $op)
            push!(Oceananigans.AbstractOperations.operators, $op)
        end

        push!(expr.args, :($(esc(define_polynary_operator))))
    end

    push!(expr.args, :(nothing))
        
    return expr
end

const polynary_operators = []

import Base: +, *

@polynary +
@polynary *

"Adapt `PolynaryOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, polynary::PolynaryOperation{X, Y, Z}) where {X, Y, Z} =
    PolynaryOperation{X, Y, Z}(adapt(to, polynary.op), adapt(to, polynary.args), 
                               adapt(to, polynary.▶), polynary.grid)
