struct PolynaryOperation{X, Y, Z, N, O, A, I, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
    args :: A
       ▶ :: I
    grid :: G

    function PolynaryOperation{X, Y, Z}(op, a, ▶, grid) where {X, Y, Z}
        return new{X, Y, Z, length(a), typeof(op), typeof(a), typeof(▶), typeof(grid)}(op, a, ▶, grid)
    end
end

function _polynary_operation(L, op, a, Largs, grid) where {X, Y, Z}
    ▶ = Tuple(interpolation_operator(Li, L) for Li in Largs)
    return PolynaryOperation{L[1], L[2], L[3]}(op, a, ▶, grid)
end

@inline Base.getindex(Π::PolynaryOperation{X, Y, Z, N}, i, j, k)  where {X, Y, Z, N} =
    Π.op(ntuple(γ -> Π.▶[γ](i, j, k, Π.grid, Π.args[γ]), Val(N))...)

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

Adapt.adapt_structure(to, polynary::PolynaryOperation{X, Y, Z}) where {X, Y, Z} =
    PolynaryOperation{X, Y, Z}(adapt(to, polynary.op), adapt(to, polynary.args), 
                               adapt(to, polynary.▶), polynary.grid)
