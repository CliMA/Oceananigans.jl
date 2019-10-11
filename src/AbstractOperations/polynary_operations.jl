struct PolynaryOperation{X, Y, Z, N, A, I, L, G, O} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
       ▶ :: I
      La :: L
    grid :: G
    function PolynaryOperation{X, Y, Z}(op, a, L, grid) where {X, Y, Z}
        ▶ = Tuple(interpolation_operator(Li, (X, Y, Z)) for Li in L)
        return new{X, Y, Z, length(a), typeof(a), typeof(▶), typeof(L), 
                   typeof(grid), typeof(op)}(op, a, ▶, L, grid)
    end
end

@propagate_inbounds getindex(Π::PolynaryOperation{X, Y, Z, N}, i, j, k)  where {X, Y, Z, N} =
    Π.op(ntuple(γ -> Π.▶[γ](i, j, k, Π.grid, Π.a[γ]), N)...)

const polynary_operators = [:+, :*]
append!(operators, polynary_operators)

for op in (:+, :*)
    @eval begin
        import Base: $op

        function $op(Lop::Tuple, b, c, d...)
            a = tuple(b, c, d...)
            L = tuple(location(b), location(c), (location(di) for di in d)...)
            grid = validate_grid(a...)
            return PolynaryOperation{Lop[1], Lop[2], Lop[3]}($op, a, L, grid)
        end

        $op(b::ALF, c::ALF, d::ALF...)    = $op(location(b), b, c, d...)
        $op(b::ALF, c::Number, d::ALF...) = $op(location(b), b, c, d...)
        $op(b::Number, c::ALF, d::ALF...) = $op(location(c), b, c, d...)
    end
end

Adapt.adapt_structure(to, polynary::PolynaryOperation{X, Y, Z}) where {X, Y, Z} =
    PolynaryOperation{X, Y, Z}(polynary.op, adapt(to, polynary.a), polynary.La, polynary.grid)
