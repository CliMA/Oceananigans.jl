struct PolynaryOperation{X, Y, Z, N, A, B, IA, IB, LA, LB, O, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
      ▶a :: IA
      La :: LA
    grid :: G
    function PolynaryOperation{X, Y, Z}(op, a, ▶a) where {X, Y, Z}
        length(a) === length(▶) || throw(ArgumentError("Length of a and associated interpolation operators
                                                        ▶a must be the same."))

        grid = validate_grid(a...)

        a_data = Tuple(data(ai) for ai in a)
        La = Tuple(location(ai) for ai in a)

        return new{X, Y, Z, length(a), typeof(a_data), typeof(▶a), typeof(La)}(op, a, ▶a, La, grid)
    end
end

@propagate_inbounds function getindex(Π::PolynaryOperation{X, Y, Z, N}, i, j, k)  where {X, Y, Z, N}
    return Π.op(ntuple(i -> Π.▶a[i](i, j, k, Π.grid, Π.a[i]), N)...)
end

for op in (:+, :*)
    @eval begin
        function $op(a::F, b::F, c::F...) where F<:AbstractLocatedField
            d = merge((a, b), Tuple(c))
            L = merge((La, Lb), Tuple(location(ci) for ci in c))
            ▶ = Tuple(interp_operator(Li, L[1]) for Li in L)
            return PolynaryOperation{L[1][1], L[1][2], L[1][3]}($op, d, ▶)
        end

        function $op(Lop::Tuple, a::F, b::F, c::F...) where F<:AbstractLocatedField
            d = merge((a, b), Tuple(c))
            L = merge((La, Lb), Tuple(location(ci) for ci in c))
            ▶ = Tuple(interp_operator(Li, Lop) for Li in L)
            return BinaryOperation{Lop[1], Lop[2], Lop[3]}($op, a, b, ▶a, ▶b)
        end
    end
end

