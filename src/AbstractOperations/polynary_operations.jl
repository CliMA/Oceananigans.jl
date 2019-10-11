struct PolynaryOperation{X, Y, Z, N, A, I, L, O, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
      ▶a :: I
      La :: L
    grid :: G
    function PolynaryOperation{X, Y, Z}(op, a, ▶a) where {X, Y, Z}
        length(a) === length(▶a) || throw(ArgumentError("Length of a and associated interpolation operators
                                                         ▶a must be the same."))

        grid = validate_grid(a...)

        a_data = Tuple(data(ai) for ai in a)
        La = Tuple(location(ai) for ai in a)

        return new{X, Y, Z, length(a), typeof(a_data), typeof(▶a), typeof(La), typeof(op), 
                   typeof(grid)}(op, a, ▶a, La, grid)
    end
end

@propagate_inbounds function getindex(Π::PolynaryOperation{X, Y, Z, N}, i, j, k)  where {X, Y, Z, N}
    return Π.op(ntuple(i -> Π.▶a[i](i, j, k, Π.grid, Π.a[i]), N)...)
end

for op in (:+, :*)
    @eval begin
        function $op(b::AbstractLocatedField, c::AbstractLocatedField, d::AbstractLocatedField...)
             a = merge((b, c), Tuple(d))
             L = merge((Lb, Lc), Tuple(location(di) for di in d))
            L1 = L[1]
             ▶ = Tuple(interp_operator(Li, L1) for Li in L)
            return PolynaryOperation{L1[1], L1[2], L1[3]}($op, a, ▶)
        end

        function $op(Lop::Tuple, b::AbstractLocatedField, c::AbstractLocatedField, d::AbstractLocatedField...)
            a = merge((b, c), Tuple(d))
            L = merge((Lb, Lc), Tuple(location(di) for di in d))
            ▶ = Tuple(interp_operator(Li, Lop) for Li in L)
            return PolynaryOperation{Lop[1], Lop[2], Lop[3]}($op, a, ▶)
        end
    end
end

