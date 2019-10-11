struct BinaryOperation{X, Y, Z, A, B, IA, IB, LA, LB, O, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
       b :: B
      ▶a :: IA
      ▶b :: IB
      La :: LA
      Lb :: LB
    grid :: G
    function BinaryOperation{X, Y, Z}(op, a, b, ▶a, ▶b) where {X, Y, Z}
        a.grid === b.grid || throw(ArgumentError("Both fields in a BinaryOperation must be on the same grid."))

        La = location(a)
        Lb = location(b)

        return new{X, Y, Z, typeof(data(a)), typeof(data(b)), 
                   typeof(▶a), typeof(▶b), typeof(La), typeof(Lb),
                   typeof(op), typeof(a.grid)}(op, data(a), data(b), ▶a, ▶b, La, Lb, a.grid)
    end
end

@propagate_inbounds function getindex(β::BinaryOperation, i, j, k) 
    return β.op(β.▶a(i, j, k, β.grid, β.a), β.▶b(i, j, k, β.grid, β.b))
end

interp_code(::Type{Face}) = :f
interp_code(::Type{Cell}) = :c
interp_code(to::L, from::L) where L = :a
interp_code(to, from) = interp_code(to)

for ξ in (:x, :y, :z)
    ▶sym = Symbol(:▶, ξ, :sym)
    @eval begin
        $▶sym(s::Symbol) = $▶sym(Val(s))
        $▶sym(::Union{Val{:f}, Val{:c}}) = string(ξ)
        $▶sym(::Val{:a}) = ""
    end
end

@inline identity(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]

function interp_operator(to, from)
    x, y, z = (interp_code(t, f) for (t, f) in zip(to, from))

    if all(ξ === :a for ξ in (x, y, z))
        return identity
    else 
        return eval(Symbol(:▶, ▶xsym(x), ▶ysym(y), ▶zsym(z), :_, x, y, z))
    end
end

for op in (:+, :-, :/, :*)
    @eval begin
        function $op(a::F, b::F) where F<:AbstractLocatedField
            La = location(a)
            Lb = location(b)
            ▶a = identity
            ▶b = interp_operator(La, Lb)
            return BinaryOperation{XA, YA, ZA}($op, a, b, ▶a, ▶b)
        end

        function $op(Lop::Tuple, a::F, b::F) where F<:AbstractLocatedField
            La = location(a)
            Lb = location(b)
            ▶a = interp_operator(Lop, La)
            ▶b = interp_operator(Lop, Lb)
            return BinaryOperation{Lop[1], Lop[2], Lop[3]}($op, a, b, ▶a, ▶b)
        end
    end
end

