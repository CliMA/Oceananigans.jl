struct BinaryOperation{X, Y, Z, A, B, IA, IB, O, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
       b :: B
      ▶a :: IA
      ▶b :: IB
    grid :: G
    function BinaryOperation{X, Y, Z}(op, a, b, ▶a, ▶b) where {X, Y, Z}
        @assert a.grid === b.grid
        return new{X, Y, Z, typeof(data(a)), typeof(data(b)), typeof(▶a), typeof(▶b), 
                   typeof(op), typeof(a.grid)}(op, data(a), data(b), ▶a, ▶b, a.grid)
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
        function $op(a::AbstractLocatedField{XA, YA, ZA}, 
                     b::AbstractLocatedField{XB, YB, ZB}) where {XA, YA, ZA, XB, YB, ZB}
            ▶a = identity
            ▶b = interp_operator((XA, YA, ZA), (XB, YB, ZB))
            return BinaryOperation{XA, YA, ZA}($op, a, b, ▶a, ▶b)
        end
    end
end

