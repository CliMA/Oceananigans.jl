module AbstractOperations

export ∂x, ∂y, ∂z

using Base: @propagate_inbounds

using Oceananigans

using Oceananigans: Face, Cell, AbstractLocatedField

import Oceananigans.TurbulenceClosures: ∂x_caa, ∂x_faa, ∂y_aca, ∂y_afa, ∂z_aac, ∂z_aaf, 
                                        ▶x_caa, ▶x_faa, ▶y_aca, ▶y_afa, ▶z_aac, ▶z_aaf

import Base: *, -, +, /, getindex

abstract type AbstractOperation{X, Y, Z, G} <: AbstractLocatedField{X, Y, Z, Nothing, G} end

data(op::AbstractOperation) = op
alldata(op::AbstractOperation) = op
alldata(a::Field) = a.data

@inline ∂x_caa(i, j, k, grid, u::AbstractOperation) = @inbounds (u[i+1, j, k] - u[i, j, k]) / grid.Δx
@inline ∂x_faa(i, j, k, grid, c::AbstractOperation) = @inbounds (c[i, j, k] - c[i-1, j, k]) / grid.Δx

@inline ∂y_aca(i, j, k, grid, v::AbstractOperation) = @inbounds (v[i, j+1, k] - v[i, j, k]) / grid.Δy
@inline ∂y_afa(i, j, k, grid, c::AbstractOperation) = @inbounds (c[i, j, k] - c[i, j-1, k]) / grid.Δy

@inline ∂z_aac(i, j, k, grid, w::AbstractOperation) = @inbounds (w[i, j, k] - w[i, j, k+1]) / grid.Δz
@inline ∂z_aaf(i, j, k, grid, c::AbstractOperation) = @inbounds (c[i, j, k-1] - c[i, j, k]) / grid.Δz

@inline ▶x_faa(i, j, k, grid::RegularCartesianGrid{FT}, F::AbstractOperation, args...) where FT =
    @inbounds FT(0.5) * (F[i, j, k] + F[i-1, j, k])

@inline ▶x_caa(i, j, k, grid::RegularCartesianGrid{FT}, F::AbstractOperation, args...) where FT =
    @inbounds FT(0.5) * (F[i, j, k] + F[i+1, j, k])

@inline ▶y_afa(i, j, k, grid::RegularCartesianGrid{FT}, F::AbstractOperation, args...) where FT =
    @inbounds FT(0.5) * (F[i, j, k] + F[i, j-1, k])

@inline ▶y_aca(i, j, k, grid::RegularCartesianGrid{FT}, F::AbstractOperation, args...) where FT =
    @inbounds FT(0.5) * (F[i, j, k] + F[i, j+1, k])

@inline ▶z_aaf(i, j, k, grid::RegularCartesianGrid{FT}, F::AbstractOperation, args...) where FT =
    @inbounds FT(0.5) * (F[i, j, k] + F[i, j, k-1])

@inline ▶z_aac(i, j, k, grid::RegularCartesianGrid{FT}, w::AbstractOperation, args...) where FT =
    @inbounds FT(0.5) * (w[i, j, k] + w[i, j, k+1])

#####
##### Type for binary operations
#####

struct BinaryOperation{X, Y, Z, A, B, IA, IB, O, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
       b :: B
      ▶a :: IA
      ▶b :: IB
    grid :: G
    function BinaryOperation{X, Y, Z}(op, a, b, ▶a, ▶b) where {X, Y, Z}
        @assert a.grid === b.grid
        return new{X, Y, Z, typeof(alldata(a)), typeof(alldata(b)), typeof(▶a), typeof(▶b), 
                   typeof(op), typeof(a.grid)}(op, alldata(a), alldata(b), ▶a, ▶b, a.grid)
    end
end

@propagate_inbounds function getindex(β::BinaryOperation, i, j, k) 
    return β.op(β.▶a(i, j, k, β.grid, β.a), β.▶b(i, j, k, β.grid, β.b))
end

interp_code(::Type{Face}) = :f
interp_code(::Type{Cell}) = :c
interp_code(to::L, from::L) where L = :a
interp_code(to, from) = interp_code(to)

flip(::Type{Face}) = Cell
flip(::Type{Cell}) = Face

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

#####
##### Derivative operator
#####

struct Derivative{X, Y, Z, A, D, G} <: AbstractOperation{X, Y, Z, G}
       a :: A
       ∂ :: D
    grid :: G
    function Derivative{X, Y, Z}(a, ∂) where {X, Y, Z}
        return new{X, Y, Z, typeof(alldata(a)), typeof(∂), typeof(a.grid)}(alldata(a), ∂, a.grid)
    end
end

data(bop::BinaryOperation) = bop

function ∂x(a::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}
    ∂ = eval(Symbol(:∂x_, interp_code(flip(X)), :aa))
    return Derivative{flip(X), Y, Z}(a, ∂)
end

function ∂y(a::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}
    ∂ = eval(Symbol(:∂y_a, interp_code(flip(Y)), :a))
    return Derivative{X, flip(Y), Z}(a, ∂)
end

function ∂z(a::AbstractLocatedField{X, Y, Z}) where {X, Y, Z}
    ∂ = eval(Symbol(:∂z_aa, interp_code(flip(Z))))
    return Derivative{X, Y, flip(Z)}(a, ∂)
end

@propagate_inbounds getindex(d::Derivative, i, j, k) = d.∂(i, j, k, d.grid, d.a)

end # module
