import Krylov
import Krylov.FloatOrComplex

using KernelAbstractions: @index, @kernel

function LinearAlgebra.norm(a::AbstractField; condition = nothing)
    conditional_a = condition_operand(a, condition, 0)
    result = zeros(a.grid, 1)
    Base.mapreducedim!(x -> x * x, +, result, conditional_a)
    return CUDA.@allowscalar sqrt(first(result))
end

function LinearAlgebra.dot(a::AbstractField, b::AbstractField)
    conditional_a = condition_operand(a, condition, 0)
    conditional_b = condition_operand(b, condition, 0)
    result = zeros(a.grid, 1)
    Base.mapreducedim!((x, y) -> x * y, +, result, conditional_a, conditional_b)
    return CUDA.@allowscalar first(result)
end

struct KrylovField{T, F <: AbstractField} <: AbstractVector{T}
    field::F
end

function KrylovField(field::F) where F <: AbstractField
    T = eltype(field)
    return KrylovField{T,F}(field)
end

function Base.similar(kf::KrylovField)
    field = similar(kf.field)
    KrylovField(field)
end

function Base.isempty(kf::KrylovField)
    bool = isempty(kf.field)
    return bool
end

Base.size(kf::KrylovField) = size(kf.field)
Base.length(kf::KrylovField) = length(kf.field)
Base.getindex(kf::KrylovField, i::Int) = getindex(kf.field, i)

function Krylov.kscal!(n::Integer, s::T, x::KrylovField{T}) where T <: FloatOrComplex
    xp = parent(x.field)
    xp .*= s
    return x
end

function Krylov.kaxpy!(n::Integer, s::T, x::KrylovField{T}, y::KrylovField{T}) where T <: FloatOrComplex
    xp = parent(x.field)
    yp = parent(y.field)
    yp .+= s .* xp
    return y
end

function Krylov.kaxpby!(n::Integer, s::T, x::KrylovField{T}, t::T, y::KrylovField{T}) where T <: FloatOrComplex
    xp = parent(x.field)
    yp = parent(y.field)
    yp .= s .* xp .+ t .* yp
    return y
end

Krylov.knorm(n::Integer, x::KrylovField{T}) where T <: FloatOrComplex = norm(x.field)
Krylov.kdot(n::Integer, x::KrylovField{T}, y::KrylovField{T}) where T <: FloatOrComplex = dot(x.field, y.field)
Krylov.kcopy!(n::Integer, y::KrylovField{T}, x::KrylovField{T}) where T <: FloatOrComplex = copyto!(y.field, x.field)
Krylov.kfill!(x::KrylovField{T}, val::T) where T <: FloatOrComplex = fill!(x.field, val)

# Only needed by the Krylov solver MINRES-QLP.
# We can implement a kernel if we need it.
#
# function Krylov.kref!(n::Integer, x::KrylovField{T}, y::KrylovField{T}, c::T, s::T) where T <: FloatOrComplex
#     mx, nx, kx = size(x.field)
#     _x = x.field
#     _y = y.field
#     for i = 1:mx
#         for j = 1:nx
#             for k = 1:kx
#                 x_ijk = _x[i,j,k]
#                 y_ijk = _y[i,j,k]
#                 _x[i,j,k] = c       * x_ijk + s * y_ijk
#                 _y[i,j,k] = conj(s) * x_ijk - c * y_ijk
#             end
#         end
#     end
#     return x, y
# end
